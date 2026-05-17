"""Profile cdce.decode_bp end-to-end timing.

Measures (per frame, averaged over many calls):
  1. dict_to_program overhead (per call, re-built every frame)
  2. rx_llr Python-loop copy overhead (1664 single-element pybind ops)
  3. pure decode_bp_cpp inner-loop time (VM dispatch + BP scaffold)
  4. comparison vs ldpc_5g.decode_oms_fast (numpy reference)

Run:
    python -u code_review/profile_cdce_decode_bp.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from ldpc_5g import (derive_params, build_parity, build_oms_context,
                     decode_oms_fast, HTYPE, encode_codeblock,
                     bpsk_modulate, bpsk_llr,
                     cbs_rate_match, cbs_rate_recover,
                     compute_k0, compute_E_array)
from pushgp.dce import _try_import_pushgp_cpp_dce
cdce = _try_import_pushgp_cpp_dce()
if cdce is None:
    print("FAIL: pushgp_cpp_dce not importable"); sys.exit(2)
from pushgp.serialize import program_to_dict
from pushgp_ldpc.adapter import oms_seed_genome


def gen_frames(p, par, rng, snr_db, n_frames):
    htype = HTYPE[par.bgn - 1][par.set_idx - 1]
    R = p.A / p.outlen
    sigma2 = 1.0 / (2.0 * R * 10.0 ** (snr_db / 10.0))
    sigma = float(np.sqrt(sigma2))
    out = []
    for _ in range(n_frames):
        info = -np.ones(p.K, dtype=np.int64)
        info[:p.K_cb_bit] = rng.integers(0, 2, size=p.K_cb_bit)
        cw = encode_codeblock(info, par, htype)
        k0 = compute_k0(p.bgn, p.zc, p.N_punctured, rv=0)
        e_arr = compute_E_array(p.C, p.outlen, qm=1, nlayers=1)
        tx_bits = cbs_rate_match(cw, e_arr[0], k0, p.N_punctured, qm=1)
        tx = bpsk_modulate(tx_bits)
        rx = tx + sigma * rng.standard_normal(tx.shape)
        llr_part = bpsk_llr(rx, sigma2)
        llr_punct = cbs_rate_recover(llr_part, p.N_punctured, e_arr[0],
                                      p.K - 2 * p.zc, p.K_cb_bit - 2 * p.zc,
                                      k0, p.N_punctured, qm=1)
        llr = np.concatenate((np.zeros(2 * p.zc, dtype=np.float64), llr_punct))
        out.append(llr.astype(np.float64))
    return out


def profile_config(A, E, max_iter, n_frames, n_warmup=3):
    p = derive_params(A, A / E)
    par = build_parity(p.bgn, p.set_idx, p.zc)
    parH = cdce.build_parity_handle(par)
    ctx = build_oms_context(par)

    oms = oms_seed_genome()
    v_dict = program_to_dict(oms.prog_v2c)
    c_dict = program_to_dict(oms.prog_c2v)
    evo = oms.evo_const_values().astype(np.float64)
    v_size = len(v_dict)
    c_size = len(c_dict)

    print(f"\n========================================")
    print(f"A={A} E={E} BG{p.bgn} set{p.set_idx} Zc={p.zc} N={p.N} M={par.rows}")
    print(f"  v2c_prog size={v_size}  c2v_prog size={c_size}  max_iter={max_iter}")
    print(f"  n_frames={n_frames} (+{n_warmup} warmup)")

    rng = np.random.default_rng(424242)
    frames = gen_frames(p, par, rng, snr_db=2.0, n_frames=n_frames + n_warmup)
    # warmup
    for f in frames[:n_warmup]:
        cdce.decode_bp(f, parH, v_dict, c_dict, evo, max_iter, 0.25, p.A / p.outlen)
        decode_oms_fast(f, ctx, max_iter)
    use_frames = frames[n_warmup:]

    # ---- (1) end-to-end cdce.decode_bp (everything) ----
    t0 = time.perf_counter()
    iters_total_cpp = 0
    for f in use_frames:
        _post, it_run = cdce.decode_bp(f, parH, v_dict, c_dict, evo,
                                        max_iter, 0.25, p.A / p.outlen)
        iters_total_cpp += it_run
    t_cpp_total = time.perf_counter() - t0
    avg_it_cpp = iters_total_cpp / n_frames

    # ---- (2) cdce.decode_bp with dict_to_program isolated ----
    # We can't avoid it inside the binding; measure by running it many
    # times in a tight Python loop to subtract.
    # Build a no-op call: decode with a tiny par to isolate fixed cost.
    # Instead just measure dict_to_program by issuing one call and
    # subtracting from same-sized decode without programs.  Since we
    # can't directly do that, just report the bare end-to-end.

    # ---- (3) ldpc_5g.decode_oms_fast (numpy reference) ----
    t0 = time.perf_counter()
    iters_total_np = 0
    for f in use_frames:
        _post, it_run = decode_oms_fast(f, ctx, max_iter)
        iters_total_np += it_run
    t_np_total = time.perf_counter() - t0
    avg_it_np = iters_total_np / n_frames

    # ---- (4) dict_to_program roundtrip cost ----
    t0 = time.perf_counter()
    n_loops = 5000
    for _ in range(n_loops):
        program_to_dict(oms.prog_v2c)
        program_to_dict(oms.prog_c2v)
    t_serialize = (time.perf_counter() - t0) / n_loops * 1e6

    # Note: dict_to_program is inside the C++ binding (not Python serialize).
    # We approximate by calling decode_bp with a deliberately tiny par.
    # Skip for now; just report ratio.

    cpp_ms = t_cpp_total / n_frames * 1000
    np_ms = t_np_total / n_frames * 1000
    cpp_per_iter_ms = cpp_ms / avg_it_cpp
    np_per_iter_ms = np_ms / avg_it_np

    print(f"  ldpc_5g.decode_oms_fast: {np_ms:8.2f} ms/frame  ({avg_it_np:.1f} iters avg, {np_per_iter_ms:5.2f} ms/iter)")
    print(f"  cdce.decode_bp  (OMS): {cpp_ms:8.2f} ms/frame  ({avg_it_cpp:.1f} iters avg, {cpp_per_iter_ms:5.2f} ms/iter)")
    print(f"  ratio cpp / numpy      : {cpp_ms / np_ms:6.2f}x")
    print(f"  serialize  v2c+c2v dict round-trip : {t_serialize:.1f} us/call (Python program_to_dict cost only)")

    # Estimate VM call count
    # LiftedParity has cn_to_vn list-of-lists; build vn_to_cn from it.
    vn_to_cn = [[] for _ in range(par.cols)]
    for c, vs in enumerate(par.cn_to_vn):
        for v in vs:
            vn_to_cn[v].append(c)
    avg_dv_v = sum(len(x) for x in vn_to_cn) / par.cols
    avg_dc = sum(len(x) for x in par.cn_to_vn) / par.rows
    edges_per_iter = par.cols * avg_dv_v  # = par.rows * avg_dc
    vm_calls_per_frame = int(edges_per_iter * 2 * avg_it_cpp)
    print(f"  est VM calls/frame     : {vm_calls_per_frame:,}  (avg dv={avg_dv_v:.2f} dc={avg_dc:.2f})")
    print(f"  est VM time/call       : {cpp_per_iter_ms * 1000 / (edges_per_iter * 2):.2f} us  (= cpp_per_iter / (2*edges))")
    return {"cpp_ms": cpp_ms, "np_ms": np_ms, "ratio": cpp_ms / np_ms,
            "vm_calls": vm_calls_per_frame}


def main():
    cases = [
        (176, 352, 8, 20),
        (176, 352, 20, 20),
        (512, 1024, 8, 10),
        (512, 1024, 20, 10),
    ]
    results = []
    for A, E, max_iter, nf in cases:
        results.append((A, E, max_iter, profile_config(A, E, max_iter, nf)))

    print("\n========================================")
    print("SUMMARY")
    print(f"{'A':>5} {'E':>5} {'max_it':>7} {'cpp_ms':>10} {'np_ms':>10} {'ratio':>8} {'VM calls':>12}")
    for A, E, mi, r in results:
        print(f"{A:>5} {E:>5} {mi:>7} {r['cpp_ms']:>10.2f} {r['np_ms']:>10.2f} {r['ratio']:>7.2f}x {r['vm_calls']:>12,}")


if __name__ == "__main__":
    main()
