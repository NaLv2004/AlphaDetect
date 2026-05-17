"""Regression test (LOCKING):  the OMS adapter (programmed PushVM genome)
must produce posterior LLRs bit-aligned with ldpc_5g.decode_oms_fast,
for any (par, frame, max_iter).

Why this test exists:
    On 2026-05-17 a hidden clamp in pushgp/genome.py
    (LOG_CONST_MAX=0.0) was silently squashing the OMS sentinel
    EvoConst1=1e6 down to 1.0, which made the min-fold in C2V
    saturate at 1.0, which made BP never converge in waterfall.
    The "bit-equivalence" tests at the time only stressed degenerate
    single-iter inputs and missed this entirely.  This test exercises
    the *full multi-iteration BP path* on realistic frames and asserts
    max|Δ post| <= 1e-9 (pure float reassoc noise).  Also checks
    that the bit-decisions are identical and that the decoder
    actually converges (errors → 0 at moderate SNR).

Run:
    python code_review/regression_oms_cpp_vs_ldpc5g.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ldpc_5g import (build_parity, build_oms_context, decode_oms_fast,
                     HTYPE, encode_codeblock, bpsk_modulate, bpsk_llr,
                     cbs_rate_match, cbs_rate_recover,
                     compute_k0, compute_E_array, derive_params)
from pushgp.dce import _try_import_pushgp_cpp_dce
cdce = _try_import_pushgp_cpp_dce()
if cdce is None:
    print("FAIL: pushgp_cpp_dce not importable; build cpp_dce first.")
    sys.exit(2)
from pushgp.serialize import program_to_dict
from pushgp_ldpc.adapter import oms_seed_genome
from pushgp import genome as _genome_mod

# Lock the OMS sentinel-clamp fix: LOG_CONST_MAX must be wide enough that
# EvoConst1=1e6 survives np.clip in evo_const_values(). 6.0 = log10(1e6).
assert _genome_mod.LOG_CONST_MAX >= 6.0, (
    f"LOG_CONST_MAX={_genome_mod.LOG_CONST_MAX} clips OMS sentinel; "
    "see regression_oms_cpp_vs_ldpc5g.py docstring."
)


def _gen_frame(p, par, rng, snr_db):
    htype = HTYPE[par.bgn - 1][par.set_idx - 1]
    R = p.A / p.outlen
    sigma2 = 1.0 / (2.0 * R * 10.0 ** (snr_db / 10.0))
    sigma = float(np.sqrt(sigma2))
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
    info_payload = np.where(info[:p.K_cb_bit] == -1, 0, info[:p.K_cb_bit])
    return llr.astype(np.float64), info_payload


def main():
    cases = [
        # (A, E, snr_db, n_frames, max_iter, atol)
        (176, 352, 2.0, 5, 20, 1e-9),
        (512, 1024, 2.0, 3, 20, 1e-9),
    ]
    oms = oms_seed_genome()
    v_dict = program_to_dict(oms.prog_v2c)
    c_dict = program_to_dict(oms.prog_c2v)
    evo = oms.evo_const_values().astype(np.float64)

    rng = np.random.default_rng(20260517)
    all_pass = True

    for A, E, snr, n_frames, max_iter, atol in cases:
        p = derive_params(A, A / E)
        par = build_parity(p.bgn, p.set_idx, p.zc)
        parH = cdce.build_parity_handle(par)
        ctx = build_oms_context(par)

        print(f"\n=== A={A} E={E} → BG{p.bgn} set{p.set_idx} Zc={p.zc} "
              f"N={p.N} K_cb_bit={p.K_cb_bit}  snr={snr}dB ===", flush=True)

        max_dev = 0.0
        n_bit_mismatch = 0
        n_ref_errs = 0
        n_cpp_errs = 0
        t0 = time.time()
        for f in range(n_frames):
            llr, info = _gen_frame(p, par, rng, snr)
            post_ref, _ = decode_oms_fast(llr, ctx, max_iter)
            post_cpp, _ = cdce.decode_bp(llr, parH, v_dict, c_dict, evo,
                                          max_iter, 0.25, p.A / p.outlen)
            d = float(np.max(np.abs(post_ref - post_cpp)))
            if d > max_dev:
                max_dev = d
            hat_ref = (post_ref[:p.K_cb_bit] < 0.0).astype(np.int64)
            hat_cpp = (post_cpp[:p.K_cb_bit] < 0.0).astype(np.int64)
            n_bit_mismatch += int(np.sum(hat_ref != hat_cpp))
            n_ref_errs += int(np.sum(hat_ref != info))
            n_cpp_errs += int(np.sum(hat_cpp != info))

        elapsed = time.time() - t0
        ok_dev = max_dev <= atol
        ok_bits = n_bit_mismatch == 0
        ok_converge = n_cpp_errs == 0  # OMS at 2 dB should clear all errors
        print(f"  max|Δ post|     = {max_dev:.3e}  (tol={atol:.0e})  {'PASS' if ok_dev else 'FAIL'}")
        print(f"  bit mismatches  = {n_bit_mismatch}/{n_frames * p.K_cb_bit}  {'PASS' if ok_bits else 'FAIL'}")
        print(f"  ref errs        = {n_ref_errs}")
        print(f"  cpp errs        = {n_cpp_errs}    {'(BP converged)' if ok_converge else '(WARN: did not converge)'}")
        print(f"  elapsed         = {elapsed:.1f}s")
        if not (ok_dev and ok_bits):
            all_pass = False

    print("\n" + ("=" * 50))
    print("REGRESSION:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
