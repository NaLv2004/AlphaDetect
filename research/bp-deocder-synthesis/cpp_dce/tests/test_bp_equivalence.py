"""Step 2 gate: cpp decode_bp must equal python ldpc_5g.decode_bp output
byte-for-byte (at 6-decimal precision, which is the DCE comparison rule).

Run:
    python cpp_dce/tests/test_bp_equivalence.py [n_progs] [n_frames] [seed]

Defaults: 100 program pairs × 5 frames at SNR=0 dB, max_iter=8.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

import pushgp_cpp_dce as cdce
from ldpc_5g import build_parity, decode_bp, HTYPE, bpsk_modulate, bpsk_llr
from pushgp.genome import Genome, program_to_list as program_to_dict
from pushgp.random_program import RandomProgramGenerator
from pushgp_ldpc.adapter import make_callables
from pushgp_ldpc.eval import _random_codeword


def _frames(par, n_frames, snr_db, seed):
    htype = HTYPE[par.bgn - 1][par.set_idx - 1]
    rate = 0.5
    rng = np.random.default_rng(seed)
    sigma2 = 1.0 / (2.0 * rate * 10.0 ** (snr_db / 10.0))
    sigma = float(np.sqrt(sigma2))
    out = []
    for _ in range(n_frames):
        cw = _random_codeword(par, htype, rng)
        tx = bpsk_modulate(cw[2 * par.zc:])
        rx = tx + sigma * rng.standard_normal(tx.shape)
        llr_part = bpsk_llr(rx, sigma2)
        llr = np.zeros(par.cols, dtype=np.float64)
        llr[2 * par.zc:] = llr_part
        out.append(llr)
    return out


def main(n_progs=100, n_frames=5, seed=4242) -> int:
    par = build_parity(2, 1, 2)         # N=104, M=84
    parH = cdce.build_parity_handle(par)
    max_iter = 8

    rpg = RandomProgramGenerator(rng=np.random.default_rng(seed))
    frames = _frames(par, n_frames, snr_db=0.0, seed=seed + 1)

    n_ok = 0
    n_finite_diff = 0
    n_mismatch = 0
    first_mismatch = None
    t_py = 0.0
    t_cpp = 0.0

    for k in range(n_progs):
        pv = rpg.random_v2c(min_size=4, max_size=20)
        pc = rpg.random_c2v(min_size=4, max_size=20)
        log_consts = rpg.random_log_constants() if hasattr(rpg, "random_log_constants") \
                     else np.zeros(8, dtype=np.float64)
        g = Genome(prog_v2c=pv, prog_c2v=pc, log_constants=log_consts.copy())
        try:
            v2c_fn, c2v_fn = make_callables(g)
        except Exception:
            continue

        pv_dict = program_to_dict(pv)
        pc_dict = program_to_dict(pc)
        evo = g.evo_const_values().astype(np.float64)

        for llr in frames:
            # Python reference
            t0 = time.perf_counter()
            try:
                post_py = decode_bp(
                    llr, par, v2c_fn=v2c_fn, c2v_fn=c2v_fn,
                    max_iter=max_iter, offset=0.25, code_rate=0.5,
                )
            except Exception:
                continue
            t_py += time.perf_counter() - t0

            # C++ implementation
            t0 = time.perf_counter()
            post_cpp, _it = cdce.decode_bp(
                llr, parH, pv_dict, pc_dict, evo,
                max_iter, 0.25, 0.5,
            )
            t_cpp += time.perf_counter() - t0

            n_ok += 1
            # 6-decimal comparison rule (same as DCE)
            eq6 = np.array_equal(np.round(post_py, 6), np.round(post_cpp, 6))
            if not eq6:
                n_mismatch += 1
                if first_mismatch is None:
                    diff = np.abs(post_py - post_cpp)
                    first_mismatch = (k, diff.max(), int(diff.argmax()),
                                      float(post_py[diff.argmax()]),
                                      float(post_cpp[diff.argmax()]))
            # Track soft float-equality too (informational)
            if not np.allclose(post_py, post_cpp, atol=1e-12, rtol=0):
                n_finite_diff += 1

    print(f"Programs : {n_progs}")
    print(f"Frames   : {n_frames}")
    print(f"Compared : {n_ok}")
    print(f"Mismatch@6dp : {n_mismatch}  ({100.0 * n_mismatch / max(1,n_ok):.2f}%)")
    print(f"Bit-bit diff : {n_finite_diff}  (informational)")
    if first_mismatch:
        k, dmax, idx, py_v, cpp_v = first_mismatch
        print(f"  first mismatch: prog#{k} max|diff|={dmax:.3e} "
              f"v={idx} py={py_v:.6f} cpp={cpp_v:.6f}")
    print(f"time(py)     : {t_py:.2f}s")
    print(f"time(cpp)    : {t_cpp:.2f}s")
    speedup = (t_py / t_cpp) if t_cpp > 0 else float('inf')
    print(f"speedup      : x{speedup:.2f}")

    if n_mismatch == 0 and n_ok > 0:
        print("\nStep 2 PASS: cpp decode_bp == python decode_bp at 6 decimals.")
        return 0
    print("\nStep 2 FAIL")
    return 1


if __name__ == "__main__":
    n_progs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 4242
    sys.exit(main(n_progs, n_frames, seed))
