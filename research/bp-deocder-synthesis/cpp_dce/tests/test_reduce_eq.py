"""Step 3 gate (T7): cpp behavioral_reduce_bp_cpp must produce the SAME
reduced program as python behavioral_reduce_bp for the same input.

Comparison: structural equality of program_to_list(reduced_py) vs
the dict returned by cdce.reduce_bp.

Run:
    python cpp_dce/tests/test_reduce_eq.py [n_pairs] [n_frames] [seed]
Defaults: 10 (v,c) pairs x 2 frames at SNR=0 dB, max_iter=4.
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
from ldpc_5g import build_parity, HTYPE, bpsk_modulate, bpsk_llr
from pushgp.dce import behavioral_reduce_bp, DCEStats
from pushgp.genome import Genome, program_to_list
from pushgp.random_program import RandomProgramGenerator
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


def main(n_pairs=10, n_frames=2, seed=8081) -> int:
    par = build_parity(2, 1, 2)
    parH = cdce.build_parity_handle(par)
    max_iter = 4

    rpg = RandomProgramGenerator(rng=np.random.default_rng(seed))
    rx_llrs = _frames(par, n_frames, snr_db=0.0, seed=seed + 1)

    n_ok = 0
    n_match = 0
    first_mismatch = None
    t_py = 0.0
    t_cpp = 0.0
    py_sizes_before, py_sizes_after, cpp_sizes_after = [], [], []

    for k in range(n_pairs):
        pv = rpg.random_v2c(min_size=4, max_size=12)
        pc = rpg.random_c2v(min_size=4, max_size=12)
        log_consts = rpg.random_log_constants() if hasattr(rpg, "random_log_constants") \
                     else np.zeros(8, dtype=np.float64)
        g = Genome(prog_v2c=pv, prog_c2v=pc, log_constants=log_consts.copy())
        evo = g.evo_const_values().astype(np.float64)

        # Reduce v2c side (peer = c2v)
        for side, prog, peer in [("v2c", pv, pc), ("c2v", pc, pv)]:
            try:
                # ----- python
                t0 = time.perf_counter()
                stats_py = DCEStats(side=side, size_before=0)
                red_py = behavioral_reduce_bp(
                    prog, side, peer_prog=peer,
                    log_constants=log_consts, par=par,
                    rx_llrs=rx_llrs, max_iter=max_iter,
                    max_passes=400, decimals=6, stats=stats_py,
                )
                t_py += time.perf_counter() - t0
                red_py_dict = program_to_list(red_py)

                # ----- cpp
                t0 = time.perf_counter()
                red_cpp_dict, stats_cpp = cdce.reduce_bp(
                    program_to_list(prog), side, program_to_list(peer),
                    parH, rx_llrs, evo,
                    max_iter, 400, -1, 6,
                )
                t_cpp += time.perf_counter() - t0
            except Exception as ex:
                print(f"[{k}/{side}] exception {ex!r}")
                continue

            n_ok += 1
            py_sizes_before.append(stats_py.size_before)
            py_sizes_after.append(stats_py.size_after)
            cpp_sizes_after.append(stats_cpp["size_after"])

            if red_py_dict == red_cpp_dict:
                n_match += 1
            elif first_mismatch is None:
                first_mismatch = (k, side, stats_py.size_after,
                                  stats_cpp["size_after"], red_py_dict, red_cpp_dict)

    print(f"Pairs    : {n_pairs}  (2 sides each = {2 * n_pairs} reductions)")
    print(f"Frames   : {n_frames}")
    print(f"Compared : {n_ok}")
    print(f"Match    : {n_match}  ({100.0 * n_match / max(1,n_ok):.2f}%)")
    if py_sizes_before:
        print(f"size before mean : {np.mean(py_sizes_before):.1f}")
        print(f"size after  py   : {np.mean(py_sizes_after):.1f}")
        print(f"size after  cpp  : {np.mean(cpp_sizes_after):.1f}")
    print(f"time(py)  : {t_py:.2f}s")
    print(f"time(cpp) : {t_cpp:.2f}s")
    print(f"speedup   : x{(t_py / t_cpp) if t_cpp > 0 else float('inf'):.2f}")
    if first_mismatch:
        k, side, sa_py, sa_cpp, rpy, rcpp = first_mismatch
        print(f"  first mismatch: pair#{k} side={side} size_py={sa_py} size_cpp={sa_cpp}")
        print(f"    py  : {[d['name'] for d in rpy]}")
        print(f"    cpp : {[d['name'] for d in rcpp]}")

    if n_match == n_ok and n_ok > 0:
        print("\nStep 3 PASS: cpp reduce_bp == python behavioral_reduce_bp (structurally).")
        return 0
    print("\nStep 3 FAIL")
    return 1


if __name__ == "__main__":
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 8081
    sys.exit(main(n_pairs, n_frames, seed))
