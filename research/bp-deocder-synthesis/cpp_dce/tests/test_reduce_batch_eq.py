"""Step 4 gate (T8): multi-threaded reduce_bp_batch must produce
identical results to the single-threaded reduce_bp on every job.

Run:
    python cpp_dce/tests/test_reduce_batch_eq.py [n_pairs] [n_frames] [threads]
Defaults: 10 pairs x 2 frames, threads=8.
"""
from __future__ import annotations

import os
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


def main(n_pairs=10, n_frames=2, threads=8) -> int:
    par = build_parity(2, 1, 2)
    parH = cdce.build_parity_handle(par)
    max_iter = 4

    rpg = RandomProgramGenerator(rng=np.random.default_rng(9090))
    rx_llrs = _frames(par, n_frames, snr_db=0.0, seed=9091)

    jobs = []
    pairs = []
    for k in range(n_pairs):
        pv = rpg.random_v2c(min_size=4, max_size=12)
        pc = rpg.random_c2v(min_size=4, max_size=12)
        log_consts = rpg.random_log_constants() if hasattr(rpg, "random_log_constants") \
                     else np.zeros(8, dtype=np.float64)
        g = Genome(prog_v2c=pv, prog_c2v=pc, log_constants=log_consts.copy())
        evo = g.evo_const_values().astype(np.float64)
        pairs.append((pv, pc, evo))
        for side, prog, peer in [("v2c", pv, pc), ("c2v", pc, pv)]:
            jobs.append({
                "prog": program_to_list(prog),
                "side": side,
                "peer_prog": program_to_list(peer),
                "evo": evo,
            })

    # ---- serial (threads=1) reference
    t0 = time.perf_counter()
    serial = cdce.reduce_bp_batch(
        jobs, parH, rx_llrs, max_iter, 400, -1, 6, 1, None)
    t_serial = time.perf_counter() - t0

    # ---- multi-thread
    t0 = time.perf_counter()
    parallel = cdce.reduce_bp_batch(
        jobs, parH, rx_llrs, max_iter, 400, -1, 6, threads, None)
    t_par = time.perf_counter() - t0

    assert len(serial) == len(parallel) == len(jobs)
    n_match = 0
    first_mismatch = None
    for i in range(len(jobs)):
        if serial[i]["prog"] == parallel[i]["prog"]:
            n_match += 1
        elif first_mismatch is None:
            first_mismatch = (i, serial[i]["prog"], parallel[i]["prog"])

    print(f"Jobs     : {len(jobs)}  ({n_pairs} pairs x 2 sides)")
    print(f"Frames   : {n_frames}")
    print(f"Match    : {n_match}  ({100.0 * n_match / len(jobs):.2f}%)")
    print(f"time serial    (t=1)        : {t_serial:.2f}s")
    print(f"time parallel  (t={threads:<3})        : {t_par:.2f}s")
    print(f"thread speedup              : x{(t_serial / t_par) if t_par > 0 else 0:.2f}")
    if first_mismatch:
        i, a, b = first_mismatch
        print(f"  first mismatch: job#{i}")
        print(f"    serial   : {[x['name'] for x in a]}")
        print(f"    parallel : {[x['name'] for x in b]}")

    if n_match == len(jobs):
        print("\nStep 4 PASS: reduce_bp_batch parallel == serial on every job.")
        return 0
    print("\nStep 4 FAIL")
    return 1


if __name__ == "__main__":
    n_pairs  = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    threads  = int(sys.argv[3]) if len(sys.argv) > 3 else min(8, os.cpu_count() or 4)
    sys.exit(main(n_pairs, n_frames, threads))
