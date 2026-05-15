"""Benchmark: C++ parallel_seed vs Python parallel_fill_random.

Generates n_target valid programs both ways and reports wall time.
"""
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import pushgp_cpp_seeder as M

def progress(side, val, att, elapsed):
    print(f"  [{side}] valid={val} attempts={att} elapsed={elapsed:.1f}s")

def main(n_target=100, threads=8):
    print(f"=== C++ parallel_seed: n_target={n_target}, threads={threads} ===")
    t0 = time.perf_counter()
    progs, attempts = M.parallel_seed(
        side="v2c",
        n_target=n_target,
        max_attempts=10_000_000,
        threads=threads,
        chunk_attempts=2000,
        min_size=4,
        max_size=30,
        deg=8,
        num_configs=3,
        num_permutations=5,
        base_seed=2025,
        progress_cb=progress,
    )
    dt = time.perf_counter() - t0
    print(f"v2c: {len(progs)}/{n_target} in {dt:.2f}s, attempts={attempts}, "
          f"pass={len(progs)/max(1,attempts)*100:.4f}%")

    t0 = time.perf_counter()
    progs2, attempts2 = M.parallel_seed(
        side="c2v", n_target=n_target, max_attempts=10_000_000,
        threads=threads, chunk_attempts=2000, min_size=4, max_size=30,
        deg=8, num_configs=3, num_permutations=5, base_seed=4096,
        progress_cb=progress,
    )
    dt2 = time.perf_counter() - t0
    print(f"c2v: {len(progs2)}/{n_target} in {dt2:.2f}s, attempts={attempts2}, "
          f"pass={len(progs2)/max(1,attempts2)*100:.4f}%")

    print(f"TOTAL: {dt + dt2:.2f}s for both sides")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(n, t)
