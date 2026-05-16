"""Benchmark DCE fingerprint cost and estimate T2/T3/T5 runtimes."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pushgp.evolution import _behav_fingerprint
from pushgp.dce import _walk_positions, behavioral_reduce
from pushgp.random_program import RandomProgramGenerator
from pushgp.validators import validate_v2c, validate_c2v
from pushgp.program import program_length

rpg = RandomProgramGenerator(rng=np.random.default_rng(0))

for side, gen_fn, val_fn in [
    ("v2c", rpg.random_v2c, validate_v2c),
    ("c2v", rpg.random_c2v, validate_c2v),
]:
    print(f"\n=== {side} ===")
    for label, mn, mx in [("small", 4, 12), ("medium", 20, 40), ("big", 40, 60)]:
        progs = []
        attempts = 0
        while len(progs) < 5 and attempts < 200_000:
            attempts += 1
            p = gen_fn(min_size=mn, max_size=mx)
            ok, _ = val_fn(p, rng=np.random.default_rng(attempts), deg=8)
            if ok:
                progs.append(p)
        if not progs:
            print(f"  [{label}] no valid progs found")
            continue

        p0 = progs[0]
        n_pos = len(_walk_positions(p0))

        # time a single FP call
        N = 10
        t0 = time.perf_counter()
        for _ in range(N):
            _behav_fingerprint(side, p0)
        fp_ms = (time.perf_counter() - t0) / N * 1000

        # time full DCE on p0
        t1 = time.perf_counter()
        r = behavioral_reduce(p0, side)
        dce_s = time.perf_counter() - t1
        sz_before = program_length(p0)
        sz_after  = program_length(r)

        print(f"  [{label:6s}] top={len(p0):3d} flat_positions={n_pos:3d}  "
              f"fp={fp_ms:6.1f}ms/call  "
              f"dce={dce_s:6.2f}s  "
              f"{sz_before}->{sz_after} ({100*(sz_before-sz_after)/max(1,sz_before):.0f}% removed)")

        # extrapolate T2/T5 cost
        est_t2_per_prog = fp_ms * n_pos * 5 / 1000   # ~5 passes typical
        est_t5_per_prog = dce_s
        print(f"           est T2(1 prog)={est_t2_per_prog:.1f}s  "
              f"T5(1 prog)={est_t5_per_prog:.1f}s")
