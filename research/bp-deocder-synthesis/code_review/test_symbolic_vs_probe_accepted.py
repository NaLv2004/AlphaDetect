"""Stage 2: take probe-accepted programs from the seeder and run symbolic on them.

This identifies probe false-positives (programs that pass probe but
symbolic rejects), which is the bug the symbolic validator should catch.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))

import pushgp_cpp_seeder as S


def seed_and_check(side="c2v", n_target=200, deg=8, threads=4):
    print(f"Seeding {n_target} probe-accepted programs (side={side}, deg={deg}) ...")
    t0 = time.perf_counter()
    progs, attempts, _fps = S.parallel_seed(
        side=side, n_target=n_target, max_attempts=10_000_000,
        threads=threads, chunk_attempts=1000,
        min_size=4, max_size=20, deg=deg,
        num_configs=32, num_permutations=8, num_evo_panels=4,
        base_seed=12345)
    print(f"  got {len(progs)} programs in {time.perf_counter()-t0:.1f}s "
          f"({attempts} attempts)")

    sym_ok = 0
    sym_reject = []
    opaque = 0
    sym_times = []
    for i, ph in enumerate(progs):
        t0 = time.perf_counter()
        ok, reason = (S.symbolic_validate_c2v(ph, deg, 0) if side == "c2v"
                      else S.symbolic_validate_v2c(ph, deg, 0))
        sym_times.append(time.perf_counter() - t0)
        if ok:
            sym_ok += 1
        else:
            if "opaque" in reason:
                opaque += 1
            if len(sym_reject) < 10:
                sym_reject.append((i, reason, [ins["name"] for ins in ph.to_dict()]))

    print(f"\n=== probe-accepted programs: {len(progs)}, side={side}, deg={deg} ===")
    print(f"  symbolic also accepts: {sym_ok}/{len(progs)} ({sym_ok/max(1,len(progs)):.1%})")
    print(f"  symbolic rejects (probe false-positives or opaque): {len(progs)-sym_ok}")
    print(f"  of which opaque: {opaque}")
    import numpy as np
    print(f"  sym time: mean={1e3*np.mean(sym_times):.3f} ms  p99={1e3*np.quantile(sym_times,0.99):.3f} ms")
    print("\n  Sample symbolic rejects (potential probe false-positives):")
    for i, r, ops in sym_reject[:8]:
        print(f"    #{i}: sym='{r}'  ops={ops}")


if __name__ == "__main__":
    side = sys.argv[1] if len(sys.argv) > 1 else "c2v"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    seed_and_check(side=side, n_target=n)
