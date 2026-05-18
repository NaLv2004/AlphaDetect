"""Stress test: validate many random programs symbolically OUTSIDE
parallel_seed (single-thread, no atomics) to isolate the crash source.
"""
import os, sys, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

# Get all available op names
all_ops = M.all_op_names()
print(f"total ops available: {len(all_ops)}", flush=True)

# Use C++ RPG via parallel_seed with mode "probe" and a tiny budget
# just to harvest random programs without running symbolic
def gen_random_progs(side, n, max_attempts=200_000):
    """Use probe-mode parallel_seed with very few accepts then discard
    the validity filter — actually we just want raw random programs.
    Easier: use parallel_seed and accept whatever probe gives us."""
    progs, _, _ = M.parallel_seed(side=side, n_target=n, max_attempts=max_attempts,
        threads=1, chunk_attempts=200, min_size=4, max_size=16, deg=8,
        num_configs=1, num_permutations=1, num_evo_panels=1,
        base_seed=42, validator_mode="probe")
    return progs

side = "v2c"
print(f"\nHarvesting random {side} programs via probe...", flush=True)
progs = gen_random_progs(side, 200)
print(f"got {len(progs)} probe-accepted {side} programs", flush=True)

print(f"\nRunning symbolic_validate_{side} on each (single-thread, with print)...", flush=True)
ok_cnt = 0
for i, p in enumerate(progs):
    try:
        if side == "v2c":
            ok, reason = M.symbolic_validate_v2c(p, 8, 0)
        else:
            ok, reason = M.symbolic_validate_c2v(p, 8, 0)
        if ok: ok_cnt += 1
        if i < 5 or (i % 50 == 0):
            print(f"  [{i}] ok={ok} reason={reason[:60]!r}", flush=True)
    except Exception as e:
        print(f"  [{i}] EXCEPTION: {e}", flush=True)
print(f"\nResult: {ok_cnt}/{len(progs)} OK; survived all calls", flush=True)
