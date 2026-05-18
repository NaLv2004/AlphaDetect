"""Granular opaque-reason histogram. Harvest small probe pool then run
symbolic, splitting opaque by full reason string."""
import os, sys, time, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

SIDE = sys.argv[1] if len(sys.argv) > 1 else "v2c"
POOL = int(sys.argv[2]) if len(sys.argv) > 2 else 300

print(f"harvest {POOL} probe-{SIDE} programs...", flush=True)
t0 = time.time()
progs, att, _ = M.parallel_seed(side=SIDE, n_target=POOL, max_attempts=20_000_000,
    threads=8, chunk_attempts=500, min_size=4, max_size=16, deg=8,
    num_configs=3, num_permutations=5, num_evo_panels=2,
    base_seed=99, validator_mode="probe")
print(f"  got {len(progs)} in {att:,} attempts ({time.time()-t0:.1f}s)", flush=True)

sym = M.symbolic_validate_v2c if SIDE == "v2c" else M.symbolic_validate_c2v
hist = collections.Counter()
n_ok = 0
for p in progs:
    ok, reason = sym(p, 8, 0)
    if ok:
        n_ok += 1
        hist["__OK__"] += 1
    else:
        # full opaque reason: keep after "opaque: "
        hist[reason.strip()] += 1

print(f"\nResult: {n_ok}/{len(progs)} OK")
print("Histogram (top 30):")
for k, v in hist.most_common(30):
    print(f"  {v:5d}  {k}")
