"""Symbolic histogram with per-program progress to catch crashes."""
import os, sys, time, collections, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

SIDE = sys.argv[1] if len(sys.argv) > 1 else "v2c"
POOL = int(sys.argv[2]) if len(sys.argv) > 2 else 100

print(f"harvest {POOL} probe-{SIDE} programs...", flush=True)
t0 = time.time()
progs, att, _ = M.parallel_seed(
    side=SIDE, n_target=POOL, max_attempts=20_000_000,
    threads=8, chunk_attempts=500, min_size=4, max_size=16, deg=8,
    num_configs=3, num_permutations=5, num_evo_panels=2,
    base_seed=99, validator_mode="probe")
print(f"  got {len(progs)} in {att:,} attempts ({time.time()-t0:.1f}s)", flush=True)

sym = M.symbolic_validate_v2c if SIDE == "v2c" else M.symbolic_validate_c2v
hist = collections.Counter()
n_ok = 0
t_sym = time.time()
for i, p in enumerate(progs):
    t1 = time.time()
    try:
        ok, reason = sym(p, 8, 0)
    except Exception as e:
        ok, reason = False, f"EXCEPTION: {e}"
    dt = time.time() - t1
    if ok:
        n_ok += 1
        hist["__OK__"] += 1
    else:
        hist[reason.strip()] += 1
    if dt > 0.5 or (i % 25 == 0):
        print(f"  [{i+1}/{len(progs)}] dt={dt*1000:.0f}ms ok={ok} reason={reason[:60]!r}",
              flush=True)

print(f"\nResult: {n_ok}/{len(progs)} OK  (sym wall {time.time()-t_sym:.1f}s)")
print("Histogram (top 30):")
for k, v in hist.most_common(30):
    print(f"  {v:5d}  {k}")
