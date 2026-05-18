"""Fast iteration debug:
  Phase A: harvest N probe-accepted programs, print every 200ms
  Phase B: run symbolic on each sequentially with progress every 200
  Reports per-reason histogram and timing.
"""
import os, sys, time, collections, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

ap = argparse.ArgumentParser()
ap.add_argument("--side", default="v2c")
ap.add_argument("--pool", type=int, default=2000)
ap.add_argument("--max-attempts", type=int, default=50_000_000)
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()

print(f"[{time.strftime('%H:%M:%S')}] Phase A: harvest {args.pool} probe-{args.side} programs", flush=True)

# Harvest in small batches so we get progress prints
got = []
t0 = time.time()
total_attempts = 0
batch = max(50, args.pool // 20)
base = 1000
while len(got) < args.pool and total_attempts < args.max_attempts:
    progs, att, _ = M.parallel_seed(
        side=args.side, n_target=batch, max_attempts=args.max_attempts // 10,
        threads=args.threads, chunk_attempts=500,
        min_size=4, max_size=16, deg=8,
        num_configs=3, num_permutations=5, num_evo_panels=2,
        base_seed=base, validator_mode="probe")
    got.extend(progs)
    total_attempts += att
    base += 7919
    dt = time.time() - t0
    print(f"  [{dt:6.1f}s] +{len(progs)} (att {att:,}) total {len(got)}/{args.pool}, attempts {total_attempts:,}, rate {total_attempts/dt/1000:.1f}k/s", flush=True)
    if len(progs) == 0:
        print("  (no progress; raising base seed and trying again)", flush=True)

print(f"\n[{time.strftime('%H:%M:%S')}] Phase B: run symbolic_{args.side} on {len(got)} programs", flush=True)
reasons = collections.Counter()
n_ok = 0
t0 = time.time()
sym_fn = M.symbolic_validate_v2c if args.side == "v2c" else M.symbolic_validate_c2v
for i, p in enumerate(got):
    try:
        ok, reason = sym_fn(p, 8, 0)
    except Exception as e:
        ok = False; reason = f"EXC:{type(e).__name__}:{e}"
    if ok:
        n_ok += 1
        reasons["OK"] += 1
    else:
        r = reason
        if r.startswith("opaque"):
            key = "opaque:" + r.split(":", 2)[1].split(" ", 1)[0] if ":" in r else "opaque"
        else:
            key = r.split(":")[0].split("(")[0].strip()
        reasons[key] += 1
    if (i+1) % 200 == 0:
        dt = time.time() - t0
        print(f"  [{dt:6.1f}s] {i+1}/{len(got)} done, OK={n_ok}, rate {(i+1)/dt:.0f}/s", flush=True)

dt = time.time() - t0
print(f"\nFinal: {n_ok}/{len(got)} OK in {dt:.2f}s ({len(got)/dt:.0f} prog/s)", flush=True)
print("Reason histogram:")
for k, v in reasons.most_common():
    pct = 100.0*v/len(got)
    print(f"  {v:6d} ({pct:5.1f}%)  {k}", flush=True)
