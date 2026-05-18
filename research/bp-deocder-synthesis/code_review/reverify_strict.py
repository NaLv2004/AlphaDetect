"""Stricter re-verification of "false reject" cases.

For each program symbolic-rejected with swap(i,j) reason:
  - Run with N=200 random (X, L) samples.
  - Compare y1 (X) vs y2 (X with X[i]<->X[j]) using EXACT bitwise inequality
    (no rtol/atol — float ops are deterministic per input on the same machine).
  - Also try wider sampling ranges and adversarial X (single large component)
    to escape any fixed-point convergence basin that might mask the diff.
  - Classify:
       CORRECT_reject       : at least one (X,L) where y1 != y2 EXACTLY
       TRUE_false_reject    : all (y1,y2) bit-equal across all samples & adversarial
       SKIP_concrete_none   : concrete VM returned None on > 90% samples
"""
import os, sys, re, time, random, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

DEG = 8
N_IN = DEG - 1
SIDE = "v2c"

def run(prog, X, L):
    evo = np.zeros(8, dtype=np.float64)
    return M.run_program(prog, SIDE, evo, X.astype(np.float64), float(L), DEG, 0)

def check_strict(prog, i, j, rng):
    n_none = 0
    n_match = 0
    n_mismatch = 0
    max_diff = 0.0
    samples = []
    # Random uniform
    for _ in range(80):
        X = rng.uniform(-2, 2, size=N_IN)
        L = rng.uniform(-2, 2)
        samples.append((X, L))
    # Wide range
    for _ in range(60):
        X = rng.uniform(-10, 10, size=N_IN)
        L = rng.uniform(-5, 5)
        samples.append((X, L))
    # Adversarial: single large at i or j
    for big in (10.0, -10.0, 100.0, -100.0, 0.001, -0.001):
        for L in (1.0, -1.0, 0.5):
            X = np.zeros(N_IN); X[i] = big; samples.append((X.copy(), L))
            X = np.zeros(N_IN); X[j] = big; samples.append((X.copy(), L))
    for _ in range(40):
        X = rng.standard_normal(N_IN) * 3.0
        L = float(rng.standard_normal()) * 2.0
        samples.append((X, L))

    first_diff = None
    for X, L in samples:
        y1 = run(prog, X, L)
        X2 = X.copy(); X2[i], X2[j] = X2[j], X2[i]
        y2 = run(prog, X2, L)
        if y1 is None or y2 is None:
            n_none += 1; continue
        if y1 == y2:
            n_match += 1
        else:
            d = abs(y1 - y2)
            max_diff = max(max_diff, d)
            n_mismatch += 1
            if first_diff is None:
                first_diff = (X, L, y1, y2, d)
    return n_none, n_match, n_mismatch, max_diff, first_diff, len(samples)

def main():
    print(f"harvest probe-{SIDE} progs...", flush=True)
    t0 = time.time()
    progs, att, _ = M.parallel_seed(
        side=SIDE, n_target=200, max_attempts=20_000_000,
        threads=8, chunk_attempts=500, min_size=4, max_size=16, deg=DEG,
        num_configs=3, num_permutations=5, num_evo_panels=2,
        base_seed=2024, validator_mode="probe")
    print(f"  got {len(progs)} in {att:,} attempts ({time.time()-t0:.1f}s)", flush=True)

    pat = re.compile(r"not permutation-invariant \(swap (\d+) <-> (\d+)\)")
    candidates = []
    for p in progs:
        ok, reason = M.symbolic_validate_v2c(p, DEG, 0)
        if ok: continue
        m = pat.search(reason)
        if m:
            candidates.append((p, int(m.group(1)), int(m.group(2)), reason))
    print(f"Got {len(candidates)} 'swap' rejections out of {len(progs)} probe-valid progs.")

    np_rng = np.random.default_rng(777)
    n_correct = 0
    n_true_false = 0
    n_skip = 0
    true_false_list = []
    print(f"\nVerifying ALL {len(candidates)} with STRICT bitwise inequality:\n")
    for k, (prog, i, j, reason) in enumerate(candidates):
        none, match, mm, md, fd, tot = check_strict(prog, i, j, np_rng)
        if none > 0.9 * tot:
            verdict = "SKIP_none"
            n_skip += 1
        elif mm > 0:
            verdict = f"CORRECT (mm={mm}/{tot-none} maxd={md:.2e})"
            n_correct += 1
        else:
            verdict = f"TRUE_FALSE_REJECT (all {match} samples bit-equal)"
            n_true_false += 1
            true_false_list.append((k, prog, i, j, reason))
        print(f"  [{k+1:3d}/{len(candidates)}] swap({i},{j}) {verdict}")

    print(f"\n=== Summary ===")
    print(f"  CORRECT  reject     : {n_correct}/{len(candidates)}")
    print(f"  TRUE     false rej. : {n_true_false}/{len(candidates)}  <-- real symbolic bugs if > 0")
    print(f"  skipped (no output) : {n_skip}/{len(candidates)}")

    # Dump up to 5 true-false-reject programs for further analysis
    if true_false_list:
        print(f"\n--- TRUE FALSE REJECTS (up to 5) ---")
        for k, prog, i, j, reason in true_false_list[:5]:
            print(f"\n[#{k+1}] swap({i},{j}) reason={reason!r}")
            # symbolic trace
            tr = M.symbolic_trace_v2c(prog, DEG, 0)
            print(f"  symbolic: opaque={tr['opaque']} steps={tr['step_count']} branches={tr['branches_seen']}")
            if tr['steps']:
                last = tr['steps'][-1]
                f = last['float']
                print(f"  final top = {(f[-1] if f else '<empty>')[:300]}")

main()
