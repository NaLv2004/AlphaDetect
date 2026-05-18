"""Verify whether programs symbolic-rejected as 'not permutation-invariant'
are actually non-perm-invariant under concrete probe execution.

For each rejected program:
  1. Parse swap (i, j) from reason string.
  2. Pick a random X vector + random LV; run concrete with original X -> y1.
  3. Swap X[i] <-> X[j]; run -> y2.
  4. If y1 != y2 (within tol) on any of K random samples -> symbolic CORRECT.
     If y1 == y2 on ALL K samples -> potential false reject.

We also try perturbed inputs and multiple deg-1 entries to be robust.
"""
import os, sys, re, time, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

DEG = 8
N_IN = DEG - 1
N_SAMPLES = 20
TOL = 1e-7
SIDE = "v2c"

def run_concrete(prog, X, L_v):
    evo = np.zeros(8, dtype=np.float64)
    return M.run_program(prog, SIDE, evo, X.astype(np.float64), float(L_v), DEG, 0)

def check_one(prog, i, j, rng):
    """Return (verdict, n_mismatch, n_total)."""
    n_mismatch = 0
    n_total = 0
    n_none = 0
    for _ in range(N_SAMPLES):
        X = rng.uniform(-2, 2, size=N_IN)
        L = rng.uniform(-2, 2)
        y1 = run_concrete(prog, X, L)
        X2 = X.copy()
        X2[i], X2[j] = X2[j], X2[i]
        y2 = run_concrete(prog, X2, L)
        if y1 is None or y2 is None:
            n_none += 1
            continue
        n_total += 1
        if abs(y1 - y2) > TOL * max(1.0, abs(y1), abs(y2)):
            n_mismatch += 1
    if n_total == 0:
        return ("INCONCL_no_output", n_mismatch, n_total, n_none)
    if n_mismatch > 0:
        return ("CORRECT_reject", n_mismatch, n_total, n_none)
    return ("POTENTIAL_false_reject", n_mismatch, n_total, n_none)

def main():
    print(f"harvest probe-{SIDE} progs...", flush=True)
    t0 = time.time()
    progs, att, _ = M.parallel_seed(
        side=SIDE, n_target=150, max_attempts=20_000_000,
        threads=8, chunk_attempts=500, min_size=4, max_size=16, deg=DEG,
        num_configs=3, num_permutations=5, num_evo_panels=2,
        base_seed=99, validator_mode="probe")
    print(f"  got {len(progs)} in {att:,} attempts ({time.time()-t0:.1f}s)", flush=True)

    sym = M.symbolic_validate_v2c
    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)

    # Find rejected progs with swap pattern
    pat = re.compile(r"not permutation-invariant \(swap (\d+) <-> (\d+)\)")
    candidates = []
    print(f"running symbolic on {len(progs)} progs ...", flush=True)
    t1 = time.time()
    for idx, p in enumerate(progs):
        if idx % 20 == 0:
            print(f"  sym [{idx}/{len(progs)}] cands={len(candidates)} t={time.time()-t1:.1f}s", flush=True)
        ok, reason = sym(p, DEG, 0)
        if ok: continue
        m = pat.search(reason)
        if m:
            candidates.append((p, int(m.group(1)), int(m.group(2)), reason))
    print(f"\nFound {len(candidates)} progs rejected with explicit swap reason.", flush=True)

    # Verify first 30
    N_VERIFY = min(30, len(candidates))
    print(f"Verifying first {N_VERIFY}:\n")
    n_correct = 0
    n_false = 0
    n_inconcl = 0
    for k, (prog, i, j, reason) in enumerate(candidates[:N_VERIFY]):
        verdict, mm, tot, none = check_one(prog, i, j, np_rng)
        flag = {"CORRECT_reject": "OK ", "POTENTIAL_false_reject": "!!!", "INCONCL_no_output": "?  "}[verdict]
        print(f"  [{k+1:2d}] {flag} swap({i},{j}) mismatch={mm}/{tot} (none={none})  reason='{reason}'")
        if verdict == "CORRECT_reject": n_correct += 1
        elif verdict == "POTENTIAL_false_reject": n_false += 1
        else: n_inconcl += 1

    print(f"\n=== Summary ===")
    print(f"  CORRECT  reject : {n_correct}/{N_VERIFY}")
    print(f"  FALSE    reject : {n_false}/{N_VERIFY}  <-- symbolic bugs if > 0")
    print(f"  inconclusive    : {n_inconcl}/{N_VERIFY}")

main()
