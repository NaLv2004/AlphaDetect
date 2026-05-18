"""Re-test only the 19 'true false reject' candidates, now varying EVO panel
to rule out 'EVO=0 zeroed the X-dependence' artifact."""
import os, sys, re, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M
DEG = 8; N_IN = DEG-1

def run(prog, X, L, EVO):
    return M.run_program(prog, "v2c", EVO.astype(np.float64), X.astype(np.float64), float(L), DEG, 0)

def check(prog, i, j, rng):
    n_none=n_match=n_mm=0; maxd=0.0
    for _ in range(400):
        EVO = rng.uniform(-3, 3, size=8)
        X = rng.uniform(-3, 3, size=N_IN)
        L = float(rng.uniform(-3, 3))
        y1 = run(prog, X, L, EVO)
        X2 = X.copy(); X2[i],X2[j] = X2[j],X2[i]
        y2 = run(prog, X2, L, EVO)
        if y1 is None or y2 is None: n_none += 1; continue
        if y1 == y2: n_match += 1
        else: n_mm += 1; maxd = max(maxd, abs(y1-y2))
    # adversarial: large at swap positions
    for big_i in (5.0, -5.0, 50.0, -50.0):
        for big_j in (5.0, -5.0, 50.0, -50.0):
            for L in (1.0, -1.0):
                X = np.ones(N_IN)*0.5
                X[i]=big_i; X[j]=big_j
                EVO = np.ones(8)*0.7
                y1 = run(prog, X, L, EVO)
                X2 = X.copy(); X2[i],X2[j] = X2[j],X2[i]
                y2 = run(prog, X2, L, EVO)
                if y1 is None or y2 is None: n_none+=1; continue
                if y1==y2: n_match+=1
                else: n_mm+=1; maxd=max(maxd,abs(y1-y2))
    return n_none, n_match, n_mm, maxd

def main():
    print("harvest 200 probe-v2c progs (same seed as before)...", flush=True)
    progs, att, _ = M.parallel_seed(
        side="v2c", n_target=200, max_attempts=20_000_000,
        threads=8, chunk_attempts=500, min_size=4, max_size=16, deg=DEG,
        num_configs=3, num_permutations=5, num_evo_panels=2,
        base_seed=2024, validator_mode="probe")
    pat = re.compile(r"not permutation-invariant \(swap (\d+) <-> (\d+)\)")
    cand = []
    for p in progs:
        ok,r = M.symbolic_validate_v2c(p, DEG, 0)
        if ok: continue
        m = pat.search(r)
        if m: cand.append((p,int(m.group(1)),int(m.group(2))))
    print(f"{len(cand)} swap rejects out of {len(progs)}.")
    rng = np.random.default_rng(13579)
    real_false = []
    print("\nRe-checking ALL with varied EVO, X, L:")
    correct = 0; false = 0; skip = 0
    for k,(p,i,j) in enumerate(cand):
        none,match,mm,md = check(p,i,j,rng)
        tot = none+match+mm
        if none > 0.9*tot:
            skip += 1; verdict = "SKIP_none"
        elif mm > 0:
            correct += 1; verdict = f"CORRECT (mm={mm}/{match+mm} maxd={md:.2e})"
        else:
            false += 1; real_false.append((k,p,i,j))
            verdict = f"TRUE_FALSE_REJECT (all {match} bit-equal)"
        # Only print non-correct for brevity
        if mm == 0 or none > 0.9*tot:
            print(f"  [{k+1:3d}] swap({i},{j}) {verdict}")
    print(f"\n=== Summary (with varied EVO) ===")
    print(f"  CORRECT   : {correct}/{len(cand)}")
    print(f"  TRUE FALSE: {false}/{len(cand)}")
    print(f"  SKIP_none : {skip}/{len(cand)}")
    if real_false:
        print(f"\n--- {len(real_false)} TRUE FALSE REJECTS (showing all symbolic outputs) ---")
        for k,p,i,j in real_false:
            tr = M.symbolic_trace_v2c(p, DEG, 0)
            top = (tr['steps'][-1]['float'][-1] if tr['steps'] and tr['steps'][-1]['float'] else '<empty>')
            print(f"\n[#{k+1}] swap({i},{j}) opaque={tr['opaque']} steps={tr['step_count']} branches={tr['branches_seen']}")
            print(f"  top = {top[:400]}{'...' if len(top)>400 else ''}")

main()
