"""For the 26 'other' (opaque/no-output) rejects: what fraction are
actually GOOD programs (satisfy spec) vs BAD (rightly rejectable)?"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

DEG=8; N_IN=DEG-1
def run(prog, X, L, EVO):
    return M.run_program(prog,"v2c", EVO.astype(np.float64),
                         X.astype(np.float64), float(L), DEG, 0)

progs,_,_ = M.parallel_seed(side="v2c", n_target=200, max_attempts=20_000_000,
  threads=8, chunk_attempts=500, min_size=4, max_size=16, deg=DEG,
  num_configs=3, num_permutations=5, num_evo_panels=2,
  base_seed=2024, validator_mode="probe")

other=[]
for p in progs:
    ok,r = M.symbolic_validate_v2c(p, DEG, 0)
    if ok: continue
    rl = r.lower()
    if ("permutation-invariant" in r) or ("depend" in rl) or ("missing" in rl) or ("odd" in rl):
        continue
    other.append((p,r))

print(f"opaque/other rejects: {len(other)}", flush=True)
rng = np.random.default_rng(42)

def empirical_spec(prog):
    """Random Monte Carlo: must be (i) X-dependent, (ii) odd in (L,X), (iii) permutation-invariant in X."""
    # 1) dependence
    dep = False
    for _ in range(80):
        EVO = rng.uniform(-3,3,size=8); L=float(rng.uniform(-3,3))
        X1=rng.uniform(-3,3,size=N_IN); X2=rng.uniform(-3,3,size=N_IN)
        y1=run(prog,X1,L,EVO); y2=run(prog,X2,L,EVO)
        if y1 is None or y2 is None: return None, "exec-none"
        if y1 != y2: dep=True; break
    if not dep: return False, "indep-of-X"
    # 2) odd
    for _ in range(80):
        EVO=rng.uniform(-3,3,size=8); L=float(rng.uniform(-3,3))
        X=rng.uniform(-3,3,size=N_IN)
        yp=run(prog,X,L,EVO); yn=run(prog,-X,-L,EVO)
        if yp is None or yn is None: return None, "exec-none"
        if abs(yp + yn) > 1e-9: return False, "not-odd"
    # 3) perm
    for _ in range(80):
        EVO=rng.uniform(-3,3,size=8); L=float(rng.uniform(-3,3))
        X=rng.uniform(-3,3,size=N_IN)
        i,j = rng.choice(N_IN,size=2,replace=False)
        Xp=X.copy(); Xp[i],Xp[j]=Xp[j],Xp[i]
        y1=run(prog,X,L,EVO); y2=run(prog,Xp,L,EVO)
        if y1 is None or y2 is None: return None, "exec-none"
        if y1 != y2: return False, "not-perm-inv"
    return True, "all-pass"

good=bad=nonexec=0
reasons={}
detail=[]
for k,(p,r) in enumerate(other):
    verdict, why = empirical_spec(p)
    reasons[why] = reasons.get(why,0)+1
    detail.append((k,r[:80],verdict,why))
    if verdict is True: good+=1
    elif verdict is False: bad+=1
    else: nonexec+=1

print(f"  empirically GOOD (false reject) : {good}")
print(f"  empirically BAD  (correct reject): {bad}")
print(f"  exec-none / undefined           : {nonexec}")
print(f"  by reason: {reasons}")
print("\nfirst 12 details:")
for k,r,v,w in detail[:12]:
    print(f"  [{k:2d}] reject='{r}' -> verdict={v} ({w})")
