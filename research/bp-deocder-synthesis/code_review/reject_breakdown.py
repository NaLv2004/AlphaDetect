"""Full breakdown: of 200 probe-valid v2c programs, how many does symbolic
reject (by reason), and of those rejects how many are CORRECT vs FALSE?"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

DEG = 8; N_IN = DEG - 1

def run(prog, X, L, EVO):
    return M.run_program(prog, "v2c", EVO.astype(np.float64),
                         X.astype(np.float64), float(L), DEG, 0)

progs, _, _ = M.parallel_seed(
    side="v2c", n_target=200, max_attempts=20_000_000,
    threads=8, chunk_attempts=500, min_size=4, max_size=16, deg=DEG,
    num_configs=3, num_permutations=5, num_evo_panels=2,
    base_seed=2024, validator_mode="probe")
print(f"probe-valid samples : {len(progs)}", flush=True)

swap_progs=[]; dep_progs=[]; odd_progs=[]; other=[]; accept=0
for p in progs:
    ok, r = M.symbolic_validate_v2c(p, DEG, 0)
    if ok:
        accept += 1; continue
    rl = r.lower()
    if "permutation-invariant" in r:    swap_progs.append(p)
    elif "depend" in rl or "missing" in rl: dep_progs.append(p)
    elif "odd" in rl:                   odd_progs.append(p)
    else:                               other.append((p, r))

total_rej = len(progs) - accept
print(f"symbolic accept     : {accept}")
print(f"symbolic reject TOT : {total_rej}")
print(f"  swap (perm)       : {len(swap_progs)}")
print(f"  dep (missing X)   : {len(dep_progs)}")
print(f"  odd (parity)      : {len(odd_progs)}")
print(f"  other             : {len(other)}")
for p, r in other[:5]:
    print("   other reason  :", r[:120])

rng = np.random.default_rng(7)
def dep_is_correct(p):
    EVO = rng.uniform(-3,3,size=8); L=float(rng.uniform(-3,3))
    for _ in range(80):
        X1=rng.uniform(-3,3,size=N_IN); X2=rng.uniform(-3,3,size=N_IN)
        y1=run(p,X1,L,EVO); y2=run(p,X2,L,EVO)
        if y1 is None or y2 is None: continue
        if y1 != y2: return False
    return True

def odd_is_correct(p):
    for _ in range(80):
        X=rng.uniform(-3,3,size=N_IN); L=float(rng.uniform(-3,3))
        EVO=rng.uniform(-3,3,size=8)
        yp=run(p,X,L,EVO); yn=run(p,-X,-L,EVO)
        if yp is None or yn is None: continue
        if abs(yp + yn) > 1e-9: return True
    return False

dep_corr = sum(dep_is_correct(p) for p in dep_progs)
odd_corr = sum(odd_is_correct(p) for p in odd_progs)
print(f"\ndep   rejects : {len(dep_progs)} -> CORRECT {dep_corr}, FALSE {len(dep_progs)-dep_corr}")
print(f"odd   rejects : {len(odd_progs)} -> CORRECT {odd_corr}, FALSE {len(odd_progs)-odd_corr}")
print(f"swap  rejects : {len(swap_progs)} -> CORRECT 94, FALSE 34  (from reverify_after_B)")
total_corr  = 94 + dep_corr + odd_corr
total_false = 34 + (len(dep_progs)-dep_corr) + (len(odd_progs)-odd_corr)
print(f"\nGRAND TOTAL   : rejected {total_rej}/{len(progs)}  CORRECT {total_corr}  FALSE {total_false}  other {len(other)}")
