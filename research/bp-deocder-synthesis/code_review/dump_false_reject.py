"""Find the first FALSE-reject program and dump its symbolic structure.
Outputs:
  - program text (op_name list)
  - per-path: cond + out
  - combined ITE-chain output
  - sigma-substituted combined output
  - structural-equal flag
This gives direct evidence of WHY check_sym says they differ.
"""
import os, sys, re, time, random, json
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

def is_false_reject(prog, i, j, np_rng):
    n_match = 0
    n_total = 0
    for _ in range(N_SAMPLES):
        X = np_rng.uniform(-2, 2, size=N_IN)
        L = np_rng.uniform(-2, 2)
        y1 = run_concrete(prog, X, L)
        X2 = X.copy(); X2[i], X2[j] = X2[j], X2[i]
        y2 = run_concrete(prog, X2, L)
        if y1 is None or y2 is None: continue
        n_total += 1
        if abs(y1 - y2) <= TOL * max(1.0, abs(y1), abs(y2)):
            n_match += 1
    return n_total > 0 and n_match == n_total, n_match, n_total

def prog_to_text(prog):
    d = M.program_to_dict(prog)
    # d is some nested dict
    return json.dumps(d, indent=1, default=str)

def main():
    print("Harvesting probe-v2c progs...", flush=True)
    t0 = time.time()
    progs, att, _ = M.parallel_seed(
        side=SIDE, n_target=150, max_attempts=20_000_000,
        threads=8, chunk_attempts=500, min_size=4, max_size=16, deg=DEG,
        num_configs=3, num_permutations=5, num_evo_panels=2,
        base_seed=99, validator_mode="probe")
    print(f"  got {len(progs)} in {att:,} attempts ({time.time()-t0:.1f}s)", flush=True)

    sym = M.symbolic_validate_v2c
    pat = re.compile(r"not permutation-invariant \(swap (\d+) <-> (\d+)\)")
    np_rng = np.random.default_rng(12345)

    print("Scanning for first FALSE-reject candidate ...", flush=True)
    for idx, p in enumerate(progs):
        ok, reason = sym(p, DEG, 0)
        if ok: continue
        m = pat.search(reason)
        if not m: continue
        i, j = int(m.group(1)), int(m.group(2))
        is_fr, mm, tot = is_false_reject(p, i, j, np_rng)
        if not is_fr:
            continue
        print(f"\n>>> FOUND FALSE-reject @ idx={idx}, swap({i},{j}), match={mm}/{tot}")
        print(f"    sym reason: {reason}")
        print(f"\n--- Program JSON ---")
        print(prog_to_text(p)[:4000])
        print(f"\n--- Symbolic dump ---")
        dump = M.symbolic_dump_v2c(p, DEG, 0, i, j)
        print(f"  #paths = {len(dump['paths'])}")
        for k, pd in enumerate(dump['paths']):
            print(f"  [path {k}] ok={pd['ok']} reason={pd['reason']}")
            print(f"           cond = {pd['cond']}")
            print(f"           out  = {pd['out']}")
        print(f"\n--- Combined ---")
        print(f"  combined = {dump['combined']}")
        print(f"\n--- After swap X[{i}]<->X[{j}] ---")
        print(f"  sigma    = {dump['sigma']}")
        print(f"  hash_equal = {dump['sym_equal']}")
        # Show explicit char-diff
        a = dump['combined']; b = dump['sigma']
        if a != b:
            # Find first diff position
            n = min(len(a), len(b))
            d = next((k for k in range(n) if a[k] != b[k]), n)
            print(f"  first diff at char {d}:")
            print(f"    combined ...{a[max(0,d-30):d]}>>>{a[d:d+50]}")
            print(f"    sigma    ...{b[max(0,d-30):d]}>>>{b[d:d+50]}")
        return

    print("No FALSE-reject found among harvested progs.")

main()
