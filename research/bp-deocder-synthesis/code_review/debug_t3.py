"""Debug T3 failure: is reduce(p) truly at fixpoint?

If reduce(reduce(p)) != reduce(p), either:
  A) reduce(p) is NOT at fixpoint (bug in behavioral_reduce)
  B) fp is non-deterministic (different calls to fp_fn give different results)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pushgp.dce import behavioral_reduce, _walk_positions, _remove_at, DCEStats
from pushgp.evolution import _behav_fingerprint
from pushgp.program import program_length
from pushgp.random_program import RandomProgramGenerator
from pushgp.serialize import program_to_dict

rpg = RandomProgramGenerator(rng=np.random.default_rng(23456))
side = "v2c"

for idx in range(5):
    p = rpg.random_v2c(min_size=4, max_size=30)
    fp0 = _behav_fingerprint(side, p)
    r1 = behavioral_reduce(p, side)
    fp1 = _behav_fingerprint(side, r1)
    r2 = behavioral_reduce(r1, side)
    fp2 = _behav_fingerprint(side, r2)

    sizes = f"|p|={program_length(p)} -> |r1|={program_length(r1)} -> |r2|={program_length(r2)}"
    print(f"\n[idx={idx}] {sizes}")
    print(f"  fp0==fp1: {fp0==fp1}  fp1==fp2: {fp1==fp2}")

    if program_length(r1) != program_length(r2):
        print("  T3 VIOLATION: r1 != r2 in size. Checking if r1 is at fixpoint...")
        # Check if r1 is truly at fixpoint
        positions = _walk_positions(r1)
        violations = []
        for pos in positions:
            try:
                cand = _remove_at(r1, pos)
            except Exception as e:
                print(f"    _remove_at raised {e} at pos={pos}")
                continue
            fp_cand = _behav_fingerprint(side, cand)
            if fp_cand == fp1:
                violations.append((pos, program_length(cand)))
                if len(violations) <= 3:
                    print(f"    FIXPOINT VIOLATION: pos={pos}, cand_size={program_length(cand)}, fp matches")

        if violations:
            print(f"  => BUG: r1 is NOT at fixpoint ({len(violations)} removable positions)")
        else:
            print(f"  => r1 IS at fixpoint. Issue is fp non-determinism or algorithm path-dep.")
            # Check fp determinism
            fp1_again = _behav_fingerprint(side, r1)
            print(f"  fp1 determinism: {fp1 == fp1_again}")
            fp2_check = _behav_fingerprint(side, r2)
            print(f"  fp(r2) == fp1: {fp2_check == fp1}")
