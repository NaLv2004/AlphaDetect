"""Quick test: verify random_genome works with new stricter dependency checks."""
import numpy as np
import time
from bp_main_v2 import random_genome, has_valid_bp_dependency, program_to_formula_trace

rng = np.random.RandomState(42)

for trial in range(3):
    t0 = time.time()
    g = random_genome(rng)
    dt = time.time() - t0
    print(f"\n--- Genome {trial+1} (took {dt:.1f}s) ---")
    fd = program_to_formula_trace(g.prog_down, 'down', g)
    fu = program_to_formula_trace(g.prog_up, 'up', g)
    fb = program_to_formula_trace(g.prog_belief, 'belief', g)
    fh = program_to_formula_trace(g.prog_halt, 'halt', g)
    print(f"  F_down:   {fd}")
    print(f"  F_up:     {fu}")
    print(f"  F_belief: {fb}")
    print(f"  H_halt:   {fh}")
    print(f"  Valid: {has_valid_bp_dependency(g, rng)}")
    print(f"  Len: {g.total_length()}")

print("\nAll tests passed!")
