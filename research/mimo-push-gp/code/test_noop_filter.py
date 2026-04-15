"""Quick diagnostic: how long does random_genome take with the noop filter?"""
import time
import numpy as np
import sys
sys.path.insert(0, '.')

from bp_main_v2 import random_genome, has_valid_bp_dependency, Genome

rng = np.random.RandomState(42)

print("Testing random_genome generation time with noop filter...")
for i in range(5):
    t0 = time.time()
    g = random_genome(rng)
    dt = time.time() - t0
    ok = has_valid_bp_dependency(g, rng)
    print(f"  Genome {i}: {dt:.2f}s  valid={ok}  total_len={g.total_length()}")
    print(f"    DOWN: {[ins.name for ins in g.prog_down]}")
    print(f"    UP:   {[ins.name for ins in g.prog_up]}")
    print(f"    BEL:  {[ins.name for ins in g.prog_belief]}")
    print(f"    HALT: {[ins.name for ins in g.prog_halt]}")

print("\nDone.")
