"""Quick smoke test for the perturbation-based BP dependency checker."""
import numpy as np
from bp_main_v2 import random_genome, has_valid_bp_dependency, program_to_formula

rng = np.random.RandomState(42)
print('Generating 10 valid genomes with perturbation check...')
for i in range(10):
    g = random_genome(rng)
    ok = has_valid_bp_dependency(g, rng)
    fd = program_to_formula(g.prog_down, ['M_par_down', 'C_i'])
    fu = program_to_formula(g.prog_up, ['C_1', 'M_1_up', 'C_2', 'M_2_up'])
    fb = program_to_formula(g.prog_belief, ['D_i', 'M_down', 'M_up'])
    fh = program_to_formula(g.prog_halt, ['old_M_up', 'new_M_up'])
    print(f'  Genome {i}: valid={ok}, len={g.total_length()}, logc=[{",".join(f"{c:.2f}" for c in g.log_constants)}]')
    print(f'    F_down  = {fd}')
    print(f'    F_up    = {fu}')
    print(f'    F_belief= {fb}')
    print(f'    H_halt  = {fh}')
print('Done!')
