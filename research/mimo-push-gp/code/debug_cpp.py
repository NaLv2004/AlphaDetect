"""Quick debug: check C++ fault count and details."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from bp_main_v2 import (
    Genome, Instruction, generate_mimo_sample, N_EVO_CONSTS
)
from cpp_bridge import CppBPEvaluator, encode_program

def make_ceiling_genome():
    return Genome(
        prog_down=[],
        prog_up=[],
        prog_belief=[Instruction(name='Node.GetCumDist')],
        prog_halt=[Instruction(name='Bool.True')],
        log_constants=np.zeros(N_EVO_CONSTS),
    )

constellation = np.array([-3, -1, 1, 3], dtype=float)
Nt, Nr = 16, 16
rng = np.random.RandomState(42)
dataset = [generate_mimo_sample(Nr, Nt, constellation, 22.0, rng) for _ in range(5)]

genome = make_ceiling_genome()

# Check encoded programs
evo_c = genome.evo_constants
for name, prog in [("down", genome.prog_down), ("up", genome.prog_up),
                    ("belief", genome.prog_belief), ("halt", genome.prog_halt)]:
    ops = encode_program(prog, evolved_constants=evo_c)
    print(f"  {name}: {[ins.name for ins in prog]} -> opcodes: {ops}")

# Evaluate with details
cpp_eval = CppBPEvaluator(
    Nt=16, Nr=16, mod_order=16,
    max_nodes=1000, flops_max=3_000_000, step_max=2000, max_bp_iters=2
)
avg_ber, avg_flops, faults, avg_bp = cpp_eval.evaluate_genome(genome, dataset)
print(f"\nC++ result: BER={avg_ber:.5f}, flops={avg_flops:.0f}, faults={faults}, bp_calls={avg_bp:.1f}")
print(f"  n_samples={len(dataset)}")
