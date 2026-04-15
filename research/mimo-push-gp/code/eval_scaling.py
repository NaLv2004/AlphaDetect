"""Quick eval of the v3-best genome at various max_nodes to understand scaling."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bp_main_v2 import (
    Genome, Instruction, full_evaluation, qam16_constellation,
    program_to_formula_trace
)
from cpp_bridge import CppBPEvaluator

# Build v3-best genome
prog_down = [
    Instruction('Node.GetCumDist'),
    Instruction('Float.Tanh'),
    Instruction('Float.Rot'),
    Instruction('Float.Max'),
]
prog_up = [
    Instruction('Float.Swap'),
    Instruction('Node.GetCumDist'),
    Instruction('Node.ForEachChild', code_block=[Instruction('Node.GetMUp')]),
    Instruction('Float.Sqrt'),
    Instruction('Float.Abs'),
]
prog_belief = [
    Instruction('Float.EvoConst3'),
    Instruction('Float.Sub'),
    Instruction('Float.Div'),
    Instruction('Float.Div'),
]
prog_halt = [
    Instruction('Float.ConstHalf'),
    Instruction('Float.Abs'),
    Instruction('Float.GT'),
]

# Use Gen 1's optimized constants from run0415_v2
# EC0=0.011845 EC1=0.166585 EC2=0.354687 EC3=0.396172
log_constants = np.array([-1.9265, -0.7784, -0.4502, -0.4021])

genome = Genome(prog_down, prog_up, prog_belief, prog_halt, log_constants)

print("F_belief =", program_to_formula_trace(genome.prog_belief, 'belief', genome))
print("F_down   =", program_to_formula_trace(genome.prog_down, 'down', genome))
print("F_up     =", program_to_formula_trace(genome.prog_up, 'up', genome))
print("EC =", genome.evo_constants)
print()

snrs = [20.0, 22.0, 24.0]
node_budgets = [500, 1000, 1500, 2000]

for mn in node_budgets:
    print(f"\n--- max_nodes={mn} ---")
    cpp = CppBPEvaluator(Nt=16, Nr=16, mod_order=16,
                         max_nodes=mn, flops_max=5_000_000,
                         step_max=5000, max_bp_iters=3)
    t0 = time.time()
    results = full_evaluation(
        genome, Nt=16, Nr=16, mod_order=16,
        n_trials=500, snr_dbs=snrs, max_nodes=mn,
        flops_max=5_000_000, step_max=5000,
        cpp_evaluator=cpp, min_bit_errors=200,
        if_eval_baseline=False)
    dt = time.time() - t0
    for r in results:
        print(f"  SNR={r['snr_db']:.0f}dB  BER={r['evolved_ber']:.6f}"
              f"  ({r['evolved_bit_errors']} errors, {r['evolved_samples']} samples)"
              f"  flops={r['evolved_flops']:.0f}")
    print(f"  [{dt:.1f}s]")
