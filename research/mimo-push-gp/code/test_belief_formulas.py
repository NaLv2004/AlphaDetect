"""Test several manually-designed F_belief formulas to find one that beats the baseline."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bp_main_v2 import (
    Genome, Instruction, full_evaluation, qam16_constellation,
    program_to_formula_trace, N_EVO_CONSTS
)
from cpp_bridge import CppBPEvaluator

def build_genome_from_instrs(down_instrs, up_instrs, belief_instrs, halt_instrs, log_consts=None):
    """Create a genome from instruction name lists."""
    def mk(item):
        if isinstance(item, dict):
            body = [mk(b) for b in item.get('body', [])]
            return Instruction(name=item['name'], code_block=body)
        return Instruction(name=item)
    if log_consts is None:
        log_consts = np.zeros(N_EVO_CONSTS)
    return Genome(
        prog_down=[mk(i) for i in down_instrs],
        prog_up=[mk(i) for i in up_instrs],
        prog_belief=[mk(i) for i in belief_instrs],
        prog_halt=[mk(i) for i in halt_instrs],
        log_constants=np.array(log_consts),
    )

snrs = [20.0, 22.0, 24.0]
eval_n = 1000
max_nodes = 1500

cpp = CppBPEvaluator(Nt=16, Nr=16, mod_order=16,
                     max_nodes=max_nodes, flops_max=5_000_000,
                     step_max=5000, max_bp_iters=3)

# Common components
# F_down: pass down cumulative distance
DOWN_V1 = ['Node.GetCumDist', 'Float.Tanh', 'Float.Rot', 'Float.Max']  # v3 seed
# F_up: |sqrt(sum m_up)|
UP_V1 = ['Float.Swap', 'Node.GetCumDist', 
          {'name': 'Node.ForEachChild', 'body': ['Node.GetMUp']}, 
          'Float.Sqrt', 'Float.Abs']
# HALT: old_m_up > |new_m_up|  (always False, so max_bp_iters iterations)
HALT_V1 = ['Float.Abs', 'Float.GT']

# Baseline: pure cum_dist (no BP)
PURE_CUMDIST = ['Node.GetCumDist']  # just return D_i

# V3 best formula: D_i / (M_down / (M_up - EC3))
BELIEF_V3 = ['Float.EvoConst3', 'Float.Sub', 'Float.Div', 'Float.Div']
# EC3 = -0.4021 in log domain → 0.396 actual

# Alternative 1: D_i - M_up (cum_dist minus up-message = lower bound correction)
BELIEF_ALT1 = ['Node.GetMUp', 'Float.Sub']  
# stack: [cd, md, mu]; pop mu, pop cd → cd - mu

# Alternative 2: D_i - alpha * M_up 
BELIEF_ALT2 = ['Node.GetMUp', 'Float.EvoConst0', 'Float.Mul', 'Float.Sub']

# Alternative 3: D_i + EC1 * M_down - EC0 * M_up
BELIEF_ALT3 = ['Node.GetMDown', 'Float.EvoConst1', 'Float.Mul', 
               'Float.Add', 'Node.GetMUp', 'Float.EvoConst0', 'Float.Mul', 
               'Float.Sub']

# Alternative 4: D_i * exp(-M_up) — exponential weighting of up-message
BELIEF_ALT4 = ['Node.GetMUp', 'Float.Neg', 'Float.Exp', 'Float.Mul']

# Alternative 5: D_i - M_up + M_down  (more symmetric)
BELIEF_ALT5 = ['Node.GetMUp', 'Float.Sub', 'Node.GetMDown', 'Float.Add']

# Alternative 6: D_i / (1 + M_up^2)^(1/2) — normalize by m_up magnitude
BELIEF_ALT6 = ['Node.GetMUp', 'Float.Square', 'Float.EvoConst0', 'Float.Add', 
               'Float.Sqrt', 'Float.Div']

# Alternative 7: sqrt(D_i^2 + M_down^2) - M_up
BELIEF_ALT7 = ['Float.Square', 'Node.GetMDown', 'Float.Square', 'Float.Add', 
               'Float.Sqrt', 'Node.GetMUp', 'Float.Sub']

def eval_genome(description, down, up, belief, halt, log_consts=None):
    g = build_genome_from_instrs(down, up, belief, halt, log_consts)
    formula_str = program_to_formula_trace(g.prog_belief, 'belief', g)
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  F_belief = {formula_str}")
    print(f"  EC = {g.evo_constants}")
    t0 = time.time()
    try:
        results = full_evaluation(
            g, Nt=16, Nr=16, mod_order=16,
            n_trials=eval_n, snr_dbs=snrs, max_nodes=max_nodes,
            flops_max=5_000_000, step_max=5000,
            cpp_evaluator=cpp, min_bit_errors=200,
            if_eval_baseline=False)
        dt = time.time() - t0
        for r in results:
            print(f"  SNR={r['snr_db']:.0f}dB  BER={r['evolved_ber']:.6f}  "
                  f"({r['evolved_bit_errors']} errors, {r['evolved_samples']} samples)")
        print(f"  [{dt:.1f}s]")
        return {r['snr_db']: r['evolved_ber'] for r in results}
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return {}

# v3-best constants optimized in gen1: EC0=-1.93, EC1=-0.78, EC2=-0.45, EC3=-0.40
LOG_CONSTS_V3 = [-1.9265, -0.7784, -0.4502, -0.4021]
# Neutral constants
LOG_CONSTS_0 = [0.0, 0.0, 0.0, 0.0]  # all 1.0
# EC0 = 1.0 scaling constant
LOG_CONSTS_HALF = [-0.301, 0.0, 0.0, 0.0]  # EC0=0.5, others=1.0

print("Baselines (max_nodes=1500):")
print(f"  KB16 target: 22dB BER=0.007, 24dB BER=0.001")
print()

r_v3 = eval_genome("V3-best: D_i/(M_down/(M_up-EC3))", 
                   DOWN_V1, UP_V1, BELIEF_V3, HALT_V1, LOG_CONSTS_V3)

r_pure = eval_genome("Pure cum_dist", 
                     DOWN_V1, UP_V1, PURE_CUMDIST, HALT_V1, LOG_CONSTS_V3)

r_alt1 = eval_genome("Alt1: D_i - M_up", 
                     DOWN_V1, UP_V1, BELIEF_ALT1, HALT_V1, LOG_CONSTS_0)

r_alt2 = eval_genome("Alt2: D_i - EC0 * M_up (EC0=0.5)", 
                     DOWN_V1, UP_V1, BELIEF_ALT2, HALT_V1, LOG_CONSTS_HALF)

r_alt3 = eval_genome("Alt3: D_i + EC1*M_down - EC0*M_up", 
                     DOWN_V1, UP_V1, BELIEF_ALT3, HALT_V1, LOG_CONSTS_0)

r_alt4 = eval_genome("Alt4: D_i * exp(-M_up)", 
                     DOWN_V1, UP_V1, BELIEF_ALT4, HALT_V1, LOG_CONSTS_0)

r_alt5 = eval_genome("Alt5: D_i - M_up + M_down", 
                     DOWN_V1, UP_V1, BELIEF_ALT5, HALT_V1, LOG_CONSTS_0)

r_alt7 = eval_genome("Alt7: sqrt(D_i^2 + M_down^2) - M_up", 
                     DOWN_V1, UP_V1, BELIEF_ALT7, HALT_V1, LOG_CONSTS_0)

print("\n" + "="*60)
print("SUMMARY (max_nodes=1500, 22dB BER):")
print(f"  V3-best:  {r_v3.get(22.0, 'N/A'):.6f}")
print(f"  PureDist: {r_pure.get(22.0, 'N/A'):.6f}")
print(f"  Alt1 (D_i-M_up): {r_alt1.get(22.0, 'N/A'):.6f}")
print(f"  Alt2 (D_i-EC*M_up): {r_alt2.get(22.0, 'N/A'):.6f}")
print(f"  Alt3 (linear combo): {r_alt3.get(22.0, 'N/A'):.6f}")
print(f"  Alt4 (exp weighting): {r_alt4.get(22.0, 'N/A'):.6f}")
print(f"  Alt5 (D_i-M_up+M_down): {r_alt5.get(22.0, 'N/A'):.6f}")
print(f"  Alt7 (sqrt combo): {r_alt7.get(22.0, 'N/A'):.6f}")
print(f"  KB16 target: 0.007000")
