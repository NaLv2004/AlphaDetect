"""Quick formula test with correct stack semantics.
Evaluates different F_belief formulas to find ones that beat the v3-best.
Faster: fewer samples, stop at first formula that beats target.
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bp_main_v2 import (
    Genome, Instruction, full_evaluation, qam16_constellation,
    program_to_formula_trace, N_EVO_CONSTS
)
from cpp_bridge import CppBPEvaluator

def mk(item):
    if isinstance(item, dict):
        body = [mk(b) for b in item.get('body', [])]
        return Instruction(name=item['name'], code_block=body)
    return Instruction(name=item)

def build_genome(down_instrs, up_instrs, belief_instrs, halt_instrs, log_consts=None):
    if log_consts is None:
        log_consts = np.zeros(N_EVO_CONSTS)
    return Genome(
        prog_down=[mk(i) for i in down_instrs],
        prog_up=[mk(i) for i in up_instrs],
        prog_belief=[mk(i) for i in belief_instrs],
        prog_halt=[mk(i) for i in halt_instrs],
        log_constants=np.array(log_consts),
    )

snrs = [22.0, 24.0]
eval_n = 3000  # samples per SNR (reduced for speed)
max_nodes = 1500

cpp = CppBPEvaluator(Nt=16, Nr=16, mod_order=16,
                     max_nodes=max_nodes, flops_max=5_000_000,
                     step_max=5000, max_bp_iters=3)

# ============================================================
# STACK SEMANTICS (CRITICAL):
# F_belief input stack (bottom→top): [cum_dist, m_down, m_up]
# Float.Sub: b=fst.pop() (top), a=fst.pop() (second) → push (a - b)
#   = (second - top)
#   Initial: pop b=m_up, pop a=m_down → push (m_down - m_up)
# Float.Add: same order → push (a + b) = (m_down + m_up)
# Float.Rot: (cum_dist, m_down, m_up) → (m_down, m_up, cum_dist)
#            i.e. moves item at _items[-3] to top
# Float.Swap: exchanges top two
# ============================================================

# F_down (v3 seed): max(tanh(cum_dist), m_parent_down)
DOWN_V3 = ['Node.GetCumDist', 'Float.Tanh', 'Float.Rot', 'Float.Max']

# F_up variations
# UP_V3: |sqrt(Sum_children(m_up))|
UP_V3 = ['Float.Swap', 'Node.GetCumDist', 
          {'name': 'Node.ForEachChild', 'body': ['Node.GetMUp']}, 
          'Float.Sqrt', 'Float.Abs']

# UP_MINSUM: min_children(m_up + local_dist) — theoretically correct lower bound
UP_MINSUM = [
    'Float.Const0',  # initialize accumulator to 0
    'Float.Pop',     # clear
    {'name': 'Node.ForEachChild', 'body': ['Node.GetMUp', 'Node.GetLocalDist', 'Float.Add']},
    'Float.Sqrt', 'Float.Abs'
]
# Actually let's use a simpler approach: just sum m_up values (same as v3 up)
# The key insight: with F_up = min-sum, the BELIEF can then use D_i - m_up

# UP_DIRECT: sum of children m_up (keep ForEachChild accumulation)
UP_DIRECT = [
    {'name': 'Node.ForEachChild', 'body': ['Node.GetMUp']},
    'Float.Abs'
]

# HALT
HALT_NEVER = ['Float.Abs', 'Float.GT']  # old_mup > |mup| → rarely true

# ============================================================
# F_belief formulas (with correct stack ops)
# Initial stack: [cum_dist, m_down, m_up]
# ============================================================

LOG_CONSTS_V3 = [-1.9265, -0.7784, -0.4502, -0.4021]
LOG_CONSTS_0 = [0.0, 0.0, 0.0, 0.0]  # all EC = 1.0

# BELIEF_CUMDIST: just D_i (baseline — no BP)
# Process: push D_i → peek D_i
BELIEF_CUMDIST = ['Node.GetCumDist']  # push cum_dist, peek = cum_dist

# BELIEF_V3: D_i / (M_down / (M_up - EC3))
BELIEF_V3 = ['Float.EvoConst3', 'Float.Sub', 'Float.Div', 'Float.Div']
# Stack: [D_i, M_down, M_up]
#   EC3 pushed → [D_i, M_down, M_up, EC3]
#   Sub: a=EC3, b=M_up → M_up-EC3 → [D_i, M_down, (M_up-EC3)]
#   Div: a=(M_up-EC3), b=M_down → M_down/(M_up-EC3) → [D_i, M_down/(M_up-EC3)]
#   Div: a=M_down/(M_up-EC3), b=D_i → D_i / (M_down/(M_up-EC3)) → peek

# BELIEF_CUMDIST_MINUS_MUP: D_i - m_up (lower bound)
# Stack: [D_i, M_down, M_up]
#   Float.Rot → [M_down, M_up, D_i]
#   Float.Swap → [M_down, D_i, M_up]
#   Float.Sub: a=M_up, b=D_i → D_i - M_up → [M_down, D_i - M_up]
# peek = D_i - M_up ✓
BELIEF_DI_MINUS_MUP = ['Float.Rot', 'Float.Swap', 'Float.Sub']

# BELIEF_MDOWN_MINUS_MUP: m_down - m_up
# Stack: [D_i, M_down, M_up]
#   Float.Sub: a=M_up, b=M_down → M_down - M_up → [D_i, M_down - M_up]
# peek = M_down - M_up
BELIEF_MDOWN_MINUS_MUP = ['Float.Sub']

# BELIEF_DI_PLUS_MDOWN_MINUS_MUP: D_i + M_down - M_up
# Stack: [D_i, M_down, M_up]
#   Float.Sub → [D_i, M_down - M_up]
#   Float.Add → [D_i + M_down - M_up]  (a = M_down-M_up, b = D_i → D_i + (M_down-M_up))
# peek = D_i + M_down - M_up
BELIEF_DI_PLUS_MDOWN_MINUS_MUP = ['Float.Sub', 'Float.Add']

# BELIEF_DI_MINUS_ALPHA_MUP: D_i - alpha * m_up
# Stack: [D_i, M_down, M_up]
#   Float.Rot → [M_down, M_up, D_i]
#   Float.Swap → [M_down, D_i, M_up]
#   Float.EvoConst0 → [M_down, D_i, M_up, EC0]
#   Float.Mul → [M_down, D_i, M_up * EC0]
#   Float.Sub: a = M_up*EC0, b = D_i → D_i - M_up*EC0 → [M_down, D_i - M_up*EC0]
# peek = D_i - EC0 * M_up
BELIEF_DI_MINUS_ALPHA_MUP = ['Float.Rot', 'Float.Swap', 'Float.EvoConst0', 'Float.Mul', 'Float.Sub']

# BELIEF_DI_MINUS_MUP_PLUS_ALPHA_MDOWN: D_i - M_up + alpha * M_down
# Stack: [D_i, M_down, M_up]
#   Float.Sub → [D_i, M_down - M_up]    (M_down - M_up)
#   Float.EvoConst0 → [D_i, M_down - M_up, EC0]
#   Float.Mul → [D_i, (M_down - M_up) * EC0]
#   Float.Add → [D_i + (M_down - M_up) * EC0]
# peek = D_i + EC0 * (M_down - M_up)
BELIEF_DI_WEIGHTED_DIFF = ['Float.Sub', 'Float.EvoConst0', 'Float.Mul', 'Float.Add']

# BELIEF_EXP_WEIGHTED: D_i * exp(-M_up)
# Stack: [D_i, M_down, M_up]
#   Float.Rot → [M_down, M_up, D_i]
#   Float.Swap → [M_down, D_i, M_up]
#   Float.Neg → [M_down, D_i, -M_up]
#   Float.Exp → [M_down, D_i, exp(-M_up)]
#   Float.Mul → [M_down, D_i * exp(-M_up)]
# peek = D_i * exp(-M_up)
BELIEF_EXP_WEIGHTED = ['Float.Rot', 'Float.Swap', 'Float.Neg', 'Float.Exp', 'Float.Mul']

# BELIEF_DI_MINUS_TANH_MUP: D_i - tanh(M_up) (normalized BP message)
# Stack: [D_i, M_down, M_up]
#   Float.Rot → [M_down, M_up, D_i]
#   Float.Swap → [M_down, D_i, M_up]
#   Float.Tanh → [M_down, D_i, tanh(M_up)]
#   Float.Sub → [M_down, D_i - tanh(M_up)]
# peek = D_i - tanh(M_up)
BELIEF_DI_MINUS_TANH_MUP = ['Float.Rot', 'Float.Swap', 'Float.Tanh', 'Float.Sub']

# BELIEF_DI_PLUS_MDOWN: D_i + M_down (no M_up)
# Stack: [D_i, M_down, M_up]
#   Float.Pop → [D_i, M_down]
#   Float.Add → [D_i + M_down]
# peek = D_i + M_down
BELIEF_DI_PLUS_MDOWN = ['Float.Pop', 'Float.Add']

# BELIEF_DI_TIMES_MDOWN: D_i * M_down (scaling by down-pass)
# Stack: [D_i, M_down, M_up]
#   Float.Pop → [D_i, M_down]
#   Float.Mul → [D_i * M_down]
BELIEF_DI_TIMES_MDOWN = ['Float.Pop', 'Float.Mul']


def eval_formula(description, up, belief, log_consts=None, n=eval_n):
    g = build_genome(DOWN_V3, up, belief, HALT_NEVER, log_consts)
    print(f"\n{'='*55}")
    print(f"  {description}")
    t0 = time.time()
    try:
        results = full_evaluation(
            g, Nt=16, Nr=16, mod_order=16,
            n_trials=n, snr_dbs=snrs, max_nodes=max_nodes,
            flops_max=5_000_000, step_max=5000,
            cpp_evaluator=cpp, min_bit_errors=100,  # relaxed for speed
            if_eval_baseline=False)
        dt = time.time() - t0
        bers = {}
        for r in results:
            print(f"  SNR={r['snr_db']:.0f}dB  BER={r['evolved_ber']:.6f}  "
                  f"({r['evolved_bit_errors']} err, {r['evolved_samples']} samples)")
            bers[r['snr_db']] = r['evolved_ber']
        print(f"  [{dt:.1f}s]")
        return bers
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {}

print("="*55)
print("QUICK BELIEF FORMULA TEST")
print(f"max_nodes={max_nodes}, eval_n={eval_n}, min_errors=100")
print(f"KB16 targets: 22dB BER=0.007, 24dB BER=0.001")
print()

results = {}

results['v3'] = eval_formula("V3: D_i/(M_down/(M_up-EC3))", 
                              UP_V3, BELIEF_V3, LOG_CONSTS_V3)

results['cumdist'] = eval_formula("Pure cum_dist (no BP)", 
                                  UP_V3, BELIEF_CUMDIST, LOG_CONSTS_0)

results['di_minus_mup'] = eval_formula("D_i - M_up (lower bound)", 
                                       UP_V3, BELIEF_DI_MINUS_MUP, LOG_CONSTS_0)

results['mdown_minus_mup'] = eval_formula("M_down - M_up (BP msgs only)", 
                                          UP_V3, BELIEF_MDOWN_MINUS_MUP, LOG_CONSTS_0)

results['di_plus_mdown_minus_mup'] = eval_formula("D_i + M_down - M_up", 
                                                   UP_V3, BELIEF_DI_PLUS_MDOWN_MINUS_MUP, LOG_CONSTS_0)

results['di_minus_alpha_mup'] = eval_formula("D_i - alpha*M_up (EC0=1)", 
                                             UP_V3, BELIEF_DI_MINUS_ALPHA_MUP, LOG_CONSTS_0)

results['di_weighted_diff'] = eval_formula("D_i + EC0*(M_down - M_up)", 
                                           UP_V3, BELIEF_DI_WEIGHTED_DIFF, LOG_CONSTS_0)

results['exp_weighted'] = eval_formula("D_i * exp(-M_up)", 
                                       UP_V3, BELIEF_EXP_WEIGHTED, LOG_CONSTS_0)

results['di_minus_tanh_mup'] = eval_formula("D_i - tanh(M_up) (normalized)", 
                                            UP_V3, BELIEF_DI_MINUS_TANH_MUP, LOG_CONSTS_0)

# Try with different F_up (direct sum, no sqrt)
results['direct_di_minus_mup'] = eval_formula("D_i - M_up (direct F_up)", 
                                              UP_DIRECT, BELIEF_DI_MINUS_MUP, LOG_CONSTS_0)

print("\n" + "="*55)
print("SUMMARY (22dB BER):")
for name, bers in sorted(results.items(), key=lambda x: x[1].get(22.0, 1.0)):
    print(f"  {name:35s}: {bers.get(22.0, 'N/A'):.6f}  24dB: {bers.get(24.0, 'N/A'):.6f}")
print(f"  {'KB16 target':35s}: 0.007000  24dB: 0.001000")
