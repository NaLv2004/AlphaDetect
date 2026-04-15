"""Test A*-style formulas with ForEachChildMin to see if they beat v3-best.
Focused test with correct stack semantics.
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bp_main_v2 import (
    Genome, Instruction, full_evaluation, qam16_constellation,
    N_EVO_CONSTS
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
eval_n = 2000
max_nodes = 1500

cpp = CppBPEvaluator(Nt=16, Nr=16, mod_order=16,
                     max_nodes=max_nodes, flops_max=5_000_000,
                     step_max=5000, max_bp_iters=3)

LOG_CONSTS_V3 = [-1.9265, -0.7784, -0.4502, -0.4021]
LOG_CONSTS_0 = [0.0, 0.0, 0.0, 0.0]

# === COMPONENT PROGRAMS ===
# F_down (v3): max(tanh(cum_dist), m_parent_down) 
DOWN_V3 = ['Node.GetCumDist', 'Float.Tanh', 'Float.Rot', 'Float.Max']
# F_down simple pass-through: m_down = m_parent_down + local_dist
DOWN_PASSTHRU = ['Float.Add']  # stack[m_parent_down, local_dist] → m_par + ld
# HALT
HALT_V3 = ['Float.Abs', 'Float.GT']

# === F_UP VARIANTS ===
# V3: |sqrt(sum_children(m_up))|
UP_V3_SUM = ['Float.Swap', 'Node.GetCumDist', 
              {'name': 'Node.ForEachChild', 'body': ['Node.GetMUp']}, 
              'Float.Sqrt', 'Float.Abs']

# A*-style: min_children(m_up) — lower bound on future cost
UP_MINMUP = [{'name': 'Node.ForEachChildMin', 'body': ['Node.GetMUp']}]

# A*-style: min_children(m_up + local_dist) — full admissible heuristic
UP_MINSUM = [{'name': 'Node.ForEachChildMin', 'body': ['Node.GetMUp', 'Node.GetLocalDist', 'Float.Add']}]

# Mix: sqrt(min_children(m_up))
UP_SQRT_MIN = [{'name': 'Node.ForEachChildMin', 'body': ['Node.GetMUp']}, 'Float.Sqrt']

# === F_BELIEF VARIANTS ===
# Stack inputs (bottom→top): [cum_dist, m_down, m_up]
# Float.Sub: b=second, a=top → push (b - a) = second - top
# Float.Add: same order → push (second + top)
# Float.Rot: (a, b, c) → (b, c, a), moves a to top where a was at items[-3]

# V3: D_i / (M_down / (M_up - EC3))
BELIEF_V3 = ['Float.EvoConst3', 'Float.Sub', 'Float.Div', 'Float.Div']

# A*-1: D_i - M_up (simplest lower bound correction)
# [D_i, M_down, M_up]: Float.Rot→[M_down,M_up,D_i]; Float.Swap→[M_down,D_i,M_up]; 
# Float.Sub b=D_i,a=M_up → D_i-M_up
BELIEF_ASTAR_SIMPLE = ['Float.Rot', 'Float.Swap', 'Float.Sub']

# A*-2: D_i - alpha*M_up
BELIEF_ASTAR_SCALED = ['Float.Rot', 'Float.Swap', 'Float.EvoConst0', 'Float.Mul', 'Float.Sub']

# A*-3: D_i + EC0 * (M_down - M_up)
# [D_i, M_down, M_up]: Float.Sub b=M_down,a=M_up→M_down-M_up; then EC0*; then + D_i
BELIEF_ASTAR_DIFF = ['Float.Sub', 'Float.EvoConst0', 'Float.Mul', 'Float.Add']

# A*-4: M_down - M_up (no cum_dist — pure BP guidance)
# [D_i, M_down, M_up]: Float.Sub → (M_down - M_up) already on stack
BELIEF_BP_ONLY = ['Float.Sub']

def eval_genome(description, up, belief, halt=HALT_V3, log_consts=None, down=DOWN_V3, n=eval_n):
    g = build_genome(down, up, belief, halt, log_consts)
    print(f"\n{'='*55}")
    print(f"  {description}")
    t0 = time.time()
    try:
        results = full_evaluation(
            g, Nt=16, Nr=16, mod_order=16,
            n_trials=n, snr_dbs=snrs, max_nodes=max_nodes,
            flops_max=5_000_000, step_max=5000,
            cpp_evaluator=cpp, min_bit_errors=100,
            if_eval_baseline=False)
        dt = time.time() - t0
        bers = {}
        for r in results:
            print(f"  SNR={r['snr_db']:.0f}dB  BER={r['evolved_ber']:.6f}  "
                  f"({r['evolved_bit_errors']} err, {r['evolved_samples']} samp)")
            bers[r['snr_db']] = r['evolved_ber']
        print(f"  [{dt:.1f}s]")
        return bers
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {}

print("="*55)
print("A*-STYLE FORMULA TEST WITH ForEachChildMin")
print(f"max_nodes={max_nodes}, min_errors=100")
print(f"KB16 targets: 22dB BER=0.007, 24dB BER=0.001")

results = {}

results['v3_baseline'] = eval_genome(
    "V3: sum-up + D_i/(M_down/(M_up-EC3))", 
    UP_V3_SUM, BELIEF_V3, log_consts=LOG_CONSTS_V3)

results['astar_simple'] = eval_genome(
    "A*-1: min-up + D_i - M_up", 
    UP_MINMUP, BELIEF_ASTAR_SIMPLE)

results['astar_sum_simple'] = eval_genome(
    "A*-1 w/sum-up: sum-up + D_i - M_up",
    UP_V3_SUM, BELIEF_ASTAR_SIMPLE)

results['astar_fullsumheur'] = eval_genome(
    "A*-2: min(m_up+ld)-up + D_i - M_up", 
    UP_MINSUM, BELIEF_ASTAR_SIMPLE)

results['astar_scaled'] = eval_genome(
    "A*-scaled: min-up + D_i - 1.0*M_up", 
    UP_MINMUP, BELIEF_ASTAR_SCALED)

# Try different scaling for astar
for alpha_log in [-0.301, 0.0, 0.301, 0.602]:
    alpha = 10**alpha_log
    lc = [alpha_log, 0.0, 0.0, 0.0]
    results[f'astar_a{alpha:.2f}'] = eval_genome(
        f"A*-scaled: alpha={alpha:.2f} (EC0={alpha:.2f})", 
        UP_MINMUP, BELIEF_ASTAR_SCALED, log_consts=lc)

results['astar_diff'] = eval_genome(
    "A*-diff: min-up + D_i + EC0*(M_down - M_up)", 
    UP_MINMUP, BELIEF_ASTAR_DIFF)

results['sqrt_min_astar'] = eval_genome(
    "A*-sqrt: sqrt(min)-up + D_i - M_up", 
    UP_SQRT_MIN, BELIEF_ASTAR_SIMPLE)

print("\n" + "="*55)
print("SUMMARY (22dB BER) — sorted best to worst:")
for name, bers in sorted(results.items(), key=lambda x: x[1].get(22.0, 1.0)):
    b22 = bers.get(22.0, float('nan'))
    b24 = bers.get(24.0, float('nan'))
    flag = " *** BEATS TARGET!" if b22 < 0.007 else ("  <-- PROMISING" if b22 < 0.015 else "")
    print(f"  {name:40s}: 22dB={b22:.6f}  24dB={b24:.6f}{flag}")
print(f"  {'KB16 target':40s}: 22dB=0.007000  24dB=0.001000")
