"""Quick test of A*-style formulas — only most promising variants.
Fewer samples, stops early at min_errors.
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
eval_n = 1500   # Cap at 1500 samples per SNR
min_errors = 30  # Quick check: 30 errors enough for ranking
max_nodes = 500  # Reduced for speed (evolution uses 500 too)

cpp = CppBPEvaluator(Nt=16, Nr=16, mod_order=16,
                     max_nodes=max_nodes, flops_max=2_000_000,
                     step_max=3000, max_bp_iters=3)

LOG_CONSTS_V3 = [-1.9265, -0.7784, -0.4502, -0.4021]

# === COMPONENT PROGRAMS ===
DOWN_V3 = ['Node.GetCumDist', 'Float.Tanh', 'Float.Rot', 'Float.Max']
HALT_V3 = ['Float.Abs', 'Float.GT']

# F_UP variants
UP_V3_SUM = ['Float.Swap', 'Node.GetCumDist', 
              {'name': 'Node.ForEachChild', 'body': ['Node.GetMUp']}, 
              'Float.Sqrt', 'Float.Abs']
UP_MINMUP = [{'name': 'Node.ForEachChildMin', 'body': ['Node.GetMUp']}]
UP_MINSUM = [{'name': 'Node.ForEachChildMin', 'body': ['Node.GetMUp', 'Node.GetLocalDist', 'Float.Add']}]

# F_BELIEF variants (stack: bottom→top = [D_i/cum_dist, M_down, M_up])
# Float.Sub: pops top(=M_up)=b, then second(=M_down)=a, pushes a-b = M_down-M_up
# Float.Add: same pop order → M_down + M_up
# Float.Rot: [D_i, M_down, M_up] → [M_down, M_up, D_i]
# Float.Swap: swaps top two

BELIEF_V3 = ['Float.EvoConst3', 'Float.Sub', 'Float.Div', 'Float.Div']

# CORRECT A*: D_i + M_up = g(n) + h(n)
# [D_i, M_down, M_up]: Swap→[D_i,M_up,M_down]; Pop→[D_i,M_up]; Add→D_i+M_up
BELIEF_ASTAR_CORRECT = ['Float.Swap', 'Float.Pop', 'Float.Add']

# D_i - M_up (WRONG sign for A*, but let's test it)
# [D_i, M_down, M_up]: Rot→[M_down,M_up,D_i]; Swap→[M_down,D_i,M_up]; Sub b=D_i,a=M_up→M_up-D_i
# Wait: Float.Sub pops b=top=M_up, then a=second=D_i → a-b = D_i-M_up
# So: Rot→[M_down,M_up,D_i]; Swap→[M_down,D_i,M_up]; Sub(b=M_up,a=D_i)→D_i-M_up
BELIEF_ASTAR_DI_MU = ['Float.Rot', 'Float.Swap', 'Float.Sub']

# M_down - M_up: Sub → Md - Mu (discards D_i)
BELIEF_BP_ONLY = ['Float.Sub']

# D_i + M_down: Swap→[D_i,M_up,M_down]; Pop→[D_i,M_up]; ... no
# Actually: [D_i,M_down,M_up]: Pop(Mu)→[D_i,M_down]; Add(b=M_down,a=D_i)→D_i+M_down
BELIEF_DI_PLUS_MD = ['Float.Pop', 'Float.Add']

# D_i alone (ignore BP messages — pure breadth first ordered by cumulative distance)  
# [D_i,M_down,M_up]: Pop,Pop → [D_i] (only cum_dist left)
BELIEF_DI_ONLY = ['Float.Pop', 'Float.Pop']

# M_up alone (only heuristic, no past cost)
# [D_i,M_down,M_up]: Rot→[M_down,M_up,D_i]; Swap→[M_down,D_i,M_up]; 
# Actually: [D_i,M_down,M_up]; Rot→[M_down,M_up,D_i]; Pop(D_i)→[M_down,M_up]; Pop(M_up)... no that's two pops
# Simpler: Pop(M_up top)→[D_i,M_down]; Pop(M_down)→[D_i]  
# Let's get M_up: Swap→[D_i,M_up,M_down]; Pop(M_down)→[D_i,M_up]; Pop(D_i)→? no
# Actually: [D_i,M_down,M_up]; Rot→[M_down,M_up,D_i]; Pop(D_i top)→[M_down,M_up]; Pop(M_up top)→[M_down]... 
# Getting M_up only: Swap→[D_i,M_up,M_down]; Rot→[M_up,M_down,D_i]; Pop(D_i)→[M_up,M_down]; Pop(M_down)→[M_up]
BELIEF_MUP_ONLY = ['Float.Swap', 'Float.Rot', 'Float.Pop', 'Float.Pop']

# M_down + alpha*(D_i - M_up): complex sum
# Stack [Di, Md, Mu]: Sub→[Di, Md-Mu], Rot→[Md-Mu, Di, ...] -- complex

def eval_genome(description, up, belief, halt=HALT_V3, log_consts=None, down=DOWN_V3, n=eval_n):
    g = build_genome(down, up, belief, halt, log_consts)
    print(f"\n{'='*55}")
    print(f"  {description}")
    t0 = time.time()
    try:
        results = full_evaluation(
            g, Nt=16, Nr=16, mod_order=16,
            n_trials=n,   # will stop at n samples or min_errors, whichever is LATER  
            snr_dbs=snrs, max_nodes=max_nodes,
            flops_max=5_000_000, step_max=5000,
            cpp_evaluator=cpp, min_bit_errors=min_errors,
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
print("QUICK A*-STYLE FORMULA TEST (min_errors=50)")
print(f"max_nodes={max_nodes}, KB16: 22dB=0.007, 24dB=0.001")

results = {}

results['v3_baseline'] = eval_genome(
    "V3: sum-up + D_i/(M_down/(M_up-EC3))", 
    UP_V3_SUM, BELIEF_V3, log_consts=LOG_CONSTS_V3)

# TRUE A* = g(n) + h(n) where h(n) = min_children(m_up)
results['astar_correct'] = eval_genome(
    "TRUE A*: min-up + D_i + M_up", 
    UP_MINMUP, BELIEF_ASTAR_CORRECT)

# D_i + M_up with sum-up (not pure A* but interesting)
results['astar_correct_sumup'] = eval_genome(
    "A*-sum-up: sum-up + D_i + M_up",
    UP_V3_SUM, BELIEF_ASTAR_CORRECT)

# A*-full: min(m_up+ld)-up + D_i + M_up
results['astar_full'] = eval_genome(
    "A*-full: min(mu+ld)-up + D_i + M_up", 
    UP_MINSUM, BELIEF_ASTAR_CORRECT)

# D_i - M_up (wrong sign but let's test anyway)
results['astar_di_minus_mu'] = eval_genome(
    "D_i-M_up: min-up + D_i - M_up", 
    UP_MINMUP, BELIEF_ASTAR_DI_MU)

# BP-only: M_down - M_up (sum up)
results['bp_only_sum'] = eval_genome(
    "BP: sum-up + M_down - M_up", 
    UP_V3_SUM, BELIEF_BP_ONLY)

# BP-only: M_down - M_up (min up)
results['bp_only_min'] = eval_genome(
    "BP: min-up + M_down - M_up", 
    UP_MINMUP, BELIEF_BP_ONLY)

# D_i + M_down (cumulative cost + downward message)
results['di_plus_md'] = eval_genome(
    "D_i+Md: min-up + D_i + M_down",
    UP_MINMUP, BELIEF_DI_PLUS_MD)

# D_i alone (pure cumulative distance, ignore BP)
results['di_only'] = eval_genome(
    "D_i only: min-up + cum_dist", 
    UP_MINMUP, BELIEF_DI_ONLY)

# M_up only (pure heuristic, no past cost)
results['mup_only'] = eval_genome(
    "M_up only: min-up + heuristic only", 
    UP_MINMUP, BELIEF_MUP_ONLY)

print("\n" + "="*55)
print("SUMMARY (22dB BER) — sorted best to worst:")
for name, bers in sorted(results.items(), key=lambda x: x[1].get(22.0, 1.0)):
    b22 = bers.get(22.0, float('nan'))
    b24 = bers.get(24.0, float('nan'))
    flag = " *** BEATS TARGET!" if b22 < 0.007 else ("  <-- PROMISING" if b22 < 0.020 else "")
    print(f"  {name:45s}: 22dB={b22:.6f}  24dB={b24:.6f}{flag}")
print(f"  {'KB16 target':45s}: 22dB=0.007000  24dB=0.001000")
