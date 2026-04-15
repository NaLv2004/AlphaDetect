"""Full evaluation of specific interesting programs from v4 experiments.

Run with: C:\ProgramData\anaconda3\envs\AutoGenOld\python.exe -B eval_best_programs.py
"""
import sys
import os
sys.path.insert(0, r'D:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\code')
os.chdir(r'D:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\code')

import json
import numpy as np
from vm import Instruction
from bp_main import full_evaluation

def make_program(description):
    """Create programs from string descriptions."""
    pass

# Gen 4 program from v4_3: ForEachSibling([Float.GetMMSELB, Node.GetScore, Float.Mul, Node.SetScore])
gen4_prog = [
    Instruction('Int.LT'),
    Instruction('Mat.PeekAt'),
    Instruction('Graph.NodeCount'),
    Instruction('Exec.If', code_block=[Instruction('Node.IsExpanded')], code_block2=[Instruction('Int.GT')]),
    Instruction('Node.ForEachSibling', code_block=[
        Instruction('Float.GetMMSELB'),
        Instruction('Node.GetScore'),
        Instruction('Float.Mul'),
        Instruction('Node.SetScore'),
    ]),
    Instruction('Int.Const2'),
    Instruction('Mat.Row'),
]

# Gen 5 program from v4_3 (best composite, but bad full eval)
gen5_prog_str = """
Node.ForEachAncestor([Node.GetLocalDist]) ; Node.GetLayer ; 
Exec.ForEachSymbol([Int.LT, Node.GetSymIm, Vec.PeekAt]) ; 
Node.GetLocalDist ; Bool.Dup ; Vec.ElementAt ; Mat.PeekAtIm ; Float.Sqrt ; 
Graph.NodeCount ; Float.GetMMSELB ; Vec.PeekAtIm ; Node.Pop ; Node.GetCumDist ; 
Node.NumChildren ; Node.ForEachChild([Int.LT, Int.Swap]) ; Float.Sub ; Bool.And
"""

# Simple A*-like program (just return mmse_lb as correction)
pure_a_star_prog = [
    Instruction('Float.GetMMSELB'),
]

# A*-like with GetCumDist explicitly
cum_plus_mmse = [
    Instruction('Node.GetCumDist'),
    Instruction('Float.GetMMSELB'),
    Instruction('Float.Add'),
    Instruction('Float.Neg'),
    Instruction('Node.GetCumDist'),
    Instruction('Float.Add'),
    # score = cum_dist + mmse_lb, then subtract cum_dist (correction = mmse_lb)
]

# The KEY program: sibling score normalization using cum_dist difference
sibling_score_corr = [
    # For each sibling: write sibling.cum_dist + our_local_dist_contribution
    Instruction('Node.ForEachSibling', code_block=[
        Instruction('Node.GetCumDist'),    # sibling cum_dist
        Instruction('Float.GetMMSELB'),    # our mmse lb (from outer env)
        Instruction('Float.Add'),          # cum_dist + mmse_lb = A* score
        Instruction('Node.SetScore'),      # update sibling
    ]),
    Instruction('Node.GetCumDist'),
    Instruction('Float.GetMMSELB'),
    Instruction('Float.Add'),
]

programs = {
    'gen4_bp': gen4_prog,
    'pure_a_star': pure_a_star_prog,
    'sibling_score_corr': sibling_score_corr,
}

print("Full evaluation of candidate programs")
print("=" * 60)
print(f"{'Program':<25} {'SNR':>5} {'Evo BER':>10} {'LMMSE BER':>10} {'Ratio':>8} {'KB32':>10}")
print("-" * 70)

for prog_name, prog in programs.items():
    results = full_evaluation(
        prog=prog,
        Nt=8, Nr=16, mod_order=16,
        n_trials=200,
        snr_dbs=[10.0, 12.0, 14.0, 16.0],
        max_nodes=2000,
        flops_max=5_000_000,
    )
    for r in results:
        ratio = r['evolved_ber'] / max(r['lmmse_ber'], 1e-6)
        print(f"{prog_name:<25} {r['snr_db']:>5.1f} {r['evolved_ber']:>10.5f} "
              f"{r['lmmse_ber']:>10.5f} {ratio:>8.3f} {r['kbest32_ber']:>10.5f}")
    print()
