"""Compare gen4_bp vs pure_a_star at different max_nodes budgets.

Key question: Is gen4_bp's advantage due to the BP scoring or just eval_max_nodes=2000?
"""
import sys, os
sys.path.insert(0, r'D:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\code')
os.chdir(r'D:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\code')

import numpy as np
from vm import Instruction
from bp_main import full_evaluation

gen4_bp = [
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

pure_a_star = [Instruction('Float.GetMMSELB')]

# Also test: identity (no correction, pure best-first with cumulative distance)
no_correction = [Instruction('Float.Const0')]

snr_dbs = [10.0, 12.0, 14.0, 16.0]
n_trials = 100

print("Node budget comparison: gen4_bp vs pure_a_star vs no_correction")
print("=" * 80)

for max_nodes in [200, 500, 1000, 2000]:
    print(f"\n--- max_nodes = {max_nodes} ---")
    print(f"{'Program':<20} {'SNR':>5} {'Evo BER':>10} {'LMMSE BER':>10} {'Ratio':>8} {'KB32':>10}")
    print("-" * 65)

    for prog_name, prog in [('gen4_bp', gen4_bp), ('pure_a_star', pure_a_star), ('no_correction', no_correction)]:
        results = full_evaluation(
            prog=prog,
            Nt=8, Nr=16, mod_order=16,
            n_trials=n_trials,
            snr_dbs=snr_dbs,
            max_nodes=max_nodes,
            flops_max=5_000_000,
        )
        for r in results:
            ratio = r['evolved_ber'] / max(r['lmmse_ber'], 1e-6)
            print(f"{prog_name:<20} {r['snr_db']:>5.1f} {r['evolved_ber']:>10.5f} "
                  f"{r['lmmse_ber']:>10.5f} {ratio:>8.3f} {r['kbest32_ber']:>10.5f}")
        print()
