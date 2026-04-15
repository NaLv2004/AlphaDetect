"""
Test the theoretical insight: Does score = max(cum_dist, floor=0.348) explain
the evolved genome's behavior?

This tests:
1. Pure A*: score = cum_dist
2. Clamped A*: score = max(cum_dist, 0.348)
3. The evolved genome: BASELINE

Expected: Clamped A* ≈ BASELINE (validates our interpretation that F_belief
produces max(cum_dist, EC1) and BP sweeps are irrelevant)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from bp_decoder_v2 import StructuredBPDecoder, qam16_constellation
from stack_decoder import lmmse_detect, kbest_detect
from vm import Instruction, MIMOPushVM
from ablate_best_genome import (
    prog_down_best, prog_up_best, prog_belief_best, prog_halt_best,
    EVOLVED_CONSTANTS, evaluate_genome_config, print_results, constellation_for
)

def make_instr(name):
    return Instruction(name=name)


# --------------------------------------------------------------------------
# A simple pure-A* test using the existing decoder with trivial programs
# --------------------------------------------------------------------------

# A* programs:
# F_down = M_par_down + C_i (cumulative distance): Int.Inc ; Float.Add ; Node.GetParent
# F_up = 0 (constant): Float.Const0
# F_belief = D_i (cumulative distance = top of float stack): Float.Pop ; Float.Pop
#   (pop M_up, pop M_down, leave D_i = cum_dist)
# H_halt = never halt: Bool.False

# Essentially: F_down = cum_dist, F_belief = score = cum_dist
prog_down_astar = [make_instr('Int.Inc'), make_instr('Float.Add'), make_instr('Node.GetParent')]
prog_up_const0 = [make_instr('Float.Const0')]
# F_belief input: float=[D_i, M_down, M_up]. We want score = D_i = cum_dist
# Pop M_up then Pop M_down → top is D_i
prog_belief_astar = [make_instr('Float.Pop'), make_instr('Float.Pop')]
prog_halt_false = [make_instr('Bool.False')]

# Clamped A*: score = max(cum_dist, floor)
# F_belief: float=[D_i, M_down, M_up]
# Pop M_up, Pop M_down → float=[D_i]
# Push EC1 (=0.348 from EVOLVED_CONSTANTS[1]=10^(-0.4585)≈0.348)
# Float.Max → score = max(D_i, 0.348)
prog_belief_clamped = [
    make_instr('Float.Pop'),         # pop M_up, top is M_down
    make_instr('Float.Pop'),         # pop M_down, top is D_i = cum_dist
    make_instr('Float.EvoConst1'),   # push EC1 ≈ 0.348
    make_instr('Float.Max'),         # max(cum_dist, 0.348)
]


if __name__ == '__main__':
    print("=" * 60)
    print("Clamped A* vs Pure A* vs Evolved BASELINE")
    print("=" * 60)
    print("EC1 = 10^(-0.4585) =", round(10**(-0.4585), 4))
    print("Evolved constants:", np.round(EVOLVED_CONSTANTS, 4))
    print()

    snrs = [8, 10, 12, 14, 16]
    fast_kwargs = dict(n_trials=50, max_nodes=600, snr_dbs=snrs)

    print_results(
        "1. EVOLVED BASELINE (max_bp_iters=3)",
        evaluate_genome_config(
            prog_down_best, prog_up_best, prog_belief_best, prog_halt_best,
            EVOLVED_CONSTANTS, max_bp_iters=3, **fast_kwargs))

    print_results(
        "2. Clamped A*: score = max(cum_dist, EC1≈0.348) (1 BP iter, correct)",
        evaluate_genome_config(
            prog_down_astar, prog_up_const0, prog_belief_clamped, prog_halt_false,
            EVOLVED_CONSTANTS, max_bp_iters=1, **fast_kwargs))

    print_results(
        "3. Pure A*: score = cum_dist (1 BP iter, correct)",
        evaluate_genome_config(
            prog_down_astar, prog_up_const0, prog_belief_astar, prog_halt_false,
            EVOLVED_CONSTANTS, max_bp_iters=1, **fast_kwargs))

    # Also try with different floor values
    for floor_log in [-1.5, -1.0, -0.5, 0.0, 0.5]:
        consts_variant = EVOLVED_CONSTANTS.copy()
        consts_variant[1] = 10**floor_log  # EC1 = floor value
        label = f"4. Clamped A* floor={10**floor_log:.3f} (log={floor_log}, 1 iter)"
        print_results(label,
            evaluate_genome_config(
                prog_down_astar, prog_up_const0, prog_belief_clamped, prog_halt_false,
                consts_variant, max_bp_iters=1, **fast_kwargs))

    print("\nDone.")
