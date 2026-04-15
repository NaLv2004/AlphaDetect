"""
Fast ablation study (50 trials per SNR) — for quick verification.

Expected runtime: ~5 minutes each ablation config = ~25 minutes total.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Re-use everything from ablate_best_genome.py
from ablate_best_genome import (
    prog_down_best, prog_up_best, prog_belief_best, prog_halt_best,
    prog_up_zero, prog_down_passthrough, prog_belief_distance,
    EVOLVED_CONSTANTS,
    evaluate_genome_config, print_results, constellation_for
)

if __name__ == '__main__':
    print("=" * 60)
    print("Fast Ablation Study (50 trials) — Best Genome from cpp_test4")
    print("=" * 60)
    print("(using 50 trials/SNR, 600 max_nodes, SNRs 8-16 dB)")

    fast_kwargs = dict(n_trials=50, max_nodes=600, snr_dbs=[8, 10, 12, 14, 16])

    print_results(
        "BASELINE: Full evolved genome (max_bp_iters=3)",
        evaluate_genome_config(
            prog_down_best, prog_up_best, prog_belief_best, prog_halt_best,
            EVOLVED_CONSTANTS, max_bp_iters=3, **fast_kwargs))

    print_results(
        "A1: F_up = 0.0 (trivial, no upward messages)",
        evaluate_genome_config(
            prog_down_best, prog_up_zero, prog_belief_best, prog_halt_best,
            EVOLVED_CONSTANTS, max_bp_iters=3, **fast_kwargs))

    print_results(
        "A2: F_down = pass-through (M_par_down only, no C_i)",
        evaluate_genome_config(
            prog_down_passthrough, prog_up_best, prog_belief_best, prog_halt_best,
            EVOLVED_CONSTANTS, max_bp_iters=3, **fast_kwargs))

    print_results(
        "A3: max_bp_iters=0 (no BP sweeps at all)",
        evaluate_genome_config(
            prog_down_best, prog_up_best, prog_belief_best, prog_halt_best,
            EVOLVED_CONSTANTS, max_bp_iters=0, **fast_kwargs))

    print_results(
        "A4: F_belief = D_i only (pure cumulative distance, no BP messages)",
        evaluate_genome_config(
            prog_down_best, prog_up_best, prog_belief_distance, prog_halt_best,
            EVOLVED_CONSTANTS, max_bp_iters=3, **fast_kwargs))

    print("\nDone.")
