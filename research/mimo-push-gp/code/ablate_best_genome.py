"""
Ablation study of the best genome from cpp_test4.

For each ablation, we replace one program with a trivial version and measure
BER change. This tells us which programs contribute genuine information.

Ablations:
  A1 - F_up = 0.0 (leaf already 0, so inner nodes get 0 aggregation)
  A2 - F_down = pass-through (return M_par_down unchanged)
  A3 - No BP sweeps (max_bp_iters=0)
  A4 - F_belief = D_i (pure cumulative distance)

Usage:
    conda run -n AutoGenOld python -B ablate_best_genome.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from vm import Instruction, MIMOPushVM, program_to_string, program_to_oneliner
from bp_decoder_v2 import StructuredBPDecoder, qam16_constellation, qpsk_constellation

def constellation_for(mod_order):
    if mod_order == 16:
        return qam16_constellation()
    elif mod_order == 4:
        return qpsk_constellation()
    else:
        raise ValueError(f"Unsupported mod_order {mod_order}")
from stack_decoder import lmmse_detect, kbest_detect

# ----- Best genome from cpp_test4 gen 4 ------
# From log:
#   DOWN:[Int.Inc ; Float.Add ; Node.GetParent]
#   UP:[Float.ConstHalf ; Float.ConstHalf ; Float.EvoConst1 ; Float.Sub ; Exec.DoTimes([]) ; Mat.Row ; Float.Div ; Float.Max ; Float.Sqrt ; Float.Log ; Float.Dup]
#   BEL:[Float.Swap ; Float.ConstHalf ; Float.Swap ; Float.EvoConst1 ; Int.Sub ; Int.Inc ; Float.Max]
#   HALT:[Bool.Or ; Node.ReadMem ; Float.Swap ; Float.LT]
#   LOGC: need to extract

# We'll load the best genome from the log file and re-evaluate manually.
# For now, define the programs exactly.

def make_instr(name):
    return Instruction(name=name)

prog_down_best = [
    make_instr('Int.Inc'),
    make_instr('Float.Add'),
    make_instr('Node.GetParent'),
]

PROGS_UP_OPCODES = [
    'Float.ConstHalf',
    'Float.ConstHalf',
    'Float.EvoConst1',
    'Float.Sub',
    'Exec.DoTimes',
    'Mat.Row',
    'Float.Div',
    'Float.Max',
    'Float.Sqrt',
    'Float.Log',
    'Float.Dup',
]

prog_up_best = [make_instr(n) for n in PROGS_UP_OPCODES]

PROGS_BEL_OPCODES = [
    'Float.Swap',
    'Float.ConstHalf',
    'Float.Swap',
    'Float.EvoConst1',
    'Int.Sub',
    'Int.Inc',
    'Float.Max',
]
prog_belief_best = [make_instr(n) for n in PROGS_BEL_OPCODES]

PROGS_HALT_OPCODES = [
    'Bool.Or',
    'Node.ReadMem',
    'Float.Swap',
    'Float.LT',
]
prog_halt_best = [make_instr(n) for n in PROGS_HALT_OPCODES]

# Evolved constants (from log Gen 4 LOGC: actual = 10^logc)
# LOGC: [0.0111,-0.4585,1.0875,-1.7998]
EVOLVED_CONSTANTS = np.array([10**0.0111, 10**(-0.4585), 10**1.0875, 10**(-1.7998)])


# ---- Trivial programs for ablation ----

# A1: Trivial F_up = returns 0.0 (stack will just have 0 after leaf m_up=0)
#    Actually: for non-leaf nodes, children's m_up are on stack.
#    A trivial F_up that returns 0.0 regardless:
prog_up_zero = [make_instr('Float.Const0')]  # push 0.0, ignore children

# A2: Trivial F_down = pass-through = return M_par_down unchanged
#    Input stack for F_down: [M_par_down, C_i]. Output: M_par_down (the pass-through).
#    Float.Pop removes C_i from stack, leaving M_par_down.
prog_down_passthrough = [make_instr('Float.Pop')]  # pop C_i, leave M_par_down

# A4: Trivial F_belief = return D_i only (pure cumulative distance, no BP messages)
#    Input: [D_i, M_down, M_up]. Need to return D_i.
#    Pop M_up, Pop M_down → leaves D_i
prog_belief_distance = [make_instr('Float.Pop'), make_instr('Float.Pop')]  # pop M_up, M_down


def evaluate_genome_config(prog_down, prog_up, prog_belief, prog_halt,
                           evo_constants, max_bp_iters,
                           Nt=8, Nr=16, mod_order=16,
                           n_trials=200, snr_dbs=None, max_nodes=600,
                           seed=2027):
    if snr_dbs is None:
        snr_dbs = [8, 10, 12, 14, 16]
    
    constellation = constellation_for(mod_order)
    vm = MIMOPushVM(flops_max=600000, step_max=1000)
    vm.evolved_constants = evo_constants
    # Patch max_bp_iters on the decoder
    decoder = StructuredBPDecoder(Nt=Nt, Nr=Nr, constellation=constellation,
                                   max_nodes=max_nodes, vm=vm)
    decoder.max_bp_iters = max_bp_iters
    
    rng = np.random.RandomState(seed)
    results = []
    for snr in snr_dbs:
        ber_list, lm_list = [], []
        for _ in range(n_trials):
            # Channel model
            H = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
            sym_idx = np.random.randint(0, len(constellation), size=Nt)
            x = constellation[sym_idx]
            sigma2 = Nt / (10 ** (snr / 10.0))
            noise = (np.random.randn(Nr) + 1j * np.random.randn(Nr)) * np.sqrt(sigma2 / 2)
            y = H @ x + noise
            
            xh, _ = decoder.detect(H, y,
                                   prog_down=prog_down,
                                   prog_up=prog_up,
                                   prog_belief=prog_belief,
                                   prog_halt=prog_halt,
                                   noise_var=float(sigma2))
            ber = np.mean(np.abs(x - xh) > 1e-6)
            ber_list.append(ber)
            
            xl, _ = lmmse_detect(H, y, sigma2, constellation)
            lm_list.append(np.mean(np.abs(x - xl) > 1e-6))
        
        results.append({
            'snr': snr,
            'ber': float(np.mean(ber_list)),
            'lmmse_ber': float(np.mean(lm_list)),
            'ratio': float(np.mean(ber_list)) / max(float(np.mean(lm_list)), 1e-9),
        })
    return results


def print_results(label, results):
    print(f"\n{label}")
    print("  SNR |  Evo BER |  LMMSE  | Ratio")
    print("  ----|----------|---------|------")
    for r in results:
        print(f"  {r['snr']:3.0f} | {r['ber']:.5f} | {r['lmmse_ber']:.5f} | {r['ratio']:.3f}")


if __name__ == '__main__':
    print("=" * 60)
    print("Structured BP Ablation Study — Best Genome from cpp_test4")
    print("=" * 60)
    print("(using 200 trials/SNR, 600 max_nodes, SNRs 8-16 dB)")

    # Baseline: best evolved genome
    print_results(
        "BASELINE: Full evolved genome (max_bp_iters=3)",
        evaluate_genome_config(
            prog_down_best, prog_up_best, prog_belief_best, prog_halt_best,
            EVOLVED_CONSTANTS, max_bp_iters=3))

    # A1: F_up = 0.0
    print_results(
        "A1: F_up = 0.0 (trivial, no upward messages)",
        evaluate_genome_config(
            prog_down_best, prog_up_zero, prog_belief_best, prog_halt_best,
            EVOLVED_CONSTANTS, max_bp_iters=3))

    # A2: F_down = pass-through
    print_results(
        "A2: F_down = pass-through (M_par_down only, no C_i)",
        evaluate_genome_config(
            prog_down_passthrough, prog_up_best, prog_belief_best, prog_halt_best,
            EVOLVED_CONSTANTS, max_bp_iters=3))

    # A3: No BP sweeps
    print_results(
        "A3: max_bp_iters=0 (no BP sweeps at all)",
        evaluate_genome_config(
            prog_down_best, prog_up_best, prog_belief_best, prog_halt_best,
            EVOLVED_CONSTANTS, max_bp_iters=0))

    # A4: F_belief = D_i only
    print_results(
        "A4: F_belief = D_i only (pure cumulative distance, no BP messages)",
        evaluate_genome_config(
            prog_down_best, prog_up_best, prog_belief_distance, prog_halt_best,
            EVOLVED_CONSTANTS, max_bp_iters=3))

    print("\nDone.")
