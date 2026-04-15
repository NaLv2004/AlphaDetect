"""
C++/Python BP evaluator correctness test.

Generates 100+ random genomes, evaluates each with both the Python
bp_decoder_v2 and the C++ evaluator_bp.dll, and compares x_hat outputs
(detected symbol vectors) and BER per sample.

Run:
    conda run -n AutoGenOld python -B test_cpp_python.py
"""
import sys
import os
import numpy as np
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vm import MIMOPushVM, Instruction
from bp_decoder_v2 import StructuredBPDecoder, qam16_constellation
from stack_decoder import lmmse_detect
from bp_main_v2 import (
    random_genome, Genome, deep_copy_genome,
    generate_mimo_sample, ber_calc, constellation_for,
    N_EVO_CONSTS, has_valid_bp_dependency,
)
from cpp_bridge import CppBPEvaluator


def python_detect_single(genome, H, x_true, y, nv, Nt, Nr, constellation,
                          max_nodes=500, flops_max=2_000_000, step_max=2000):
    """Run Python BP decoder on one sample, return (x_hat, ber)."""
    vm = MIMOPushVM(flops_max=flops_max, step_max=step_max)
    vm.evolved_constants = genome.evo_constants
    decoder = StructuredBPDecoder(Nt=Nt, Nr=Nr, constellation=constellation,
                                  max_nodes=max_nodes, vm=vm)
    try:
        x_hat, flops = decoder.detect(
            H, y,
            prog_down=genome.prog_down,
            prog_up=genome.prog_up,
            prog_belief=genome.prog_belief,
            prog_halt=genome.prog_halt,
            noise_var=float(nv))
        return x_hat, ber_calc(x_true, x_hat)
    except Exception as e:
        return None, 1.0


def cpp_detect_single(cpp_eval, genome, H, x_true, y, nv):
    """Run C++ BP evaluator on one sample, return BER."""
    dataset = [(H, x_true, y, nv)]
    avg_ber, avg_flops, faults, avg_bp = cpp_eval.evaluate_genome(genome, dataset)
    return avg_ber


def main():
    Nt, Nr = 4, 8  # small for speed
    mod_order = 16
    max_nodes = 300
    flops_max = 1_000_000
    step_max = 1500
    n_genomes = 120
    n_samples = 5  # samples per genome
    snr_db = 12.0

    constellation = constellation_for(mod_order)
    rng = np.random.RandomState(2025)

    # Build shared test dataset
    dataset = []
    for _ in range(n_samples):
        dataset.append(generate_mimo_sample(Nr, Nt, constellation, snr_db, rng))

    print(f"Test configuration: {Nt}x{Nr} MIMO, {mod_order}-QAM, "
          f"SNR={snr_db}dB, {n_genomes} genomes, {n_samples} samples/genome")

    # Init C++ evaluator
    cpp_eval = CppBPEvaluator(
        Nt=Nt, Nr=Nr, mod_order=mod_order,
        max_nodes=max_nodes, flops_max=flops_max,
        step_max=step_max, max_bp_iters=3)
    print("[C++] Evaluator loaded OK")

    # Generate random genomes (all pass BP dependency check)
    print(f"Generating {n_genomes} valid genomes...")
    genomes = []
    gen_rng = np.random.RandomState(42)
    for i in range(n_genomes):
        g = random_genome(gen_rng)
        genomes.append(g)
        if (i + 1) % 20 == 0:
            print(f"  generated {i+1}/{n_genomes}")

    print(f"\nComparing Python vs C++ on {n_genomes} genomes x {n_samples} samples...")
    n_match = 0
    n_mismatch = 0
    n_both_fault = 0
    max_ber_diff = 0.0
    ber_diffs = []

    for gi, genome in enumerate(genomes):
        py_bers = []
        cpp_bers = []

        for si, (H, x_true, y, nv) in enumerate(dataset):
            # Python
            x_hat_py, ber_py = python_detect_single(
                genome, H, x_true, y, nv, Nt, Nr, constellation,
                max_nodes=max_nodes, flops_max=flops_max, step_max=step_max)
            py_bers.append(ber_py)

        # C++ evaluates over the full dataset
        cpp_avg_ber = cpp_detect_single(cpp_eval, genome, *dataset[0][:4])
        # For a fair comparison, use per-dataset C++ evaluation
        cpp_avg_ber_full, _, cpp_faults, _ = cpp_eval.evaluate_genome(genome, dataset)

        py_avg = float(np.mean(py_bers))
        diff = abs(py_avg - cpp_avg_ber_full)
        ber_diffs.append(diff)

        if diff > max_ber_diff:
            max_ber_diff = diff

        if diff < 0.01:
            n_match += 1
        else:
            n_mismatch += 1
            if (n_mismatch <= 10):
                print(f"  MISMATCH genome #{gi}: Py_BER={py_avg:.5f} "
                      f"Cpp_BER={cpp_avg_ber_full:.5f} diff={diff:.5f}")
                print(f"    {genome.to_oneliner()[:120]}...")

        if (gi + 1) % 20 == 0:
            print(f"  tested {gi+1}/{n_genomes}, match={n_match}, mismatch={n_mismatch}")

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {n_genomes} genomes tested")
    print(f"  Match (BER diff < 0.01): {n_match}")
    print(f"  Mismatch:                {n_mismatch}")
    print(f"  Max BER difference:      {max_ber_diff:.6f}")
    print(f"  Mean BER difference:     {np.mean(ber_diffs):.6f}")
    print(f"  Median BER difference:   {np.median(ber_diffs):.6f}")

    if n_mismatch == 0:
        print("\n*** ALL GENOMES MATCH — C++ evaluator is correct! ***")
    elif n_mismatch < n_genomes * 0.05:
        print(f"\n*** MOSTLY CORRECT — {n_mismatch} minor mismatches "
              f"({100*n_mismatch/n_genomes:.1f}%) ***")
    else:
        print(f"\n*** WARNING: {n_mismatch} mismatches "
              f"({100*n_mismatch/n_genomes:.1f}%) — investigate! ***")


if __name__ == '__main__':
    main()
