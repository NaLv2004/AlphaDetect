"""Quick test to verify batch evaluation produces identical results to sequential."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from bp_main_v2 import (
    StructuredBPEvaluator, random_genome, GenomeIndividual,
    FitnessResult
)

def main():
    print("Testing batch vs sequential evaluation...", flush=True)
    
    evaluator = StructuredBPEvaluator(
        Nt=16, Nr=16, mod_order=16,
        flops_max=3_000_000, max_nodes=500,
        train_samples=50, snr_choices=[22.0],
        step_max=2000, use_cpp=True)
    
    if evaluator.cpp_eval is None:
        print("ERROR: C++ evaluator not available")
        return
    
    # Generate some random genomes
    rng = np.random.RandomState(42)
    n_genomes = 10
    genomes = [random_genome(rng) for _ in range(n_genomes)]
    
    # Build datasets
    ds = evaluator.build_dataset(7000)
    ds_hold = evaluator.build_dataset(17000, n=25)
    
    # Sequential evaluation
    print(f"Sequential evaluation of {n_genomes} genomes...", flush=True)
    seq_results = []
    for g in genomes:
        fit, _ = evaluator.evaluate(g, ds, ds_hold)
        seq_results.append(fit)
    
    # Batch evaluation
    print(f"Batch evaluation of {n_genomes} genomes...", flush=True)
    cpp = evaluator.cpp_eval
    ber_m, fl_m, faults_m, bp_m = cpp.evaluate_batch(genomes, ds)
    ber_h, fl_h, faults_h, bp_h = cpp.evaluate_batch(genomes, ds_hold)
    baseline_ber, _, _ = cpp.evaluate_baselines(ds)
    
    n_main = len(ds)
    n_hold = len(ds_hold)
    n_total = n_main + n_hold
    
    batch_results = []
    for i in range(n_genomes):
        avg_ber = 0.55 * ber_m[i] + 0.45 * ber_h[i]
        avg_fl = 0.55 * fl_m[i] + 0.45 * fl_h[i]
        frac_f = (0.55 * faults_m[i] + 0.45 * faults_h[i]) / max(1, n_main)
        gap = abs(ber_m[i] - ber_h[i])
        ratio = avg_ber / max(baseline_ber, 1e-6)
        avg_bp = (bp_m[i] * n_main + bp_h[i] * n_hold) / max(1, n_total)
        
        fit = FitnessResult(
            ber=avg_ber, mse=0.0, avg_flops=avg_fl,
            code_length=genomes[i].total_length(),
            frac_faults=frac_f, baseline_ber=baseline_ber,
            ber_ratio=ratio, generalization_gap=gap,
            bp_updates=avg_bp,
            nonlocal_bp_updates=0.0,
            bp_gain=0.0)
        batch_results.append(fit)
    
    # Compare
    print(f"\n{'Genome':>8} {'Seq BER':>12} {'Batch BER':>12} {'Match':>8}")
    all_match = True
    for i in range(n_genomes):
        s = seq_results[i]
        b = batch_results[i]
        match = abs(s.ber - b.ber) < 1e-10
        if not match:
            all_match = False
        print(f"  {i:>4d}   {s.ber:12.8f}  {b.ber:12.8f}  {'OK' if match else 'MISMATCH'}")
    
    print(f"\nOverall: {'ALL MATCH' if all_match else 'MISMATCHES FOUND'}")

if __name__ == '__main__':
    main()
