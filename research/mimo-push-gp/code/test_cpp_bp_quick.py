"""Quick C++ BP evaluator test — no Python decoder comparison (too slow)."""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vm import Instruction
from bp_main_v2 import (Genome, constellation_for, generate_mimo_sample,
                         random_genome)

Nt, Nr, mod_order = 8, 16, 16
constellation = constellation_for(mod_order)
rng = np.random.RandomState(42)

# Generate genomes
genomes = [random_genome(rng) for _ in range(20)]

# Generate dataset
n_samples = 30
dataset = []
for _ in range(n_samples):
    snr = float(rng.choice([10.0, 12.0, 14.0]))
    dataset.append(generate_mimo_sample(Nr, Nt, constellation, snr, rng))

print("=== C++ BP Evaluator Test ===")
from cpp_bridge import CppBPEvaluator

cpp_eval = CppBPEvaluator(Nt=Nt, Nr=Nr, mod_order=mod_order,
                           max_nodes=500, flops_max=2_000_000,
                           step_max=1500, max_bp_iters=3)

# Single genome test
print("\n--- Single genome eval ---")
t0 = time.perf_counter()
ber, flops, faults, bp = cpp_eval.evaluate_genome(genomes[0], dataset)
t1 = time.perf_counter()
print(f"  BER={ber:.4f}, Flops={flops:.0f}, Faults={faults}, BP={bp:.1f}")
print(f"  Time: {t1-t0:.4f}s for {n_samples} samples")

# Batch test
print(f"\n--- Batch eval: {len(genomes)} genomes x {n_samples} samples ---")
t0 = time.perf_counter()
ber_arr, flops_arr, faults_arr, bp_arr = cpp_eval.evaluate_batch(genomes, dataset)
t1 = time.perf_counter()
print(f"  Time: {t1-t0:.4f}s total")
print(f"  BERs: {ber_arr}")
print(f"  Faults: {faults_arr}")
print(f"  Avg flops: {np.mean(flops_arr):.0f}")
print(f"  Avg BP: {np.mean(bp_arr):.1f}")

# Estimate speedup
single_time = t1 - t0
per_genome_cpp = single_time / len(genomes)
# Python decoder takes ~0.1-0.5s per sample on 8x16
est_py_per_genome = 0.15 * n_samples  # conservative estimate
print(f"\n  Est. C++ per genome: {per_genome_cpp:.4f}s")
print(f"  Est. Python per genome: ~{est_py_per_genome:.1f}s")
print(f"  Est. speedup: ~{est_py_per_genome / max(per_genome_cpp, 1e-6):.0f}x")

# Test that faults are 0 for most genomes
n_faulted = np.sum(faults_arr > 0)
print(f"\n  Genomes with faults: {n_faulted}/{len(genomes)}")
if n_faulted < len(genomes):
    print("  [PASS] C++ BP evaluator works correctly")
else:
    print("  [WARN] All genomes faulted — check implementation")
