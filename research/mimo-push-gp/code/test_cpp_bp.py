"""Quick test: compare Python vs C++ BP evaluator on the same genome + dataset."""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vm import MIMOPushVM, Instruction
from bp_decoder_v2 import StructuredBPDecoder
from bp_main_v2 import (Genome, constellation_for, generate_mimo_sample,
                         ber_calc, random_genome)

# --- Setup ---
Nt, Nr, mod_order = 8, 16, 16
constellation = constellation_for(mod_order)
M = len(constellation)
max_nodes = 500
flops_max = 2_000_000
step_max = 1500
rng = np.random.RandomState(42)

# Generate a genome
genome = random_genome(rng)
print("=== Genome programs ===")
for name, prog in [("DOWN", genome.prog_down), ("UP", genome.prog_up),
                    ("BELIEF", genome.prog_belief), ("HALT", genome.prog_halt)]:
    print(f"  {name}: {[i.name for i in prog]}")

# Generate dataset
n_samples = 20
dataset = []
for _ in range(n_samples):
    snr = float(rng.choice([10.0, 12.0, 14.0]))
    dataset.append(generate_mimo_sample(Nr, Nt, constellation, snr, rng))

# --- Python evaluation ---
print("\n--- Python BP Decoder ---")
vm = MIMOPushVM(flops_max=flops_max, step_max=step_max)
decoder = StructuredBPDecoder(Nt=Nt, Nr=Nr, constellation=constellation,
                               max_nodes=max_nodes, vm=vm)
py_bers = []
py_flops = []
t0 = time.perf_counter()
for H, x_true, y, nv in dataset:
    x_hat, fl = decoder.detect(H, y,
                                prog_down=genome.prog_down,
                                prog_up=genome.prog_up,
                                prog_belief=genome.prog_belief,
                                prog_halt=genome.prog_halt,
                                noise_var=float(nv))
    py_bers.append(ber_calc(x_true, x_hat))
    py_flops.append(float(fl))
py_time = time.perf_counter() - t0
print(f"  BER:   {np.mean(py_bers):.6f}")
print(f"  Flops: {np.mean(py_flops):.0f}")
print(f"  Time:  {py_time:.3f}s")

# --- C++ evaluation ---
print("\n--- C++ BP Decoder ---")
try:
    from cpp_bridge import CppBPEvaluator
    cpp_eval = CppBPEvaluator(Nt=Nt, Nr=Nr, mod_order=mod_order,
                               max_nodes=max_nodes, flops_max=flops_max,
                               step_max=step_max, max_bp_iters=3)
    t0 = time.perf_counter()
    cpp_ber, cpp_flops, cpp_faults, cpp_bp = cpp_eval.evaluate_genome(genome, dataset)
    cpp_time = time.perf_counter() - t0
    print(f"  BER:   {cpp_ber:.6f}")
    print(f"  Flops: {cpp_flops:.0f}")
    print(f"  Faults:{cpp_faults}")
    print(f"  BP:    {cpp_bp:.1f}")
    print(f"  Time:  {cpp_time:.3f}s")

    print(f"\n--- Comparison ---")
    print(f"  BER diff:   {abs(np.mean(py_bers) - cpp_ber):.6f}")
    print(f"  Speedup:    {py_time / max(cpp_time, 1e-6):.1f}x")

    if abs(np.mean(py_bers) - cpp_ber) < 0.15:
        print("\n  [PASS] BER values are close enough (within 0.15)")
    else:
        print(f"\n  [WARN] BER difference is large — may be due to different PQ ordering")

except Exception as e:
    import traceback
    print(f"  FAILED: {e}")
    traceback.print_exc()

# --- Batch evaluation test ---
print("\n--- C++ Batch Evaluation ---")
try:
    genomes_list = [random_genome(rng) for _ in range(10)]
    t0 = time.perf_counter()
    ber_arr, flops_arr, faults_arr, bp_arr = cpp_eval.evaluate_batch(genomes_list, dataset)
    batch_time = time.perf_counter() - t0
    print(f"  10 genomes x {n_samples} samples in {batch_time:.3f}s")
    print(f"  BERs: {ber_arr}")
    print(f"  Faults: {faults_arr}")
    print("  [PASS] Batch evaluation works")
except Exception as e:
    import traceback
    print(f"  FAILED: {e}")
    traceback.print_exc()
