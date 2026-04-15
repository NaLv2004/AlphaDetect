import time, sys, numpy as np
sys.path.insert(0, '.')
from bp_main_v2 import random_genome, StructuredBPEvaluator, GenomeIndividual

rng = np.random.RandomState(777)

# Generate genomes
print("Generating 120 genomes...", flush=True)
t0 = time.time()
genomes = []
for i in range(120):
    g = random_genome(rng)
    genomes.append(g)
print(f'Generated 120 in {time.time()-t0:.2f}s', flush=True)

# Evaluate with C++
print("Setting up evaluator...", flush=True)
evaluator = StructuredBPEvaluator(
    train_samples=150,
    Nt=16, Nr=16, mod_order=16,
    max_nodes=500, flops_max=3000000,
    step_max=500, snr_choices=[22.0, 24.0],
    use_cpp=True
)

print("Building dataset...", flush=True)
ds = evaluator.build_dataset(7000)
print(f"Dataset: {len(ds)} samples", flush=True)

print("Evaluating batch...", flush=True)
t0 = time.time()
cpp = evaluator.cpp_eval
ber, fl, faults, bp = cpp.evaluate_batch(genomes, ds)
t1 = time.time()
print(f"Batch eval: {t1-t0:.2f}s", flush=True)
for i in range(5):
    print(f"  Genome {i+1}: BER={ber[i]:.5f} FLOPs={fl[i]:.0f} faults={faults[i]} bp={bp[i]:.1f}", flush=True)
print(f"  ...and {115} more genomes", flush=True)
