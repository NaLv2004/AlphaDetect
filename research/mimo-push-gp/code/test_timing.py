"""Time a realistic 150-pop initial evaluation."""
import time
import numpy as np
from main import MIMOEvaluator, EvolutionEngine

print('Creating evaluator (80 nodes, SNR 6/8/10)...')
ev = MIMOEvaluator(Nt=8, Nr=16, mod_order=16, flops_max=80000, max_nodes=80,
                   train_samples=10, snr_choices=[6.0, 8.0, 10.0])

engine = EvolutionEngine(pop_size=150, evaluator=ev, fresh_injection_rate=0.2)

print('Creating initial population...')
pop = engine.init_population()
sizes = [len(ind.program) for ind in pop]
print(f'Pop sizes: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f}')

print('Evaluating population...')
t0 = time.time()
engine.evaluate_pop(pop, seed=100)
dt = time.time() - t0
print(f'Evaluation time: {dt:.1f}s')

n_faults = sum(1 for ind in pop if ind.fitness and ind.fitness.frac_faults > 0.5)
best = min(pop, key=lambda x: x.fitness.composite_score() if x.fitness else 1e9)
print(f'Faults: {n_faults}/{len(pop)}')
print(f'Best: BER={best.fitness.ber:.5f} len={best.fitness.code_length} '
      f'flops={best.fitness.avg_flops:.0f}')
print('DONE')
