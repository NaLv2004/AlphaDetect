"""Quick test of evaluation pipeline."""
import sys
import numpy as np
from main import MIMOEvaluator, EvolutionEngine, program_to_oneliner
from evolution import random_program

print('Creating evaluator...')
ev = MIMOEvaluator(Nt=8, Nr=16, mod_order=16, flops_max=80000, max_nodes=80,
                   train_samples=4, snr_choices=[8.0])

print('Building dataset...')
ds = ev.build_dataset(100)
print(f'Dataset: {len(ds)} samples')

print('Testing dist baseline...')
dist_bers, dist_fl, dist_f = ev._eval_per_sample(ev._dist_prog, ds, ev._dist_decoder)
print(f'Dist BERs: {dist_bers}')
print(f'Dist FLOPs: {dist_fl}')

print('Testing random program...')
rng = np.random.RandomState(42)
prog = random_program(min_size=3, max_size=8, rng=rng)
print(f'Prog: {program_to_oneliner(prog)}')
fit, per_sample = ev.evaluate(prog, ds, ds[:2], dist_bers)
print(f'Fitness: {fit}')
print(f'Per-sample BERs: {per_sample}')

print('\nTesting engine init + evaluate...')
engine = EvolutionEngine(pop_size=10, evaluator=ev)
pop = engine.init_population()
print(f'Pop sizes: {[len(ind.program) for ind in pop]}')
engine.evaluate_pop(pop, seed=100)
best = min(pop, key=lambda x: x.fitness.composite_score())
print(f'Best: BER={best.fitness.ber:.5f} len={best.fitness.code_length}')
print(f'Best prog: {program_to_oneliner(best.program)}')
print('DONE')
