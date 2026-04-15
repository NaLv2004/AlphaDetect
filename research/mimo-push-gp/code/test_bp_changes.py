"""Quick smoke test for BP-related code changes."""
from vm import MIMOPushVM, Instruction, program_to_oneliner
from bp_decoder import BPStackDecoder, qam16_constellation
from evolution import random_program, random_bp_pattern
import numpy as np

rng = np.random.RandomState(42)

# Test BP pattern generation
for i in range(5):
    prog = random_bp_pattern(rng)
    print(f'BP pattern {i}: {program_to_oneliner(prog)}')

# Test one decode with a BP-pattern program
const = qam16_constellation()
vm = MIMOPushVM(flops_max=500000, step_max=3000)
dec = BPStackDecoder(Nt=4, Nr=8, constellation=const, max_nodes=100, vm=vm)

H = (np.random.randn(8,4) + 1j*np.random.randn(8,4)) / np.sqrt(2)
x = const[np.random.randint(len(const), size=4)]
nv = 0.1
y = H @ x + np.sqrt(nv/2) * (np.random.randn(8) + 1j*np.random.randn(8))

prog = random_bp_pattern(rng)
print(f'\nTest prog: {program_to_oneliner(prog)}')
x_hat, flops = dec.detect(H, y, prog, noise_var=nv)
print(f'bp_updates = {dec.bp_updates}')
print(f'flops = {flops}')
print(f'x_hat shape = {x_hat.shape}')

# Test that evaluate collects bp_updates
from bp_main import BPEvaluator, generate_mimo_sample, ber_calc
ev = BPEvaluator(Nt=4, Nr=8, mod_order=16, flops_max=500000,
                 max_nodes=100, train_samples=4, snr_choices=[10.0],
                 step_max=3000)
ds = ev.build_dataset(seed=0, n=4)
ds_hold = ev.build_dataset(seed=1, n=2)

# Test with a BP-pattern program
prog = random_bp_pattern(rng)
fit, bers = ev.evaluate(prog, ds, ds_hold)
print(f'\nEval result: {fit}')
print(f'bp_updates in fitness: {fit.bp_updates}')

# Test with a regular program
prog2 = random_program(min_size=5, max_size=10, rng=rng, env_bias=0.4)
fit2, bers2 = ev.evaluate(prog2, ds, ds_hold)
print(f'Regular prog: {fit2}')

print('\nAll tests passed!')
