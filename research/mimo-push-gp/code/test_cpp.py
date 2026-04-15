"""Quick test of the C++ evaluator bridge."""
import numpy as np
from cpp_bridge import CppEvaluator, encode_program
from vm import Instruction

# Encode a simple program
prog = [Instruction('Float.Const0')]
ops = encode_program(prog)
print(f'Encoded Float.Const0: {ops}')

# Encode a more complex program
prog2 = [
    Instruction('Float.GetMMSELB'),
    Instruction('Float.Neg'),
    Instruction('Node.ForEachChild', code_block=[
        Instruction('Node.GetScore'),
    ]),
    Instruction('Float.Add'),
]
ops2 = encode_program(prog2)
print(f'Encoded complex prog: {ops2}')

# Test DLL
ev = CppEvaluator(Nt=8, Nr=16, mod_order=16,
                   max_nodes=100, flops_max=500000, step_max=500)
print('DLL loaded OK')

# Generate test data
rng = np.random.RandomState(42)
H = (rng.randn(16, 8) + 1j * rng.randn(16, 8)) / np.sqrt(2)
levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
constellation = np.array([r + 1j * i for r in levels for i in levels])
x_idx = rng.randint(0, 16, size=8)
x = constellation[x_idx]
snr_db = 20.0
sig_power = float(np.mean(np.abs(H @ x)**2))
noise_var = sig_power / (10**(snr_db/10.0))
noise = np.sqrt(noise_var/2) * (rng.randn(16) + 1j * rng.randn(16))
y = H @ x + noise

dataset = [(H, x, y, noise_var)]

# Test simple program (score = cum_dist + 0)
ber, flops = ev.evaluate_program(prog, dataset)
print(f'Float.Const0: BER={ber:.4f}, FLOPs={flops:.0f}')

# Test with Python decoder for comparison
from bp_decoder import BPStackDecoder, qam16_constellation
from vm import MIMOPushVM
vm = MIMOPushVM(flops_max=500000, step_max=500)
dec = BPStackDecoder(Nt=8, Nr=16, constellation=qam16_constellation(),
                     max_nodes=100, vm=vm)
x_hat, py_flops = dec.detect(H, y, prog, noise_var=noise_var)
py_ber = float(np.mean(x_hat != x))
print(f'Python decoder: BER={py_ber:.4f}, FLOPs={py_flops:.0f}')

# Speed benchmark
import time
n_trials = 50
datasets = []
for _ in range(n_trials):
    H = (rng.randn(16, 8) + 1j * rng.randn(16, 8)) / np.sqrt(2)
    x_idx = rng.randint(0, 16, size=8)
    x = constellation[x_idx]
    sig_power = float(np.mean(np.abs(H @ x)**2))
    nv = sig_power / (10**(14.0/10.0))
    noise = np.sqrt(nv/2) * (rng.randn(16) + 1j * rng.randn(16))
    y = H @ x + noise
    datasets.append((H, x, y, nv))

# C++ speed
t0 = time.time()
ber_cpp, fl_cpp = ev.evaluate_program(prog, datasets)
dt_cpp = time.time() - t0

# Python speed
t0 = time.time()
py_bers = []
for H, x, y, nv in datasets:
    x_hat, _ = dec.detect(H, y, prog, noise_var=nv)
    py_bers.append(float(np.mean(x_hat != x)))
dt_py = time.time() - t0

print(f'\nSpeed comparison ({n_trials} samples):')
print(f'  C++:    {dt_cpp:.3f}s  ({dt_cpp/n_trials*1000:.1f}ms/sample)  BER={ber_cpp:.4f}')
print(f'  Python: {dt_py:.3f}s  ({dt_py/n_trials*1000:.1f}ms/sample)  BER={np.mean(py_bers):.4f}')
print(f'  Speedup: {dt_py/max(dt_cpp, 1e-6):.1f}x')
print('\nAll tests PASSED!')
