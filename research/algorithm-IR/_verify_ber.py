"""Verify BER of evolution winner with 2000 trials."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from evolution.mimo_evaluator import generate_mimo_sample, qam16_constellation

# Load the evolved detector
best_src = open("results/evolution_16x16_16qam_24dB/best_detector.py").read()
ns = {"np": np}
exec(compile(best_src, "<evolved>", "exec"), ns)
detect_fn = ns["ep"]

constellation = qam16_constellation()
rng = np.random.default_rng(123)
n_trials = 5000

sym_errors = 0
bit_errors = 0
total_syms = 0
total_bits = 0

for t in range(n_trials):
    H, x_true, y, sigma2 = generate_mimo_sample(16, 16, constellation, 24.0, rng)
    try:
        x_hat = detect_fn(H, y, sigma2, constellation)
        if x_hat is not None and len(x_hat) == 16:
            for i in range(16):
                dists = np.abs(constellation - x_hat[i])**2
                x_hat[i] = constellation[np.argmin(dists)]
            sym_errs = int(np.sum(np.abs(x_true - x_hat) > 1e-6))
            sym_errors += sym_errs
            for i in range(16):
                idx_true = np.argmin(np.abs(constellation - x_true[i])**2)
                idx_hat = np.argmin(np.abs(constellation - x_hat[i])**2)
                diff_bits = idx_true ^ idx_hat
                bit_errors += bin(diff_bits).count('1')
            total_bits += 16 * 4
        else:
            sym_errors += 16
            bit_errors += 16 * 4
            total_bits += 16 * 4
    except Exception as e:
        print(f"Trial {t}: error {e}")
        sym_errors += 16
        bit_errors += 16 * 4
        total_bits += 16 * 4
    total_syms += 16

ser = sym_errors / total_syms
ber = bit_errors / total_bits
print(f"Evolution winner (EP detector):")
print(f"  SER = {ser:.6f} ({sym_errors}/{total_syms})")
print(f"  BER = {ber:.6f} ({bit_errors}/{total_bits})")
print(f"  BER < 1e-3: {'YES' if ber < 1e-3 else 'NO'}")
