"""Focused BER measurement: OSIC at 24 dB, 16x16, 16-QAM, many trials."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from evolution.ir_pool import build_ir_pool
from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig, generate_mimo_sample, qam16_constellation
from evolution.materialize import materialize, materialize_to_callable

rng_pool = np.random.default_rng(42)
pool = build_ir_pool(rng_pool)

constellation = qam16_constellation()

# Test each pool algorithm with 2000 trials at 24 dB
for g in pool:
    fn = materialize_to_callable(g)
    rng = np.random.default_rng(123)
    n_trials = 2000
    sym_errors = 0
    bit_errors = 0
    total_syms = 0
    total_bits = 0
    for _ in range(n_trials):
        H, x_true, y, sigma2 = generate_mimo_sample(16, 16, constellation, 24.0, rng)
        try:
            x_hat = fn(H, y, sigma2, constellation)
            if x_hat is not None and len(x_hat) == 16:
                # Map to nearest constellation point
                for i in range(16):
                    dists = np.abs(constellation - x_hat[i])**2
                    x_hat[i] = constellation[np.argmin(dists)]
                sym_errs = int(np.sum(np.abs(x_true - x_hat) > 1e-6))
                sym_errors += sym_errs
                # Count bit errors (Gray code: assume nearest symbol error = 1 bit)
                # For exact BER: map symbols to bits and compare
                for i in range(16):
                    idx_true = np.argmin(np.abs(constellation - x_true[i])**2)
                    idx_hat = np.argmin(np.abs(constellation - x_hat[i])**2)
                    # XOR bit indices for Gray-coded 16-QAM
                    diff_bits = idx_true ^ idx_hat
                    bit_errors += bin(diff_bits).count('1')
                total_bits += 16 * 4  # 4 bits per 16-QAM symbol
            else:
                sym_errors += 16
                bit_errors += 16 * 4
                total_bits += 16 * 4
        except Exception:
            sym_errors += 16
            bit_errors += 16 * 4
            total_bits += 16 * 4
        total_syms += 16

    ser = sym_errors / total_syms
    ber = bit_errors / total_bits
    print(f"{g.algo_id:12s}: SER={ser:.6f} ({sym_errors}/{total_syms}), "
          f"BER={ber:.6f} ({bit_errors}/{total_bits})")
