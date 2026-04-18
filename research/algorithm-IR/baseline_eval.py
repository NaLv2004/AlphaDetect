"""Quick baseline: evaluate all 8 detectors at SNR 24 dB, 16x16 16QAM."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from evolution.ir_pool import build_ir_pool
from evolution.materialize import materialize_to_callable
from evolution.mimo_evaluator import (
    MIMOEvalConfig, MIMOFitnessEvaluator, qam16_constellation,
    generate_mimo_sample, _nearest_symbols,
)

def main():
    pool = build_ir_pool()
    constellation = qam16_constellation()
    rng = np.random.default_rng(42)
    Nr, Nt = 16, 16
    snr_db = 24.0
    n_trials = 500

    print(f"Baseline: {Nr}x{Nt} 16QAM, SNR={snr_db} dB, {n_trials} trials")
    print("=" * 60)

    for genome in pool:
        name = genome.structural_ir.name if genome.structural_ir else genome.algo_id
        try:
            fn = materialize_to_callable(genome)
        except Exception as e:
            print(f"{name:20s}: COMPILE FAILED ({e})")
            continue

        errors = 0
        total = 0
        for _ in range(n_trials):
            try:
                H, x_true, y, sigma2 = generate_mimo_sample(
                    Nr, Nt, constellation, snr_db, rng,
                )
                x_hat = fn(H, y, sigma2, constellation)
                if x_hat is None or len(x_hat) != Nt:
                    errors += Nt
                else:
                    x_hat = _nearest_symbols(x_hat, constellation)
                    errors += int(np.sum(np.abs(x_true - x_hat) > 1e-6))
            except Exception:
                errors += Nt
            total += Nt

        ser = errors / max(total, 1)
        ber = ser  # For 16QAM with gray coding, BER ≈ SER/4, but we use SER as proxy
        print(f"{name:20s}: SER = {ser:.6f}  ({errors}/{total})")

    print("\nNote: SER shown. For 16QAM Gray, BER ≈ SER/4.")

if __name__ == "__main__":
    main()
