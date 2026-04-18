"""Evaluate OSIC at multiple SNRs with more trials for accurate BER estimation."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from evolution.ir_pool import build_ir_pool
from evolution.materialize import materialize_to_callable
from evolution.mimo_evaluator import (
    qam16_constellation, generate_mimo_sample, _nearest_symbols,
)

def main():
    pool = build_ir_pool()
    constellation = qam16_constellation()
    rng = np.random.default_rng(42)
    Nr, Nt = 16, 16
    n_trials = 2000

    # Find OSIC genome
    osic_genome = None
    for g in pool:
        if hasattr(g.structural_ir, 'name') and 'osic' in g.structural_ir.name.lower():
            osic_genome = g
            break
    if osic_genome is None:
        print("OSIC genome not found!")
        return

    fn = materialize_to_callable(osic_genome)

    print(f"OSIC: {Nr}x{Nt} 16QAM, {n_trials} trials per SNR")
    print("=" * 60)

    for snr_db in [18, 20, 22, 24, 26, 28]:
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
        ber_approx = ser / 4  # Gray coding approximation
        print(f"SNR={snr_db:4.0f} dB: SER = {ser:.6f}  BER ≈ {ber_approx:.6f}  ({errors}/{total})")

if __name__ == "__main__":
    main()
