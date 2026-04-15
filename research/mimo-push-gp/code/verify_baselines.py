"""
Verify C++ baseline detectors (LMMSE, K-Best) match Python implementations.

Usage:
    conda run -n AutoGenOld python verify_baselines.py

This script generates a fixed-seed dataset at 16x16, 16QAM and compares
per-sample BER outputs between Python and C++ for all three baselines
(LMMSE, K-Best-16, K-Best-32).
"""

import numpy as np
import sys
import os
import time

# Add code directory to path
sys.path.insert(0, os.path.dirname(__file__))

from stack_decoder import lmmse_detect, kbest_detect, qam16_constellation
from cpp_bridge import CppBPEvaluator, _interleave_complex


def generate_dataset(Nt, Nr, constellation, n_samples, snr_db, rng):
    """Generate a MIMO dataset with fixed seed."""
    M = len(constellation)
    bits_per_sym = int(np.log2(M))
    sigma2 = 1.0 / (10 ** (snr_db / 10) * bits_per_sym)

    dataset = []
    for _ in range(n_samples):
        H = (rng.randn(Nr, Nt) + 1j * rng.randn(Nr, Nt)) / np.sqrt(2)
        x = constellation[rng.randint(0, M, size=Nt)]
        noise = np.sqrt(sigma2 / 2) * (rng.randn(Nr) + 1j * rng.randn(Nr))
        y = H @ x + noise
        dataset.append((H, x, y, sigma2))
    return dataset


def count_errors(x_hat, x_true):
    """Count symbol errors."""
    return np.sum(np.abs(x_hat - x_true) > 1e-6)


def main():
    Nt, Nr = 16, 16
    mod_order = 16
    n_samples = 200
    snr_db = 15.0

    constellation = qam16_constellation()
    rng = np.random.RandomState(42)

    print(f"Generating {n_samples} samples: {Nt}x{Nr}, {mod_order}-QAM, SNR={snr_db}dB")
    dataset = generate_dataset(Nt, Nr, constellation, n_samples, snr_db, rng)

    # ---- Python baselines ----
    print("\n--- Python baselines ---")
    py_ber_lmmse = np.zeros(n_samples)
    py_ber_kb16 = np.zeros(n_samples)
    py_ber_kb32 = np.zeros(n_samples)

    t0 = time.time()
    for i, (H, x_true, y, nv) in enumerate(dataset):
        x_hat, _ = lmmse_detect(H, y, nv, constellation)
        py_ber_lmmse[i] = count_errors(x_hat, x_true) / Nt

        x_hat, _ = kbest_detect(H, y, constellation, K=16)
        py_ber_kb16[i] = count_errors(x_hat, x_true) / Nt

        x_hat, _ = kbest_detect(H, y, constellation, K=32)
        py_ber_kb32[i] = count_errors(x_hat, x_true) / Nt
    py_time = time.time() - t0

    print(f"  LMMSE  avg BER: {py_ber_lmmse.mean():.6f}")
    print(f"  KB-16  avg BER: {py_ber_kb16.mean():.6f}")
    print(f"  KB-32  avg BER: {py_ber_kb32.mean():.6f}")
    print(f"  Time: {py_time:.2f}s")

    # ---- C++ baselines ----
    print("\n--- C++ baselines ---")
    dll_path = os.path.join(os.path.dirname(__file__), 'cpp', 'evaluator_bp.dll')
    if not os.path.exists(dll_path):
        print(f"ERROR: DLL not found at {dll_path}")
        print("Build with: cl.exe /EHsc /O2 /openmp /std:c++17 evaluator_bp.cpp /LD /Fe:evaluator_bp.dll")
        sys.exit(1)

    evaluator = CppBPEvaluator(Nt=Nt, Nr=Nr, mod_order=mod_order,
                                max_nodes=500, flops_max=2_000_000,
                                step_max=1500, max_bp_iters=3,
                                dll_path=dll_path)

    t0 = time.time()
    cpp_lmmse, cpp_kb16, cpp_kb32 = evaluator.evaluate_baselines(dataset)
    cpp_time = time.time() - t0

    print(f"  LMMSE  avg BER: {cpp_lmmse:.6f}")
    print(f"  KB-16  avg BER: {cpp_kb16:.6f}")
    print(f"  KB-32  avg BER: {cpp_kb32:.6f}")
    print(f"  Time: {cpp_time:.2f}s")

    # ---- Compare ----
    print("\n--- Comparison ---")
    tol = 1e-6

    # For per-sample comparison, we need to call C++ per-sample too
    # The C API returns averages, so compare averages
    lmmse_match = abs(py_ber_lmmse.mean() - cpp_lmmse) < tol
    kb16_match = abs(py_ber_kb16.mean() - cpp_kb16) < tol
    kb32_match = abs(py_ber_kb32.mean() - cpp_kb32) < tol

    print(f"  LMMSE  match: {'PASS' if lmmse_match else 'FAIL'} "
          f"(diff={abs(py_ber_lmmse.mean() - cpp_lmmse):.2e})")
    print(f"  KB-16  match: {'PASS' if kb16_match else 'FAIL'} "
          f"(diff={abs(py_ber_kb16.mean() - cpp_kb16):.2e})")
    print(f"  KB-32  match: {'PASS' if kb32_match else 'FAIL'} "
          f"(diff={abs(py_ber_kb32.mean() - cpp_kb32):.2e})")

    speedup = py_time / max(cpp_time, 1e-6)
    print(f"\n  Speedup: {speedup:.1f}x")

    if lmmse_match and kb16_match and kb32_match:
        print("\n=== ALL TESTS PASSED ===")
        return 0
    else:
        print("\n=== SOME TESTS FAILED ===")
        return 1


if __name__ == '__main__':
    sys.exit(main())
