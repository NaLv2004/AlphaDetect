"""Empirically check whether FLOAT_CLAMP=10 hurts OMS BP decoding.

Compares OMS (seeded genome) BER across SNR with two clamp settings:
  * tight  = 10.0   (current default, suspected to truncate large LLR sums)
  * loose  = 1e6    (effectively unclamped for BP-relevant magnitudes)

Uses the *Python* fitness path (use_cpp_fitness=False) so the clamp
actually applies (the C++ side is not yet updated).
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

import pushgp.types as ptypes
from pushgp_ldpc.adapter import oms_seed_genome
from pushgp_ldpc.eval import FitnessConfig
from pushgp_ldpc.eval_logged import evaluate_genome_with_ber


def run_one(clamp: float, cfg: FitnessConfig):
    ptypes.set_float_clamp(clamp)
    t0 = time.time()
    metrics = evaluate_genome_with_ber(oms_seed_genome(), cfg)
    return metrics, time.time() - t0


def main():
    cfg = FitnessConfig(
        info_len_A=1024,
        code_length_E=2048,
        snr_list=(0.0, 1.0, 2.0, 3.0, 4.0),
        n_frames_per_snr=20,
        max_iter=25,
        use_cpp_fitness=False,   # MUST be False so clamp applies
    )
    print(f"# OMS BP decode under varying FLOAT_CLAMP")
    print(f"# config: A={cfg.info_len_A} E={cfg.code_length_E} "
          f"frames/snr={cfg.n_frames_per_snr} iters={cfg.max_iter}")
    print(f"# snr_list = {cfg.snr_list}")

    results = {}
    for label, clamp in [("tight=10", 10.0), ("loose=1e6", 1e6)]:
        m, dt = run_one(clamp, cfg)
        results[label] = (m, dt)
        ber_str = "  ".join(f"{b:.3e}" for b in m.ber_per_snr)
        print(f"\n{label:>10s}  fitness={m.fitness:+.4f}  valid={m.valid}"
              f"  t={dt:.1f}s")
        print(f"{'':12s}BER  : {ber_str}")
        if not m.valid:
            print(f"{'':12s}error: {m.error}")

    if all(r[0].valid for r in results.values()):
        a = np.asarray(results["tight=10"][0].ber_per_snr)
        b = np.asarray(results["loose=1e6"][0].ber_per_snr)
        rel = (a - b) / np.where(b > 0, b, 1.0)
        print("\n# BER(tight) - BER(loose) (per SNR)")
        for snr, da, db, r in zip(cfg.snr_list, a, b, rel):
            print(f"   snr={snr:4.1f}  tight={da:.3e}  loose={db:.3e}  rel={r:+.2%}")
        max_abs = float(np.max(np.abs(a - b)))
        print(f"\n# max |\u0394BER| across SNR = {max_abs:.3e}")
        # Verdict
        if np.all(np.abs(rel) < 0.05) and max_abs < 1e-3:
            print("# Verdict: clamp=10 is INDISTINGUISHABLE from unclamped.")
        elif np.all(np.abs(rel) < 0.2):
            print("# Verdict: clamp=10 produces minor BER drift (<20%).")
        else:
            print("# Verdict: clamp=10 SIGNIFICANTLY changes OMS BER.")


if __name__ == "__main__":
    main()
