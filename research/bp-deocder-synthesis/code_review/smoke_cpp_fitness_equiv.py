"""Smoke test: C++ fitness eval must be numerically equivalent to Python.

Runs N random (V2C, C2V, K) triples through both:
  - the Python loop in `pushgp_ldpc.eval_logged.evaluate_genome_with_ber`
    with `use_cpp_fitness=False`
  - the C++ wrapper `pushgp_ldpc.eval_cpp.evaluate_genome_cpp_ber`
    via `use_cpp_fitness=True`

Asserts:
  1. Per-SNR BER differs by at most BER_TOL (default 1e-8).
  2. Per-SNR FER differs by at most FER_TOL (default 1e-8).
  3. Scalar fitness differs by at most FIT_TOL (default 1e-6).
  4. Spearman rank correlation of the fitness scalars across the N
     genomes is exactly 1.0 (the GA selection signal is preserved).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ldpc_5g import build_parity  # noqa: E402
from pushgp.genome import Genome  # noqa: E402
from pushgp_ldpc.adapter import make_callables  # noqa: E402
from pushgp_ldpc.eval import FitnessConfig  # noqa: E402
from pushgp_ldpc.eval_logged import evaluate_genome_with_ber  # noqa: E402


def _spearman_eq(a, b) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.array_equal(np.argsort(a), np.argsort(b))


def main(n_genomes: int = 8,
         snr_list=(0.0, 1.0, 2.0),
         n_frames: int = 4,
         max_iter: int = 8,
         seed: int = 4242,
         ber_tol: float = 1e-8,
         fer_tol: float = 1e-8,
         fit_tol: float = 1e-6) -> int:

    # ---- build cfg ----
    par = build_parity(bgn=2, set_idx=1, zc=2)
    cfg_py = FitnessConfig(
        par=par,
        snr_list=tuple(snr_list),
        n_frames_per_snr=n_frames,
        max_iter=max_iter,
        code_rate=0.5,
        seed_base=12345,
        use_cpp_fitness=False,
    )
    cfg_cpp = replace(cfg_py, use_cpp_fitness=True)

    # ---- generate random genomes via cpp_seeder ----
    from pushgp.random_program import RandomProgramGenerator
    rpg = RandomProgramGenerator(rng=np.random.default_rng(seed))

    genomes = []
    attempts = 0
    while len(genomes) < n_genomes and attempts < 200:
        attempts += 1
        try:
            pv = rpg.random_v2c(min_size=4, max_size=20)
            pc = rpg.random_c2v(min_size=4, max_size=20)
            lc = (rpg.random_log_constants()
                  if hasattr(rpg, "random_log_constants")
                  else np.zeros(8, dtype=np.float64))
            g = Genome(prog_v2c=pv, prog_c2v=pc, log_constants=lc.copy())
            # Must be adapter-runnable so the Python reference path works.
            make_callables(g)
            genomes.append(g)
        except Exception:
            continue
    if len(genomes) < n_genomes:
        print(f"[FAIL] only got {len(genomes)} / {n_genomes} valid genomes "
              f"in {attempts} attempts")
        return 2

    print(f"[smoke] sampled {len(genomes)} valid genomes  "
          f"snr={list(snr_list)}  n_frames={n_frames}  max_iter={max_iter}")

    fits_py, fits_cpp = [], []
    n_bad = 0
    t_py_total = 0.0
    t_cpp_total = 0.0
    for gi, g in enumerate(genomes):
        t0 = time.perf_counter()
        m_py = evaluate_genome_with_ber(g, cfg_py)
        t_py_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        m_cpp = evaluate_genome_with_ber(g, cfg_cpp)
        t_cpp_total += time.perf_counter() - t0

        # Skip pairs where both paths short-circuit on adapter failure.
        if not m_py.valid and not m_cpp.valid:
            print(f"[g{gi}] both invalid -- skipping")
            continue
        if m_py.valid != m_cpp.valid:
            print(f"[g{gi}] FAIL: valid mismatch py={m_py.valid} cpp={m_cpp.valid}")
            n_bad += 1
            continue

        d_fit = abs(m_py.fitness - m_cpp.fitness)
        if d_fit > fit_tol:
            print(f"[g{gi}] FAIL: |Δfitness|={d_fit:.3e} > {fit_tol}")
            n_bad += 1

        for s, (b_py, b_cpp) in enumerate(zip(m_py.ber_per_snr, m_cpp.ber_per_snr)):
            d = abs(b_py - b_cpp)
            if d > ber_tol:
                print(f"[g{gi}] FAIL: BER@snr#{s}  py={b_py:.6e} cpp={b_cpp:.6e}  "
                      f"Δ={d:.3e}")
                n_bad += 1
        for s, (f_py, f_cpp) in enumerate(zip(m_py.fer_per_snr, m_cpp.fer_per_snr)):
            d = abs(f_py - f_cpp)
            if d > fer_tol:
                print(f"[g{gi}] FAIL: FER@snr#{s}  py={f_py:.6e} cpp={f_cpp:.6e}  "
                      f"Δ={d:.3e}")
                n_bad += 1

        fits_py.append(m_py.fitness)
        fits_cpp.append(m_cpp.fitness)
        print(f"[g{gi}] fit_py={m_py.fitness:+.6f}  fit_cpp={m_cpp.fitness:+.6f}  "
              f"|Δ|={d_fit:.2e}")

    if n_bad > 0:
        print(f"\n[SMOKE FAILED] {n_bad} mismatches.")
        return 1

    # Selection-signal preservation (Spearman ρ = 1).
    if len(fits_py) >= 2:
        if not _spearman_eq(fits_py, fits_cpp):
            print(f"\n[SMOKE FAILED] Spearman rank mismatch:")
            print(f"  py  argsort = {np.argsort(fits_py)}")
            print(f"  cpp argsort = {np.argsort(fits_cpp)}")
            return 1
        print(f"\n[smoke] Spearman rank: IDENTICAL")

    speedup = t_py_total / t_cpp_total if t_cpp_total > 0 else float("inf")
    print(f"[smoke] timing  python={t_py_total:.2f}s  cpp={t_cpp_total:.2f}s  "
          f"speedup={speedup:.1f}x")
    print("SMOKE PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
