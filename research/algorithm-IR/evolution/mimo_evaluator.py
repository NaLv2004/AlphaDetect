"""MIMO fitness evaluator for the two-level algorithm evolution engine.

Evaluates candidate detector algorithms by:
  1. Materializing the genome (replacing AlgSlot ops with slot variants)
  2. Running Monte Carlo trials over random MIMO channels
  3. Computing BER + complexity metrics
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from evolution.fitness import FitnessResult
from evolution.pool_types import AlgorithmFitnessEvaluator, AlgorithmGenome
from evolution.materialize import materialize_to_callable


# ═══════════════════════════════════════════════════════════════════════════
# Constellation helpers
# ═══════════════════════════════════════════════════════════════════════════

def qam16_constellation() -> np.ndarray:
    """Normalized 16-QAM constellation."""
    pts = []
    for a in (-3, -1, 1, 3):
        for b in (-3, -1, 1, 3):
            pts.append(complex(a, b))
    c = np.array(pts, dtype=complex)
    c /= np.sqrt(np.mean(np.abs(c) ** 2))
    return c


def qpsk_constellation() -> np.ndarray:
    """Normalized QPSK constellation."""
    c = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=complex)
    c /= np.sqrt(np.mean(np.abs(c) ** 2))
    return c


# ═══════════════════════════════════════════════════════════════════════════
# MIMO sample generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_mimo_sample(
    Nr: int,
    Nt: int,
    constellation: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate one MIMO sample: (H, x_true, y, sigma2).

    Parameters
    ----------
    Nr, Nt : int
        Antenna dimensions.
    constellation : ndarray
        Modulation symbol set.
    snr_db : float
        SNR in dB.
    rng : Generator
        NumPy random generator.

    Returns
    -------
    H : (Nr, Nt) complex
    x_true : (Nt,) complex — transmitted symbols
    y : (Nr,) complex — received signal
    sigma2 : float — noise variance
    """
    H = (rng.standard_normal((Nr, Nt)) + 1j * rng.standard_normal((Nr, Nt))) / np.sqrt(2.0)
    x_idx = rng.integers(0, len(constellation), size=Nt)
    x_true = constellation[x_idx]
    sig_power = float(np.mean(np.abs(H @ x_true) ** 2))
    sigma2 = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(sigma2 / 2.0) * (rng.standard_normal(Nr) + 1j * rng.standard_normal(Nr))
    y = H @ x_true + noise
    return H, x_true, y, sigma2


def symbol_error_rate(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    """Symbol error rate between two symbol vectors."""
    return float(np.mean(np.abs(x_true - x_hat) > 1e-6))


# ═══════════════════════════════════════════════════════════════════════════
# Evaluator configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MIMOEvalConfig:
    """Configuration for MIMO evaluation."""
    Nr: int = 16
    Nt: int = 16
    mod_order: int = 16            # 4=QPSK, 16=16QAM
    snr_db_list: list[float] = field(default_factory=lambda: [10.0, 15.0, 20.0])
    n_trials: int = 200            # Monte Carlo trials per SNR
    timeout_sec: float = 5.0       # max seconds per genome
    complexity_weight: float = 0.1 # weight for complexity in composite score
    batch_workers: int = 1         # outer parallelism across genomes
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════════
# MIMO Fitness Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class MIMOFitnessEvaluator(AlgorithmFitnessEvaluator):
    """Concrete fitness evaluator: BER over MIMO Rayleigh fading channels.

    Fitness = average SER across SNR points + complexity penalty.
    Lower is better.
    """

    def __init__(self, config: MIMOEvalConfig | None = None):
        self.config = config or MIMOEvalConfig()
        if self.config.mod_order == 16:
            self.constellation = qam16_constellation()
        else:
            self.constellation = qpsk_constellation()

    def evaluate(self, genome: AlgorithmGenome) -> FitnessResult:
        """Evaluate a genome by materializing and running Monte Carlo."""
        cfg = self.config

        # 1. Materialize
        try:
            fn = materialize_to_callable(genome)
        except Exception as exc:
            return FitnessResult(
                metrics={"ser": 1.0, "compile_error": 1.0},
                is_valid=False,
            )

        # 2. Monte Carlo evaluation
        rng = np.random.default_rng(cfg.seed)
        ser_per_snr: dict[float, float] = {}
        total_time = 0.0

        # Per-trial timeout using a thread executor (works on Windows).
        # NOTE: We create a new single-use executor per trial because
        # `executor.shutdown(wait=True)` would block if a thread is stuck.
        import concurrent.futures
        _trial_timeout = max(1.0, cfg.timeout_sec / max(cfg.n_trials, 1) * 3)

        for snr_db in cfg.snr_db_list:
            errors = 0
            total = 0
            t0 = time.perf_counter()

            # Reuse a single executor; only recreate after a timeout to avoid
            # stale threads blocking subsequent trials.
            _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            _ex_dirty = False

            for trial_idx in range(cfg.n_trials):
                H, x_true, y, sigma2 = generate_mimo_sample(
                    cfg.Nr, cfg.Nt, self.constellation, snr_db, rng,
                )
                if _ex_dirty:
                    _ex.shutdown(wait=False)
                    _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    _ex_dirty = False
                try:
                    future = _ex.submit(fn, H, y, sigma2, self.constellation)
                    x_hat = future.result(timeout=_trial_timeout)
                    if x_hat is None or len(x_hat) != cfg.Nt:
                        errors += cfg.Nt
                    else:
                        x_hat = _nearest_symbols(x_hat, self.constellation)
                        errors += int(np.sum(np.abs(x_true - x_hat) > 1e-6))
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    errors += cfg.Nt
                    _ex_dirty = True  # Thread may be stuck; recreate executor
                except Exception:
                    errors += cfg.Nt

                total += cfg.Nt
                elapsed = time.perf_counter() - t0
                if elapsed > cfg.timeout_sec:
                    remaining = cfg.n_trials - (trial_idx + 1)
                    errors += remaining * cfg.Nt
                    total += remaining * cfg.Nt
                    break

            _ex.shutdown(wait=False)
            total_time += time.perf_counter() - t0
            ser_per_snr[snr_db] = errors / max(total, 1)

        # 3. Metrics
        avg_ser = float(np.mean(list(ser_per_snr.values())))
        complexity = _estimate_complexity(genome)

        metrics = {
            "ser": avg_ser,
            "complexity": complexity,
            "eval_time": total_time,
        }
        for snr, ser in ser_per_snr.items():
            metrics[f"ser_{snr:.0f}dB"] = ser

        # Only ser and complexity contribute to fitness; per-snr and time are informational
        weights = {
            "ser": 1.0,
            "complexity": cfg.complexity_weight,
            "eval_time": 0.0,
        }
        for snr in ser_per_snr:
            weights[f"ser_{snr:.0f}dB"] = 0.0

        return FitnessResult(metrics=metrics, is_valid=True, weights=weights)

    def evaluate_batch(self, genomes: list[AlgorithmGenome]) -> list[FitnessResult]:
        """Batch-evaluate genomes with optional outer thread parallelism."""
        if not genomes:
            return []
        workers = max(1, int(getattr(self.config, "batch_workers", 1)))
        if workers <= 1 or len(genomes) <= 1:
            return [self.evaluate(g) for g in genomes]

        results: list[FitnessResult | None] = [None] * len(genomes)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_idx = {
                ex.submit(self.evaluate, genome): idx
                for idx, genome in enumerate(genomes)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = FitnessResult(
                        metrics={"ser": 1.0, "batch_error": 1.0},
                        is_valid=False,
                    )
        return [
            r if r is not None else FitnessResult(metrics={"ser": 1.0}, is_valid=False)
            for r in results
        ]

    def evaluate_single_result(self, result: Any) -> float:
        """Quick scalar fitness from a single trial result.

        Expects result = (x_true, x_hat) tuple.
        """
        if result is None:
            return 1.0
        x_true, x_hat = result
        return symbol_error_rate(x_true, x_hat)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _nearest_symbols(
    x_soft: np.ndarray,
    constellation: np.ndarray,
) -> np.ndarray:
    """Map soft estimates to nearest constellation symbols."""
    x_hat = np.empty(len(x_soft), dtype=complex)
    for i in range(len(x_soft)):
        dists = np.abs(constellation - x_soft[i]) ** 2
        x_hat[i] = constellation[np.argmin(dists)]
    return x_hat


def _estimate_complexity(genome: AlgorithmGenome) -> float:
    """Rough complexity proxy: total number of ops in skeleton + slot variants.

    Returns a normalized score in [0, 1].
    """
    total_ops = len(genome.structural_ir.ops) if genome.structural_ir else 0
    for pop in genome.slot_populations.values():
        if pop.variants and pop.best_idx < len(pop.variants):
            best = pop.variants[pop.best_idx]
            if best is not None:
                total_ops += len(best.ops)
    # Normalize: typical detector ~50-200 ops
    return min(total_ops / 500.0, 1.0)
