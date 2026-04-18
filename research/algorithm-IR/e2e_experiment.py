"""End-to-end evolution experiment: 16×16 16QAM MIMO detection.

Target: discover a detector algorithm with BER < 1E-3 at SNR = 24 dB.

Strategy:
  1. Start from 8-detector pool (LMMSE, ZF, OSIC, K-Best, Stack, BP, EP, AMP)
  2. Use ExpertPatternMatcher for structural grafting
  3. Run dual-track evolution:
     - Micro-level: slot variant mutation within each genome
     - Macro-level: crossover + grafting across genomes
  4. Evaluate the best genome at 24 dB with high Monte Carlo count
"""
import sys
import pathlib
import time
import logging

ROOT = pathlib.Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from evolution.ir_pool import build_ir_pool
from evolution.materialize import materialize_to_callable
from evolution.mimo_evaluator import (
    MIMOEvalConfig,
    MIMOFitnessEvaluator,
    qam16_constellation,
    generate_mimo_sample,
    _nearest_symbols,
)
from evolution.pool_types import AlgorithmEvolutionConfig
from evolution.algorithm_engine import AlgorithmEvolutionEngine
from evolution.pattern_matchers import (
    ExpertPatternMatcher,
    RandomGraftPatternMatcher,
    CompositePatternMatcher,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_detector(fn, Nr, Nt, snr_db, constellation, n_trials, rng):
    """Evaluate a detector function at a single SNR point."""
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
    return errors, total


def main():
    Nr, Nt = 16, 16
    target_snr = 24.0
    target_ber = 1e-3

    print("=" * 70)
    print(f"E2E Evolution: {Nr}×{Nt} 16QAM, target BER < {target_ber} @ {target_snr} dB")
    print("=" * 70)

    # ----- Configuration -----
    eval_cfg = MIMOEvalConfig(
        Nr=Nr,
        Nt=Nt,
        mod_order=16,
        snr_db_list=[18.0, 22.0, 24.0],
        n_trials=100,          # trials per SNR during evolution
        timeout_sec=10.0,
        complexity_weight=0.01,
        seed=42,
    )
    evo_cfg = AlgorithmEvolutionConfig(
        pool_size=12,
        n_generations=30,
        micro_generations=5,
        micro_pop_size=8,
        micro_mutation_rate=0.8,
        seed=42,
    )

    evaluator = MIMOFitnessEvaluator(eval_cfg)

    # ----- Pattern Matchers -----
    pattern_matcher = CompositePatternMatcher([
        ExpertPatternMatcher(max_proposals_per_gen=2),
        RandomGraftPatternMatcher(proposals_per_gen=1, seed=42),
    ])

    # ----- Engine -----
    engine = AlgorithmEvolutionEngine(
        evaluator=evaluator,
        config=evo_cfg,
        rng=np.random.default_rng(42),
        pattern_matcher=pattern_matcher,
    )

    # ----- Run -----
    t0 = time.perf_counter()

    def progress_callback(gen, best_fitness, population):
        score = best_fitness.composite_score() if best_fitness else float("inf")
        ser_24 = best_fitness.metrics.get("ser_24dB", "N/A") if best_fitness else "N/A"
        logger.info(f"  Gen {gen}: score={score:.6f}, SER@24dB={ser_24}")

    best_genome = engine.run(callback=progress_callback)
    elapsed = time.perf_counter() - t0

    print(f"\nEvolution completed in {elapsed:.1f}s")
    print(f"Best genome: {best_genome.algo_id}")
    print(f"Best fitness: {engine.best_fitness}")
    print(f"Tags: {best_genome.tags}")
    print(f"Generation: {best_genome.generation}")

    # ----- Final evaluation -----
    print("\n" + "=" * 70)
    print("Final evaluation with 5000 trials")
    print("=" * 70)

    constellation = qam16_constellation()
    rng = np.random.default_rng(123)

    try:
        fn = materialize_to_callable(best_genome)
    except Exception as e:
        print(f"MATERIALIZE FAILED: {e}")
        return

    for snr_db in [18, 20, 22, 24, 26, 28]:
        errs, tot = evaluate_detector(fn, Nr, Nt, snr_db, constellation, 5000, rng)
        ser = errs / max(tot, 1)
        ber = ser / 4
        status = "✓ PASS" if ber < target_ber else ""
        print(f"  SNR={snr_db:4.0f} dB: SER={ser:.6f}  BER≈{ber:.6f}  {status}")

    # ----- Also evaluate OSIC baseline for comparison -----
    print("\n--- OSIC baseline ---")
    pool = build_ir_pool()
    rng_base = np.random.default_rng(123)
    for g in pool:
        if 'osic' in g.structural_ir.name.lower():
            fn_osic = materialize_to_callable(g)
            errs, tot = evaluate_detector(fn_osic, Nr, Nt, 24, constellation, 5000, rng_base)
            ser = errs / max(tot, 1)
            ber = ser / 4
            print(f"  OSIC @ 24 dB: SER={ser:.6f}  BER≈{ber:.6f}")
            break

    # ----- Hall of fame -----
    print("\n--- Hall of Fame ---")
    for entry in engine.hall_of_fame[:5]:
        print(f"  {entry.algo_id} (gen={entry.generation}, tags={entry.tags})")


if __name__ == "__main__":
    main()
