"""End-to-end runner for the multi-level MIMO detector evolution.

Usage (from research/algorithm-IR/)::

    conda run -n AutoGenOld python -m evolution.run_evolution --generations 50

Or in Python::

    from evolution.run_evolution import run
    best = run(generations=20, nr=8, nt=8)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from evolution.pool_types import AlgorithmEvolutionConfig, PatternMatcherFn
from evolution.algorithm_engine import AlgorithmEvolutionEngine
from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
from evolution.materialize import materialize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(
    *,
    generations: int = 50,
    pool_size: int = 16,
    nr: int = 16,
    nt: int = 16,
    mod_order: int = 16,
    snr_list: list[float] | None = None,
    n_trials: int = 100,
    seed: int = 42,
    output_dir: str | None = None,
    pattern_matcher: PatternMatcherFn | None = None,
) -> dict:
    """Run the full multi-level evolution experiment.

    Parameters
    ----------
    generations : int
        Number of macro generations.
    pool_size : int
        Population size.
    nr, nt : int
        Antenna dimensions.
    mod_order : int
        Modulation order (4=QPSK, 16=16QAM).
    snr_list : list[float]
        SNR evaluation points in dB.
    n_trials : int
        Monte Carlo trials per SNR point.
    seed : int
        Random seed.
    output_dir : str, optional
        Directory to save results.

    Returns
    -------
    dict
        Summary with best genome info, history, and source code.
    """
    if snr_list is None:
        snr_list = [10.0, 15.0, 20.0]

    logger.info("=" * 60)
    logger.info("Multi-Level MIMO Detector Evolution")
    logger.info("  %dx%d, %d-QAM, SNR=%s dB", nr, nt, mod_order, snr_list)
    logger.info("  pool=%d, generations=%d, trials=%d, seed=%d",
                pool_size, generations, n_trials, seed)
    logger.info("=" * 60)

    # Configure evaluator
    eval_config = MIMOEvalConfig(
        Nr=nr,
        Nt=nt,
        mod_order=mod_order,
        snr_db_list=snr_list,
        n_trials=n_trials,
        seed=seed,
    )
    evaluator = MIMOFitnessEvaluator(eval_config)

    # Configure evolution
    evo_config = AlgorithmEvolutionConfig(
        pool_size=pool_size,
        n_generations=generations,
        seed=seed,
    )

    # Create engine
    engine = AlgorithmEvolutionEngine(
        evaluator=evaluator,
        config=evo_config,
        rng=np.random.default_rng(seed),
        pattern_matcher=pattern_matcher,
    )

    # Run
    t0 = time.perf_counter()
    best = engine.run(callback=_progress_callback)
    total_time = time.perf_counter() - t0

    logger.info("=" * 60)
    logger.info("Evolution complete in %.1f seconds", total_time)
    logger.info("Best: %s (score=%.6f)",
                best.algo_id if best else "None",
                engine.best_fitness.composite_score() if engine.best_fitness else float("inf"))

    # Materialize best genome
    best_source = None
    if best is not None:
        try:
            best_source = materialize(best)
            logger.info("Materialized source:\n%s", best_source)
        except Exception as exc:
            logger.warning("Failed to materialize best: %s", exc)

    # Build result summary
    result = {
        "best_algo_id": best.algo_id if best else None,
        "best_score": engine.best_fitness.composite_score() if engine.best_fitness else None,
        "best_metrics": engine.best_fitness.metrics if engine.best_fitness else {},
        "best_source": best_source,
        "total_time_sec": total_time,
        "generations": engine.generation,
        "history": engine.history,
        "hall_of_fame": [
            {"algo_id": g.algo_id, "score": f.composite_score()}
            for g, f in engine.hall_of_fame
        ],
    }

    # Save results
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "evolution_result.json", "w") as fp:
            json.dump(
                {k: v for k, v in result.items() if k != "best_source"},
                fp, indent=2, default=str,
            )
        if best_source:
            with open(out / "best_detector.py", "w") as fp:
                fp.write(best_source)
        logger.info("Results saved to %s", out)

    return result


def _progress_callback(gen: int, best_fitness, population) -> None:
    """Logging callback."""
    score = best_fitness.composite_score() if best_fitness else float("inf")
    algos = {}
    for g in population:
        algos[g.algo_id] = algos.get(g.algo_id, 0) + 1
    logger.info("  Gen %3d | best=%.6f | pop=%s", gen, score, dict(algos))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-level MIMO detector evolution"
    )
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--nr", type=int, default=16)
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--mod-order", type=int, default=16)
    parser.add_argument("--snr", type=float, nargs="+", default=[10.0, 15.0, 20.0])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/evolution")

    args = parser.parse_args()
    run(
        generations=args.generations,
        pool_size=args.pool_size,
        nr=args.nr,
        nt=args.nt,
        mod_order=args.mod_order,
        snr_list=args.snr,
        n_trials=args.trials,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
