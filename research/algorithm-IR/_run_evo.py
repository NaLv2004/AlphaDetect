"""Quick runner for evolution experiment with BER-focused fitness."""
import sys
import os
# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import time
import json
import logging
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from evolution.pool_types import AlgorithmEvolutionConfig
from evolution.algorithm_engine import AlgorithmEvolutionEngine
from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
from evolution.materialize import materialize
from evolution.pattern_matchers import CompositePatternMatcher, ExpertPatternMatcher, StaticStructurePatternMatcher
from evolution.gnn_pattern_matcher import GNNPatternMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# BER-focused: complexity_weight=0.01 (almost zero)
eval_config = MIMOEvalConfig(
    Nr=16, Nt=16, mod_order=16,
    snr_db_list=[24.0],
    n_trials=200,
    timeout_sec=5.0,         # default timeout
    complexity_weight=0.01,  # focus on SER, not complexity
    seed=42,
)
evaluator = MIMOFitnessEvaluator(eval_config)

# Pattern matcher for structural grafting (Expert + Static + GNN)
pattern_matcher = CompositePatternMatcher([
    ExpertPatternMatcher(max_proposals_per_gen=2),
    StaticStructurePatternMatcher(max_proposals_per_gen=2),
    GNNPatternMatcher(max_proposals_per_gen=4, top_k_pairs=8),
])

evo_config = AlgorithmEvolutionConfig(
    pool_size=48,
    n_generations=50,
    seed=42,
)

engine = AlgorithmEvolutionEngine(
    evaluator=evaluator,
    config=evo_config,
    rng=np.random.default_rng(42),
    pattern_matcher=pattern_matcher,
)

def _cb(gen, best_f, pop):
    score = best_f.composite_score() if best_f else float("inf")
    ser = best_f.metrics.get("ser", float("inf")) if best_f else float("inf")
    logger.info("  Gen %3d | score=%.6f | SER=%.6f", gen, score, ser)

t0 = time.perf_counter()
best = engine.run(callback=_cb)
total_time = time.perf_counter() - t0

print("\n" + "=" * 60)
print("EVOLUTION COMPLETE")
print(f"Total time: {total_time:.1f}s")
if best and engine.best_fitness:
    metrics = engine.best_fitness.metrics
    print(f"Best algo: {best.algo_id}")
    print(f"Best score: {engine.best_fitness.composite_score():.6f}")
    print(f"SER at 24dB: {metrics.get('ser_24dB', metrics.get('ser', '?'))}")
    print(f"All metrics: {metrics}")
    try:
        src = materialize(best)
        print(f"\nBest source ({len(src)} chars):\n{src[:800]}")
        out = Path("results/evolution_16x16_16qam_24dB")
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "best_detector.py", "w") as f:
            f.write(src)
        with open(out / "evolution_result.json", "w") as f:
            json.dump({"best_algo": best.algo_id, "metrics": metrics,
                       "score": engine.best_fitness.composite_score(),
                       "time": total_time}, f, indent=2, default=str)
    except Exception as e:
        print(f"Materialize failed: {e}")
print("=" * 60)
