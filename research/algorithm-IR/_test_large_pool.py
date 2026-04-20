"""Quick 3-gen evolution test with large pool."""
import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
import numpy as np
from evolution.pool_types import AlgorithmEvolutionConfig
from evolution.algorithm_engine import AlgorithmEvolutionEngine
from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
from evolution.pattern_matchers import CompositePatternMatcher, ExpertPatternMatcher, StaticStructurePatternMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s — %(message)s", datefmt="%H:%M:%S")

eval_config = MIMOEvalConfig(
    Nr=16, Nt=16, mod_order=16,
    snr_db_list=[24.0],
    n_trials=50,  # fewer trials for quick test
    timeout_sec=3.0,
    complexity_weight=0.01,
    seed=42,
)
evaluator = MIMOFitnessEvaluator(eval_config)

pattern_matcher = CompositePatternMatcher([
    ExpertPatternMatcher(max_proposals_per_gen=2),
    StaticStructurePatternMatcher(max_proposals_per_gen=2),
])

evo_config = AlgorithmEvolutionConfig(
    pool_size=48,
    n_generations=3,
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
    algos = set(g.algo_id for g in pop)
    print(f"  Gen {gen}: score={score:.6f}, unique_algos={len(algos)}, pop={len(pop)}")

best = engine.run(callback=_cb)
print(f"\nBest: {best.algo_id}, score={engine.best_fitness.composite_score():.6f}")
print(f"Population algo diversity: {set(g.algo_id for g in engine.population)}")
