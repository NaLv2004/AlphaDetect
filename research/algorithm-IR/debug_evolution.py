"""Debug evolution stagnation issues."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

import numpy as np
from evolution.ir_pool import build_ir_pool
from evolution.materialize import materialize_to_callable
from evolution.mimo_evaluator import (
    MIMOEvalConfig, MIMOFitnessEvaluator, qam16_constellation, generate_mimo_sample
)
from evolution.pool_types import AlgorithmEvolutionConfig
from evolution.algorithm_engine import AlgorithmEvolutionEngine

eval_cfg = MIMOEvalConfig(
    Nr=16, Nt=16, mod_order=16,
    snr_db_list=[18.0, 22.0, 24.0],
    n_trials=10,  # quick
    timeout_sec=10.0,
    complexity_weight=0.01,
    seed=42,
)
evaluator = MIMOFitnessEvaluator(eval_cfg)

pool = build_ir_pool()
print("=== Evaluating all baseline genomes ===")
for g in pool:
    res = evaluator.evaluate(g)
    print(f"  {g.algo_id:15s} (tags={g.tags}): score={res.composite_score():.4f}, SER@24dB={res.metrics.get('ser_24dB', 'N/A'):.4f}, valid={res.is_valid}")

# Check hall of fame type
from evolution.algorithm_engine import AlgorithmEvolutionEngine
evo_cfg = AlgorithmEvolutionConfig(pool_size=8, n_generations=1, micro_generations=1, micro_pop_size=2, micro_mutation_rate=0.3, seed=42)
engine = AlgorithmEvolutionEngine(evaluator=evaluator, config=evo_cfg, rng=np.random.default_rng(42))
engine.init_population()
print("\n=== Population after init ===")
for g, f in zip(engine.population, engine.fitness):
    print(f"  {g.algo_id}: score={f.composite_score():.4f}")
print(f"Best: {engine.best_genome.algo_id if engine.best_genome else 'None'}")
print(f"Hall of fame type: {type(engine.hall_of_fame[0]) if engine.hall_of_fame else 'empty'}")
