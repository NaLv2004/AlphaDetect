"""Quick diagnostic: check if micro-evolution or grafting is hanging."""
import sys, os, time
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from evolution.pool_types import AlgorithmEvolutionConfig
from evolution.algorithm_engine import AlgorithmEvolutionEngine
from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
from evolution.pattern_matchers import (
    CompositePatternMatcher, ExpertPatternMatcher, StaticStructurePatternMatcher,
)

eval_config = MIMOEvalConfig(
    Nr=16, Nt=16, mod_order=16,
    snr_db_list=[24.0],
    n_trials=10,  # Very few trials for speed
    timeout_sec=2.0,
    complexity_weight=0.01,
    seed=42,
)
evaluator = MIMOFitnessEvaluator(eval_config)

pattern_matcher = CompositePatternMatcher([
    ExpertPatternMatcher(max_proposals_per_gen=3),
    StaticStructurePatternMatcher(max_proposals_per_gen=3),
])

evo_config = AlgorithmEvolutionConfig(
    pool_size=12,  # Small pool for speed
    n_generations=1,
    seed=42,
)

engine = AlgorithmEvolutionEngine(
    evaluator=evaluator,
    config=evo_config,
    rng=np.random.default_rng(42),
    pattern_matcher=pattern_matcher,
)

print("=== INIT ===")
t0 = time.perf_counter()
engine.init_population()
print(f"Init: {time.perf_counter()-t0:.1f}s, {len(engine.population)} genomes")

print("\n=== MICRO-EVOLVE (1 genome) ===")
t0 = time.perf_counter()
engine._micro_evolve(engine.population[0])
print(f"Micro-evolve: {time.perf_counter()-t0:.1f}s")

print("\n=== EVALUATE 1 GENOME ===")
t0 = time.perf_counter()
f = evaluator.evaluate(engine.population[0])
print(f"Evaluate: {time.perf_counter()-t0:.1f}s, score={f.composite_score():.6f}")

print("\n=== PATTERN MATCH ===")
entries = [g.to_entry(f) for g, f in zip(engine.population, engine.fitness)]
t0 = time.perf_counter()
proposals = pattern_matcher(entries, 1)
print(f"Pattern match: {time.perf_counter()-t0:.1f}s, {len(proposals)} proposals")
for p in proposals:
    print(f"  {p.proposal_id}: host={p.host_algo_id[:15]}, donor={p.donor_algo_id[:15] if p.donor_algo_id else '?'}, "
          f"region={len(p.region.op_ids)} ops, conf={p.confidence}")

print("\n=== GRAFT EACH PROPOSAL ===")
for i, p in enumerate(proposals[:3]):
    t0 = time.perf_counter()
    try:
        child = engine._execute_graft(p)
        elapsed = time.perf_counter() - t0
        if child:
            n_ops = len(child.structural_ir.ops)
            print(f"  Graft {i}: OK in {elapsed:.1f}s, {n_ops} ops")
            # Quick evaluate
            t1 = time.perf_counter()
            f = evaluator.evaluate(child)
            print(f"  Evaluate: {time.perf_counter()-t1:.1f}s, score={f.composite_score():.6f}")
        else:
            print(f"  Graft {i}: returned None in {elapsed:.1f}s")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  Graft {i}: FAILED in {elapsed:.1f}s: {exc}")

print("\n=== DONE ===")
