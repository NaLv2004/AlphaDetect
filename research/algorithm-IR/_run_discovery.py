"""
Discovery run: 20-gen evolution with focus on grafting-based algorithm discovery.
Tracks which grafted algorithms are structurally novel and competitive.
"""
import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
import numpy as np
from evolution.pool_types import AlgorithmEvolutionConfig
from evolution.algorithm_engine import AlgorithmEvolutionEngine
from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
from evolution.pattern_matchers import (
    CompositePatternMatcher, ExpertPatternMatcher, StaticStructurePatternMatcher,
)
from evolution.gnn_pattern_matcher import GNNPatternMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

eval_config = MIMOEvalConfig(
    Nr=16, Nt=16, mod_order=16,
    snr_db_list=[24.0],
    n_trials=100,           # More trials for stable BER estimates
    timeout_sec=5.0,
    complexity_weight=0.01,
    seed=42,
)
evaluator = MIMOFitnessEvaluator(eval_config)

gnn_matcher = GNNPatternMatcher(max_proposals_per_gen=4, top_k_pairs=8)

pattern_matcher = CompositePatternMatcher([
    ExpertPatternMatcher(max_proposals_per_gen=3),
    StaticStructurePatternMatcher(max_proposals_per_gen=3),
    gnn_matcher,
])

evo_config = AlgorithmEvolutionConfig(
    pool_size=48,
    n_generations=20,
    seed=42,
)

engine = AlgorithmEvolutionEngine(
    evaluator=evaluator,
    config=evo_config,
    rng=np.random.default_rng(42),
    pattern_matcher=pattern_matcher,
)

# Track discoveries
discovered_grafts: list[dict] = []

def _cb(gen: int, best_f, pop):
    score = best_f.composite_score() if best_f else float("inf")
    algos = set(g.algo_id for g in pop)
    grafted = [g for g in pop if "grafted" in g.tags]
    n_grafted = len(grafted)

    # Find best grafted
    best_graft_score = float("inf")
    best_graft_id = None
    for g in grafted:
        idx = pop.index(g)
        gs = engine.fitness[idx].composite_score() if idx < len(engine.fitness) else float("inf")
        if gs < best_graft_score:
            best_graft_score = gs
            best_graft_id = g.algo_id

    # Check if any grafted algorithm is competitive
    if best_graft_id and best_graft_score < float("inf"):
        # Is it a NEW discovery?
        known_ids = {d["algo_id"] for d in discovered_grafts}
        if best_graft_id not in known_ids and best_graft_score < 0.1:
            discovered_grafts.append({
                "gen": gen,
                "algo_id": best_graft_id,
                "score": best_graft_score,
                "parent": getattr(pop[pop.index(next(g for g in pop if g.algo_id == best_graft_id))], 'parent_algo_id', '?'),
            })

    gnn_stats = gnn_matcher.get_stats()
    print(f"  Gen {gen:2d}: best={score:.6f}, grafted={n_grafted:2d}/48, "
          f"best_graft={best_graft_score:.6f}, gnn_proposals={gnn_stats['total_proposals']}")

    if gen % 5 == 0:
        print(f"         Unique algos: {len(algos)}")
        print(f"         Discoveries: {len(discovered_grafts)} novel grafted algorithms")
        for d in discovered_grafts[-3:]:
            print(f"           - Gen {d['gen']}: {d['algo_id']} score={d['score']:.6f}")

print("=" * 70)
print("DISCOVERY RUN: 20 gen, pool=48, 100 trials, 16x16 MIMO 16QAM SNR=24")
print("=" * 70)

best = engine.run(callback=_cb)

print("\n" + "=" * 70)
print(f"FINAL BEST: {best.algo_id}, score={engine.best_fitness.composite_score():.6f}")
print(f"Grafted in population: {sum(1 for g in engine.population if 'grafted' in g.tags)}")
print(f"Total discoveries: {len(discovered_grafts)} novel grafted algorithms")
for d in discovered_grafts:
    print(f"  Gen {d['gen']:2d}: {d['algo_id']} score={d['score']:.6f}")
print(f"GNN stats: {gnn_matcher.get_stats()}")
print("=" * 70)
