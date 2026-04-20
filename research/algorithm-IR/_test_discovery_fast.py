"""Fast discovery test: 10 gen, 48 pool, 50 trials — focused on validating
that grafting discovers structurally novel algorithms."""
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
    n_trials=50,
    timeout_sec=3.0,
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
    n_generations=10,
    seed=42,
)

engine = AlgorithmEvolutionEngine(
    evaluator=evaluator,
    config=evo_config,
    rng=np.random.default_rng(42),
    pattern_matcher=pattern_matcher,
)

best_grafted_ever = {"score": float("inf"), "id": None, "gen": 0}

def _cb(gen: int, best_f, pop):
    score = best_f.composite_score() if best_f else float("inf")
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

    if best_graft_score < best_grafted_ever["score"]:
        best_grafted_ever["score"] = best_graft_score
        best_grafted_ever["id"] = best_graft_id
        best_grafted_ever["gen"] = gen

    print(f"  Gen {gen:2d}: best={score:.6f}, grafted={n_grafted:2d}/48, "
          f"best_graft={best_graft_score:.6f}")

    # Every 5 gens, show graft history of best grafted
    if gen % 5 == 0 and best_graft_id:
        bg = next((g for g in pop if g.algo_id == best_graft_id), None)
        if bg and bg.graft_history:
            gh = bg.graft_history[-1]
            print(f"         Best graft: host={gh.host_algo_id[:8]}, "
                  f"donor={gh.donor_algo_id[:8] if gh.donor_algo_id else '?'}")
            # Count ops in the grafted IR
            n_ops = len(bg.structural_ir.ops)
            print(f"         IR size: {n_ops} ops, {len(bg.slot_populations)} slots")

print("=" * 60)
print("FAST DISCOVERY: 10 gen, pool=48, 50 trials")
print("=" * 60)

best = engine.run(callback=_cb)

print("\n" + "=" * 60)
print(f"BEST OVERALL: {best.algo_id}, score={engine.best_fitness.composite_score():.6f}")
print(f"BEST GRAFTED: {best_grafted_ever['id']}, score={best_grafted_ever['score']:.6f} (gen {best_grafted_ever['gen']})")
print(f"Grafted in final pop: {sum(1 for g in engine.population if 'grafted' in g.tags)}/48")

# Show graft history of best grafted
bg = next((g for g in engine.population if g.algo_id == best_grafted_ever["id"]), None)
if bg:
    print(f"\nBest grafted algorithm structure:")
    print(f"  Graft history ({len(bg.graft_history)} grafts):")
    for gh in bg.graft_history:
        print(f"    Gen {gh.generation}: host={gh.host_algo_id[:8]}... + donor={gh.donor_algo_id[:8] if gh.donor_algo_id else '?'}...")
        if gh.new_slots_created:
            print(f"      New slots: {gh.new_slots_created}")
    print(f"  IR: {len(bg.structural_ir.ops)} ops, {len(bg.slot_populations)} slots")
    print(f"  Parent IDs: {bg.parent_ids}")

    # Check if it's truly novel (different structure from all originals)
    original_ids = set()
    for g in engine.gene_bank:
        if "grafted" not in g.tags:
            original_ids.add(len(g.structural_ir.ops))
    grafted_n_ops = len(bg.structural_ir.ops)
    if grafted_n_ops not in original_ids:
        print(f"\n  *** NOVEL STRUCTURE: {grafted_n_ops} ops (not in any original) ***")
    else:
        print(f"\n  Structure size {grafted_n_ops} ops (matches some originals)")

print("=" * 60)
