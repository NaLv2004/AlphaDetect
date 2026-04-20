"""Massive evolution run for GNN donor-region-selector training.

Trains GNNPatternMatcher via REINFORCE through large-scale algorithm
evolution.  The GNN learns to select matching sub-regions from donor
algorithms, enabling the discovery of structurally novel MIMO detectors.

Usage:
    conda run -n AutoGenOld python _run_massive_evo.py
    conda run -n AutoGenOld python _run_massive_evo.py --quick   # 15 gen

Tracks per generation:
  - best overall score (SER composite)
  - best grafted algorithm score
  - number of grafted algorithms in population
  - GNN reward baseline (shows whether GNN is learning)
  - number of GNN training samples seen

Final report:
  - GNN learning curve (early vs late reward baseline)
  - Novel grafted algorithm list with op counts
  - IR opcode analysis of best discovered algorithm
  - Checkpoint saved to gnn_ckpt_final.pt
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import time
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from evolution.algorithm_engine import AlgorithmEvolutionEngine
from evolution.gnn_pattern_matcher import GNNPatternMatcher
from evolution.mimo_evaluator import MIMOEvalConfig, MIMOFitnessEvaluator
from evolution.pattern_matchers import (
    CompositePatternMatcher,
    ExpertPatternMatcher,
    StaticStructurePatternMatcher,
)
from evolution.pool_types import AlgorithmEvolutionConfig

# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
QUICK_MODE = "--quick" in sys.argv

N_GENERATIONS = 15 if QUICK_MODE else 100
POOL_SIZE     = 8  if QUICK_MODE else 24
N_TRIALS      = 10
TIMEOUT_SEC   = 2.0
SEED          = 42

# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "_massive_evo.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

os = __import__("os")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# ─────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────
print("=" * 72)
print(f"MASSIVE EVOLUTION  |  {N_GENERATIONS} generations  |  pool={POOL_SIZE}")
print("Goal: GNN learns donor-region selection → novel MIMO algorithms")
print("=" * 72)

eval_config = MIMOEvalConfig(
    Nr=16, Nt=16, mod_order=16,
    snr_db_list=[24.0],
    n_trials=N_TRIALS,
    timeout_sec=TIMEOUT_SEC,
    complexity_weight=0.001,
    seed=SEED,
)

evo_config = AlgorithmEvolutionConfig(
    pool_size=POOL_SIZE,
    n_generations=N_GENERATIONS,
    micro_generations=1,
    micro_pop_size=3,
    micro_mutation_rate=0.4,
    seed=SEED,
)

evaluator = MIMOFitnessEvaluator(eval_config)

# GNN: 6 proposals/gen × 100 gen = 600 training samples
gnn_matcher = GNNPatternMatcher(
    max_proposals_per_gen=6,
    top_k_pairs=8,
    min_region_size=3,
    max_region_size=12,
    lr=5e-4,
    buffer_size=512,
    train_interval=1,
)

pattern_matcher = CompositePatternMatcher([
    ExpertPatternMatcher(max_proposals_per_gen=2),
    StaticStructurePatternMatcher(max_proposals_per_gen=2),
    gnn_matcher,
])

engine = AlgorithmEvolutionEngine(
    evaluator=evaluator,
    config=evo_config,
    rng=np.random.default_rng(SEED),
    pattern_matcher=pattern_matcher,
)

# ─────────────────────────────────────────────────────────────────────
# Per-generation callback
# ─────────────────────────────────────────────────────────────────────
gen_log: list[dict] = []
best_grafted_ever: float = float("inf")
discovered: list[dict] = []  # novel grafted algorithms


def _cb(gen: int, best_f, pop) -> None:
    global best_grafted_ever

    score = best_f.composite_score() if best_f else float("inf")

    # Grafted individuals in the population
    grafted_genomes = [g for g in pop if "grafted" in g.tags]
    n_grafted = len(grafted_genomes)

    # Best grafted score
    best_graft_score = float("inf")
    best_graft_genome = None
    for g in grafted_genomes:
        try:
            idx = next(i for i, p in enumerate(pop) if p is g)
            if idx < len(engine.fitness):
                gs = engine.fitness[idx].composite_score()
                if gs < best_graft_score:
                    best_graft_score = gs
                    best_graft_genome = g
        except StopIteration:
            pass

    # Track discoveries: grafted algo with score < 0.1 (working detector)
    if best_graft_score < 0.1 and best_graft_score < best_grafted_ever:
        best_grafted_ever = best_graft_score
        if best_graft_genome is not None:
            algo_id = best_graft_genome.algo_id
            if not any(d["algo_id"] == algo_id for d in discovered):
                ir = best_graft_genome.structural_ir
                n_ops = len(ir.ops) if ir else "?"
                discovered.append({
                    "gen": gen,
                    "algo_id": algo_id,
                    "score": best_graft_score,
                    "n_ops": n_ops,
                })
                print(f"\n  *** DISCOVERY @ Gen {gen}: {algo_id} "
                      f"score={best_graft_score:.6f}  ops={n_ops} ***\n")

    # GNN stats
    gnn_stats = gnn_matcher.get_stats()
    baseline = gnn_matcher._reward_baseline

    record = {
        "gen": gen,
        "best_score": score,
        "n_grafted": n_grafted,
        "best_graft_score": best_graft_score if best_graft_score < float("inf") else None,
        "gnn_proposals": gnn_stats["total_proposals"],
        "gnn_baseline": baseline,
    }
    gen_log.append(record)

    graft_str = f"{best_graft_score:.6f}" if best_graft_score < float("inf") else "   n/a  "
    print(
        f"  Gen {gen:3d}: best={score:.6f}  grafted={n_grafted:2d}/{POOL_SIZE}"
        f"  best_graft={graft_str}"
        f"  gnn_baseline={baseline:+.4f}"
        f"  proposals={gnn_stats['total_proposals']:4d}"
    )

    # Periodic detailed reports
    if gen % 10 == 0 and gen > 0:
        print(f"         [GNN] buffer={gnn_stats['experience_buffer']:3d} "
              f"reward_buffer={gnn_stats['outcome_buffer']:3d}")
        if discovered:
            print(f"         [Discoveries so far: {len(discovered)}]")
            for d in discovered[-3:]:
                print(f"           Gen {d['gen']:3d}: {d['algo_id']} "
                      f"score={d['score']:.6f}  ops={d['n_ops']}")
        # Save GNN checkpoint
        _save_checkpoint(gen)


def _save_checkpoint(gen: int) -> None:
    """Save GNN model weights."""
    ckpt = ROOT / f"gnn_ckpt_gen{gen}.pt"
    torch.save({
        "encoder": gnn_matcher.encoder.state_dict(),
        "scorer": gnn_matcher.scorer.state_dict(),
        "region_proposer": gnn_matcher.region_proposer.state_dict(),
        "donor_region_selector": gnn_matcher.donor_region_selector.state_dict(),
        "generation": gen,
        "reward_baseline": gnn_matcher._reward_baseline,
        "gen_log": gen_log,
    }, ckpt)
    print(f"         [Checkpoint saved: {ckpt.name}]")


# ─────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
best = engine.run(callback=_cb)
elapsed = time.perf_counter() - t0

# ─────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"MASSIVE EVOLUTION COMPLETE  |  {elapsed / 60:.1f} min  |  {N_GENERATIONS} gen")
print(f"Final best: {best.algo_id}  score={engine.best_fitness.composite_score():.6f}")
print()

# ── GNN learning curve ──────────────────────────────────────────────
if len(gen_log) >= 5:
    early = [g["gnn_baseline"] for g in gen_log[:5]]
    late  = [g["gnn_baseline"] for g in gen_log[-5:]]
    e_mean = float(np.mean(early))
    l_mean = float(np.mean(late))
    delta  = l_mean - e_mean
    print(f"GNN learning curve:")
    print(f"  Early baseline (gen 1-5):  {e_mean:+.4f}")
    print(f"  Late  baseline (gen -5):   {l_mean:+.4f}")
    print(f"  Δ baseline (improvement):  {delta:+.4f}  "
          f"{'↑ GNN LEARNED' if delta > 0.001 else '→ flat / not yet converged'}")
    print()

# ── Novel discoveries ───────────────────────────────────────────────
print(f"Novel grafted algorithms discovered: {len(discovered)}")
for d in discovered:
    print(f"  Gen {d['gen']:3d}: {d['algo_id']}  score={d['score']:.6f}  ops={d['n_ops']}")

if not discovered:
    # Still useful: show best grafted score trend
    graft_scores = [g["best_graft_score"] for g in gen_log if g["best_graft_score"] is not None]
    if graft_scores:
        print(f"  Best grafted score ever: {min(graft_scores):.6f}")
        print(f"  Grafted score trend: gen-1={graft_scores[0]:.4f} "
              f"→ gen-last={graft_scores[-1]:.4f}")
    else:
        print("  No grafted algorithms survived selection.")
print()

# ── IR analysis of best discovered algorithm ────────────────────────
if discovered:
    best_d = min(discovered, key=lambda d: d["score"])
    print(f"Best discovery: {best_d['algo_id']}  score={best_d['score']:.6f}")
    all_genomes = engine.population + list(engine.gene_bank)
    best_genome = next((g for g in all_genomes if g.algo_id == best_d["algo_id"]), None)
    if best_genome and best_genome.structural_ir:
        ir = best_genome.structural_ir
        opcodes: dict[str, int] = {}
        for op in ir.ops.values():
            opcodes[op.opcode] = opcodes.get(op.opcode, 0) + 1
        calls = [
            op.attrs.get("callee", op.attrs.get("name", "?"))
            for op in ir.ops.values()
            if op.opcode == "call"
        ]
        print(f"  Total ops:  {len(ir.ops)}")
        print(f"  Blocks:     {len(ir.blocks)}")
        print(f"  Opcodes:    {opcodes}")
        if calls:
            cc = Counter(calls)
            print(f"  Func calls: {dict(cc.most_common(10))}")
        print()
        # Print parent chain
        print(f"  Parent IDs: {best_genome.parent_ids[:3]}")
        if best_genome.graft_history:
            for gr in best_genome.graft_history[:2]:
                print(f"  Graft: host={gr.host_algo_id}, "
                      f"donor={gr.donor_algo_id}, region={gr.region_op_count} ops")

# ── GNN final stats ─────────────────────────────────────────────────
final_stats = gnn_matcher.get_stats()
print(f"\nGNN final stats: {final_stats}")
print(f"GNN reward baseline: {gnn_matcher._reward_baseline:+.4f}")

# ── Save final checkpoint ───────────────────────────────────────────
ckpt_final = ROOT / "gnn_ckpt_final.pt"
torch.save({
    "encoder": gnn_matcher.encoder.state_dict(),
    "scorer": gnn_matcher.scorer.state_dict(),
    "region_proposer": gnn_matcher.region_proposer.state_dict(),
    "donor_region_selector": gnn_matcher.donor_region_selector.state_dict(),
    "generation": N_GENERATIONS,
    "reward_baseline": gnn_matcher._reward_baseline,
    "gen_log": gen_log,
    "discovered": discovered,
}, ckpt_final)
print(f"Final checkpoint: {ckpt_final}")

# ── Save JSON log ───────────────────────────────────────────────────
log_path = ROOT / "_massive_evo_results.json"
with open(log_path, "w") as f:
    json.dump({
        "n_generations": N_GENERATIONS,
        "pool_size": POOL_SIZE,
        "gen_log": gen_log,
        "discovered": discovered,
        "gnn_final_baseline": gnn_matcher._reward_baseline,
        "elapsed_minutes": elapsed / 60,
    }, f, indent=2)
print(f"JSON log: {log_path}")
print("=" * 72)
