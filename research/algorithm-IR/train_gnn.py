"""Dedicated GNN training script — massive-sample grafting evolution.

Trains the GNN pattern matcher with 500+ graft proposals per generation,
SNR curriculum (20dB→24dB), and the full 91-genome algorithm pool.

Outputs a human-readable log of the top-50 grafted individuals each
generation to ``results/gnn_training/top50_grafts.log``.

Usage:
    conda activate AutoGenOld
    python train_gnn.py [--gens 200] [--snr-start 20] [--snr-target 24]
"""
import sys
import os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import json
import logging
import re
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from evolution.pool_types import AlgorithmEvolutionConfig, AlgorithmGenome
from evolution.algorithm_engine import AlgorithmEvolutionEngine
from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
from evolution.materialize import materialize
from evolution.gnn_pattern_matcher import GNNPatternMatcher
from evolution.ir_pool import find_algslot_ops
from evolution.fitness import FitnessResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_gnn")

# ── CLI args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GNN-guided grafting trainer")
parser.add_argument("--gens", type=int, default=200, help="Total generations")
parser.add_argument("--snr-start", type=float, default=20.0, help="Initial SNR (dB)")
parser.add_argument("--snr-target", type=float, default=24.0, help="Target SNR (dB)")
parser.add_argument("--phase-thresh", type=float, default=0.5,
                    help="Fraction of grafts with SER<0.1 to trigger SNR increase")
parser.add_argument("--proposals", type=int, default=500,
                    help="GNN graft proposals per generation")
parser.add_argument("--top-k", type=int, default=20,
                    help="Deprecated compatibility arg; top-k pair pruning is disabled")
parser.add_argument("--pool-size", type=int, default=141,
                    help="Population size (91 original + 50 grafted survivors)")
parser.add_argument("--n-trials", type=int, default=5,
                    help="Evaluation trials per individual (320 bits/trial)")
parser.add_argument("--timeout", type=float, default=1.5,
                    help="Per-genome evaluation timeout in seconds (lower = faster, less accurate)")
parser.add_argument("--eval-workers", type=int, default=1,
                    help="Parallel genome evaluations for the main evaluator")
parser.add_argument("--warmstart-gens", type=int, default=1,
                    help="Number of early generations that evaluate every host/donor pair once")
parser.add_argument("--warmstart-trials", type=int, default=1,
                    help="Trials per graft during warm-start pair sweeps")
parser.add_argument("--warmstart-timeout", type=float, default=0.5,
                    help="Per-genome timeout for the lightweight warm-start graft evaluator")
parser.add_argument("--warmstart-eval-workers", type=int, default=8,
                    help="Parallel genome evaluations for the warm-start graft evaluator")
parser.add_argument("--warmstart-survivor-cap", type=int, default=48,
                    help="How many warm-start graft children may enter population selection")
parser.add_argument("--train-steps", type=int, default=5,
                    help="Mini-batch gradient steps per generation")
parser.add_argument("--train-interval", type=int, default=1,
                    help="Train GNN every N generations")
parser.add_argument("--buffer-size", type=int, default=20000,
                    help="Replay buffer size for graft experiences")
parser.add_argument("--pair-temp", type=float, default=0.7,
                    help="Softmax temperature for pair sampling after warm-start")
parser.add_argument("--pair-eps", type=float, default=0.10,
                    help="Uniform exploration mix for pair sampling")
parser.add_argument("--region-eps", type=float, default=0.10,
                    help="Uniform exploration mix for host region sampling")
parser.add_argument("--donor-eps", type=float, default=0.10,
                    help="Uniform exploration mix for donor region sampling")
parser.add_argument("--ckpt-interval", type=int, default=10,
                    help="Checkpoint every N generations")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default=None,
                    help="Path to GNN checkpoint to resume from")
args = parser.parse_args()

# ── Output directory ──────────────────────────────────────────────────────
out_dir = Path("results/gnn_training")
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "training_log.jsonl"
top50_log_path = out_dir / "top50_grafts.log"

# ══════════════════════════════════════════════════════════════════════════
# Graft quality analysis utilities
# ══════════════════════════════════════════════════════════════════════════

def _analyse_graft_quality(genome: AlgorithmGenome, source: str) -> dict:
    """Analyse whether a grafted genome is non-trivial.

    Returns a dict with quality flags:
      - is_grafted: bool — has graft history
      - n_graft_ops: int — how many grafted ops in the IR
      - has_dead_graft: bool — grafted ops whose outputs are never used
      - is_cascade: bool — source is just two detector calls in sequence
      - is_identity: bool — looks like the original algo (graft is no-op)
      - n_slots: int — number of AlgSlot ops
      - slot_pop_sizes: dict — {slot_id: n_variants}
      - graft_lineage: str — human-readable lineage
      - quality_verdict: str — TRIVIAL / DEAD_CODE / CASCADE / NON_TRIVIAL
    """
    result: dict = {
        "is_grafted": bool(genome.graft_history),
        "n_graft_ops": 0,
        "has_dead_graft": False,
        "is_cascade": False,
        "is_identity": False,
        "n_slots": 0,
        "slot_pop_sizes": {},
        "graft_lineage": "",
        "quality_verdict": "UNKNOWN",
    }

    if not genome.graft_history:
        result["quality_verdict"] = "NOT_GRAFTED"
        return result

    # Lineage
    lineage_parts = []
    for gr in genome.graft_history:
        lineage_parts.append(f"{gr.host_algo_id}<-{gr.donor_algo_id}")
    result["graft_lineage"] = " | ".join(lineage_parts)

    # Count grafted ops in the IR
    ir = genome.structural_ir
    grafted_ops = []
    for op in ir.ops.values():
        if op.attrs.get("grafted"):
            grafted_ops.append(op)
    result["n_graft_ops"] = len(grafted_ops)

    # Dead-code detection: grafted ops whose outputs have no downstream uses
    n_dead = 0
    for op in grafted_ops:
        all_dead = True
        for out_vid in op.outputs:
            val = ir.values.get(out_vid)
            if val and val.use_ops:
                all_dead = False
                break
        if all_dead and op.outputs:
            n_dead += 1
    result["has_dead_graft"] = n_dead > 0
    result["n_dead_graft_ops"] = n_dead

    # Detect if graft produced zero ops (identity graft)
    if len(grafted_ops) == 0:
        result["is_identity"] = True

    # Slot analysis
    slot_ops = find_algslot_ops(ir)
    result["n_slots"] = len(slot_ops)
    for sid, pop in genome.slot_populations.items():
        result["slot_pop_sizes"][sid] = len(pop.variants)

    # Cascade detection: look for pattern of two separate detector call
    # sequences with no data mixing between them (output of first is not
    # input of second's computation, just overwritten)
    # Simple heuristic: if source has two or more top-level "return"
    # patterns or the graft just prepends code before the return but
    # doesn't connect to the computation
    lines = source.strip().split("\n")
    def_count = sum(1 for l in lines if l.strip().startswith("def "))
    if def_count >= 3:
        # Check if the main function body is just calling two sub-functions
        # in sequence and returning the last one's result
        main_body_calls = re.findall(r'\b_slot_\w+\b', source)
        # This is normal — slots are expected
    # More robust cascade check: if ALL grafted ops are call ops to known
    # detector functions, and none of them feed into each other
    call_grafts = [op for op in grafted_ops if op.opcode == "call"]
    if len(call_grafts) >= 2 and len(call_grafts) == len(grafted_ops):
        # Check if they're independent (no output of one feeds into another)
        graft_outputs = set()
        graft_inputs = set()
        for op in call_grafts:
            graft_outputs.update(op.outputs)
            graft_inputs.update(op.inputs)
        if not (graft_outputs & graft_inputs):
            result["is_cascade"] = True

    # Verdict
    if result["is_identity"]:
        result["quality_verdict"] = "IDENTITY"
    elif result["has_dead_graft"] and n_dead == len(grafted_ops):
        result["quality_verdict"] = "DEAD_CODE"
    elif result["is_cascade"]:
        result["quality_verdict"] = "CASCADE"
    elif result["has_dead_graft"]:
        result["quality_verdict"] = "PARTIAL_DEAD"
    else:
        result["quality_verdict"] = "NON_TRIVIAL"

    return result


def _log_top50_grafts(
    gen: int,
    population: list[AlgorithmGenome],
    fitness: list[FitnessResult],
    snr_db: float,
    log_file: Path,
) -> dict:
    """Log top-50 grafted individuals with human-readable source & analysis.

    Returns summary stats for this generation's grafts.
    """
    # Pair grafted genomes with fitness
    grafted_pairs = []
    for g, f in zip(population, fitness):
        if "grafted" in g.tags:
            grafted_pairs.append((g, f))

    # Sort by SER (best first)
    grafted_pairs.sort(key=lambda x: x[1].composite_score())
    top50 = grafted_pairs[:50]

    n_nontrivial = 0
    n_dead = 0
    n_identity = 0
    n_cascade = 0

    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(f"\n{'='*80}\n")
        fh.write(f"GENERATION {gen} | SNR={snr_db:.0f}dB | "
                 f"{len(grafted_pairs)} grafted in population\n")
        fh.write(f"{'='*80}\n\n")

        for rank, (genome, fit) in enumerate(top50, 1):
            ser = fit.metrics.get("ser", float("inf"))
            score = fit.composite_score()

            try:
                source = materialize(genome)
            except Exception as e:
                source = f"<materialize failed: {e}>"

            quality = _analyse_graft_quality(genome, source)

            if quality["quality_verdict"] == "NON_TRIVIAL":
                n_nontrivial += 1
            elif quality["quality_verdict"] == "DEAD_CODE":
                n_dead += 1
            elif quality["quality_verdict"] == "IDENTITY":
                n_identity += 1
            elif quality["quality_verdict"] == "CASCADE":
                n_cascade += 1

            fh.write(f"--- #{rank} | {genome.algo_id} | SER={ser:.6f} | "
                     f"score={score:.6f} | verdict={quality['quality_verdict']} ---\n")
            fh.write(f"  Lineage: {quality['graft_lineage']}\n")
            fh.write(f"  Graft ops: {quality['n_graft_ops']} "
                     f"(dead: {quality.get('n_dead_graft_ops', 0)})\n")
            fh.write(f"  Slots: {quality['n_slots']} | "
                     f"Slot pops: {quality['slot_pop_sizes']}\n")
            fh.write(f"  Parents: {genome.parent_ids}\n")
            # Truncate source to 1500 chars for readability
            src_preview = source[:1500]
            if len(source) > 1500:
                src_preview += f"\n  ... ({len(source)} chars total)"
            fh.write(f"  Source:\n")
            for line in src_preview.split("\n"):
                fh.write(f"    {line}\n")
            fh.write("\n")

        fh.write(f"SUMMARY: {n_nontrivial} non-trivial, {n_dead} dead-code, "
                 f"{n_identity} identity, {n_cascade} cascade, "
                 f"{len(top50)} total\n\n")

    return {
        "n_grafted_total": len(grafted_pairs),
        "n_top50": len(top50),
        "n_nontrivial": n_nontrivial,
        "n_dead": n_dead,
        "n_identity": n_identity,
        "n_cascade": n_cascade,
    }


# ══════════════════════════════════════════════════════════════════════════
# Core setup
# ══════════════════════════════════════════════════════════════════════════

current_snr = args.snr_start
snr_step = 2.0


def make_evaluator(
    snr_db: float,
    n_trials: int,
    *,
    timeout_sec: float,
    batch_workers: int,
) -> MIMOFitnessEvaluator:
    """Create a MIMO evaluator at the given SNR."""
    return MIMOFitnessEvaluator(MIMOEvalConfig(
        Nr=16, Nt=16, mod_order=16,
        snr_db_list=[snr_db],
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        complexity_weight=0.01,
        batch_workers=batch_workers,
        seed=args.seed,
    ))


gnn_matcher = GNNPatternMatcher(
    max_proposals_per_gen=args.proposals,
    top_k_pairs=args.top_k,
    min_region_size=1,
    max_region_size=8,
    lr=1e-3,
    buffer_size=args.buffer_size,
    train_interval=args.train_interval,
    train_steps=args.train_steps,
    warmstart_generations=args.warmstart_gens,
    pair_temperature=args.pair_temp,
    pair_exploration=args.pair_eps,
    region_exploration=args.region_eps,
    donor_exploration=args.donor_eps,
)

if args.resume and Path(args.resume).exists():
    ckpt = torch.load(args.resume, map_location=gnn_matcher.device, weights_only=False)
    gnn_matcher.encoder.load_state_dict(ckpt["encoder"], strict=False)
    gnn_matcher.scorer.load_state_dict(ckpt["scorer"], strict=False)
    gnn_matcher.region_proposer.load_state_dict(ckpt["region_proposer"], strict=False)
    gnn_matcher.donor_region_selector.load_state_dict(ckpt["donor_region_selector"], strict=False)
    if "optimizer" in ckpt:
        try:
            gnn_matcher.optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            logger.warning("Optimizer state restore skipped: %s", exc)
    logger.info("Resumed GNN from checkpoint: %s", args.resume)

evo_config = AlgorithmEvolutionConfig(
    pool_size=args.pool_size,
    n_generations=1,  # we drive the loop ourselves
    seed=args.seed,
    micro_generations=1,  # Reduce from default 5: each gen is slow with functional slots
)

evaluator = make_evaluator(
    current_snr,
    args.n_trials,
    timeout_sec=args.timeout,
    batch_workers=args.eval_workers,
)
engine = AlgorithmEvolutionEngine(
    evaluator=evaluator,
    config=evo_config,
    rng=np.random.default_rng(args.seed),
    pattern_matcher=gnn_matcher,
)
engine.graft_eval_evaluator = make_evaluator(
    current_snr,
    args.warmstart_trials,
    timeout_sec=args.warmstart_timeout,
    batch_workers=args.warmstart_eval_workers,
)
engine.graft_survivor_cap = args.warmstart_survivor_cap
engine.init_population()
logger.info(
    "Warm-start config: gens=%d trials=%d timeout=%.2fs main_workers=%d warm_workers=%d survivor_cap=%d",
    args.warmstart_gens,
    args.warmstart_trials,
    args.warmstart_timeout,
    args.eval_workers,
    args.warmstart_eval_workers,
    args.warmstart_survivor_cap,
)

logger.info(
    "Starting GNN training: %d gens, SNR=%.0fdB→%.0fdB, "
    "%d proposals/gen, pool=%d, trials=%d",
    args.gens, args.snr_start, args.snr_target,
    args.proposals, len(engine.population), args.n_trials,
)


def save_checkpoint(gen: int, tag: str = "") -> Path:
    """Save GNN model checkpoint."""
    suffix = f"_{tag}" if tag else ""
    path = out_dir / f"gnn_ckpt_gen{gen}{suffix}.pt"
    torch.save({
        "generation": gen,
        "snr_db": current_snr,
        "encoder": gnn_matcher.encoder.state_dict(),
        "scorer": gnn_matcher.scorer.state_dict(),
        "region_proposer": gnn_matcher.region_proposer.state_dict(),
        "donor_region_selector": gnn_matcher.donor_region_selector.state_dict(),
        "optimizer": gnn_matcher.optimizer.state_dict(),
    }, path)
    return path


# ── Tracking ──────────────────────────────────────────────────────────────
best_ser_ever = float("inf")
phase_history: list[dict] = []
t_start = time.perf_counter()

# Clear previous log
with open(top50_log_path, "w", encoding="utf-8") as fh:
    fh.write("GNN Grafting Training Log\n")
    fh.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    fh.write(f"Config: {vars(args)}\n\n")

# ── Main training loop ────────────────────────────────────────────────────
for gen in range(1, args.gens + 1):
    t0 = time.perf_counter()

    # Run one generation through the engine
    try:
        engine.run(n_generations=1)
    except Exception as exc:
        logger.error("Engine.run failed at gen %d: %s", gen, exc)
        logger.error(traceback.format_exc())
        continue

    # Collect generation stats
    gen_sers = []
    for f in engine.fitness:
        ser = f.metrics.get("ser", float("inf"))
        gen_sers.append(ser)

    best_ser = min(gen_sers) if gen_sers else float("inf")
    median_ser = float(np.median(gen_sers)) if gen_sers else float("inf")
    n_good = sum(1 for s in gen_sers if s < 0.1)
    frac_good = n_good / len(gen_sers) if gen_sers else 0.0
    n_grafted = sum(1 for g in engine.population if "grafted" in g.tags)

    if best_ser < best_ser_ever:
        best_ser_ever = best_ser

    elapsed = time.perf_counter() - t0

    # ── Log top-50 grafted individuals ─────────────────────────────────
    graft_stats = _log_top50_grafts(
        gen, engine.population, engine.fitness, current_snr, top50_log_path,
    )

    # ── SNR curriculum: escalate only when GNN has learned meaningful grafts ──
    # frac_good measures the whole population (incl. non-grafted).
    # Use nontrivial graft fraction + minimum gens at current SNR instead.
    snr_changed = False
    gens_at_current_snr = sum(1 for r in phase_history if r.get("snr_db") == current_snr)
    n_nontrivial = graft_stats.get("n_nontrivial", 0)
    frac_nontrivial_grafts = n_nontrivial / max(n_grafted, 1) if n_grafted > 0 else 0.0
    if (
        gens_at_current_snr >= 20             # spend at least 20 gens at each SNR
        and frac_nontrivial_grafts >= args.phase_thresh  # and most grafts must be non-trivial
        and current_snr < args.snr_target
    ):
        old_snr = current_snr
        current_snr = min(current_snr + snr_step, args.snr_target)
        evaluator = make_evaluator(
            current_snr,
            args.n_trials,
            timeout_sec=args.timeout,
            batch_workers=args.eval_workers,
        )
        engine.evaluator = evaluator
        engine.graft_eval_evaluator = make_evaluator(
            current_snr,
            args.warmstart_trials,
            timeout_sec=args.warmstart_timeout,
            batch_workers=args.warmstart_eval_workers,
        )
        snr_changed = True
        logger.info(
            "*** SNR escalation: %.0fdB → %.0fdB (%d gens at %.0fdB, %.0f%% nontrivial) ***",
            old_snr, current_snr, gens_at_current_snr, old_snr, frac_nontrivial_grafts * 100,
        )

    # ── Logging ────────────────────────────────────────────────────────
    record = {
        "gen": gen,
        "snr_db": current_snr,
        "best_ser": best_ser,
        "median_ser": median_ser,
        "best_ser_ever": best_ser_ever,
        "frac_good": frac_good,
        "frac_nontrivial_grafts": round(frac_nontrivial_grafts, 3),
        "gens_at_snr": gens_at_current_snr,
        "n_grafted": n_grafted,
        "pop_size": len(engine.population),
        "elapsed_sec": round(elapsed, 1),
        "snr_changed": snr_changed,
        "graft_quality": graft_stats,
        "matcher_stats": gnn_matcher.get_stats(),
        "engine_stats": dict(engine.last_generation_stats),
    }
    phase_history.append(record)

    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.info(
        "Gen %3d | SNR=%.0fdB | best_SER=%.6f | median=%.4f | "
        "good=%.0f%% | grafted=%d (nontrivial=%d, dead=%d) | %.1fs",
        gen, current_snr, best_ser, median_ser,
        frac_good * 100, n_grafted,
        graft_stats.get("n_nontrivial", 0),
        graft_stats.get("n_dead", 0),
        elapsed,
    )
    matcher_stats = gnn_matcher.get_stats()
    logger.info(
        "  Train stats: matched=%s score=%.4f baseline=%.4f | proposals=%s pairs=%s/%s | graft_eval=%s kept=%s warmstart=%s",
        matcher_stats.get("last_train", {}).get("matched_samples"),
        matcher_stats.get("last_train", {}).get("mean_graft_score", float("nan")),
        matcher_stats.get("reward_baseline", 0.0),
        matcher_stats.get("last_proposals", {}).get("proposals_built"),
        matcher_stats.get("last_proposals", {}).get("pair_candidates_selected"),
        matcher_stats.get("last_proposals", {}).get("pair_candidates_scored"),
        engine.last_generation_stats.get("graft_evaluated"),
        engine.last_generation_stats.get("graft_kept"),
        engine.last_generation_stats.get("warmstart_active"),
    )

    # ── Checkpointing ─────────────────────────────────────────────────
    if gen % args.ckpt_interval == 0:
        ckpt_path = save_checkpoint(gen)
        logger.info("Checkpoint saved: %s", ckpt_path)

    # ── Population diversity check ─────────────────────────────────────
    algo_ids = [g.algo_id.split("_")[0] if "_" in g.algo_id else g.algo_id
                for g in engine.population]
    unique_base = len(set(g.parent_ids[0] if g.parent_ids else g.algo_id
                          for g in engine.population))
    if unique_base < 5:
        logger.warning("POPULATION HOMOGENIZED: only %d unique base algos", unique_base)

# ── Final save ────────────────────────────────────────────────────────────
total_time = time.perf_counter() - t_start
save_checkpoint(gen, tag="final")

if engine.best_genome:
    try:
        src = materialize(engine.best_genome)
        with open(out_dir / "best_detector.py", "w") as f:
            f.write(src)
    except Exception as e:
        logger.warning("Materialize failed: %s", e)

    with open(out_dir / "training_summary.json", "w") as f:
        json.dump({
            "total_generations": gen,
            "total_time_sec": round(total_time, 1),
            "final_snr_db": current_snr,
            "best_ser_ever": best_ser_ever,
            "best_algo": engine.best_genome.algo_id,
            "args": vars(args),
            "phase_history": phase_history[-20:],
        }, f, indent=2, default=str)

print("\n" + "=" * 60)
print("GNN TRAINING COMPLETE")
print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
print(f"Generations: {gen}")
print(f"Final SNR: {current_snr:.0f}dB")
print(f"Best SER ever: {best_ser_ever:.6f}")
if engine.best_genome:
    print(f"Best algo: {engine.best_genome.algo_id}")
print(f"Results: {out_dir}")
print(f"Top-50 log: {top50_log_path}")
print("=" * 60)
