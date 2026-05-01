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

# Periodically dump all thread stacks so we can diagnose silent hangs.
# Note: a previous incarnation of this used dump_traceback_later(120, repeat=True),
# but the watchdog dump itself segfaulted inside the xdsl printer when a frame
# held an xdsl op + numpy array combination.  Disabled by default; re-enable
# manually with TRAIN_GNN_WATCHDOG_SEC env var.
import faulthandler
import pickle
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
faulthandler.enable()
try:
    _wd = int(os.environ.get("TRAIN_GNN_WATCHDOG_SEC", "0") or 0)
    if _wd > 0:
        faulthandler.dump_traceback_later(_wd, repeat=True)
except Exception:
    pass

import argparse
import json
import logging
import re
import time
import traceback
import warnings
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from evolution.pool_types import AlgorithmEvolutionConfig, AlgorithmGenome
from evolution.algorithm_engine import AlgorithmEvolutionEngine
from evolution.mimo_evaluator import (
    MIMOFitnessEvaluator,
    MIMOEvalConfig,
    generate_mimo_sample,
    qam16_constellation,
)
from evolution.materialize import materialize, materialize_to_callable
from evolution.gnn_pattern_matcher import GNNPatternMatcher
from evolution.ir_pool import find_algslot_ops
from evolution.fitness import FitnessResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_gnn")

# ── Config file support ──────────────────────────────────────────────────
# Two-step parse: first check for --config-path, then merge JSON overrides
# with the remaining CLI args.
_config_parser = argparse.ArgumentParser(add_help=False)
_config_parser.add_argument("--config-path", type=str, default=None)
_config_args, _remaining = _config_parser.parse_known_args()

if _config_args.config_path:
    _cfg = json.loads(Path(_config_args.config_path).read_text(encoding="utf-8"))
    # Convert snake_case JSON keys to --kebab-case CLI strings
    _cfg_argv = []
    for _key, _val in _cfg.items():
        _flag = "--" + _key.replace("_", "-")
        if isinstance(_val, bool):
            if _val:
                _cfg_argv.append(_flag)
            # false booleans are skipped (they use the default False)
        elif _val is None:
            continue
        else:
            _cfg_argv.append(_flag)
            _cfg_argv.append(str(_val))
    # Merge: config values first, then remaining CLI args (CLI overrides config)
    sys.argv = [sys.argv[0]] + _cfg_argv + _remaining

# ── CLI args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GNN-guided grafting trainer")
parser.add_argument("--config-path", type=str, default=None,
                    help="Path to JSON config file. If provided, loads all params "
                         "from this file. CLI args override config values.")
parser.add_argument("--gens", type=int, default=200, help="Total generations")
parser.add_argument("--snr-start", type=float, default=16.0, help="Initial SNR (dB)")
parser.add_argument("--snr-target", type=float, default=20.0, help="Target SNR (dB)")
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
parser.add_argument("--warmstart-gens", type=int, default=20,
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
parser.add_argument("--proposal-batch", action="store_true",
                    help="Enable batched GNN graft proposal forward passes")
parser.add_argument("--proposal-batch-size", type=int, default=64,
                    help="Batch size for batched GNN graft proposal forward passes")
parser.add_argument("--max-boundary-outputs", type=int, default=2,
                    help="Maximum number of BCIR output-boundary values per sampled region")
parser.add_argument("--max-cut-values", type=int, default=3,
                    help="Maximum number of BCIR cut-boundary values per sampled region")
parser.add_argument("--max-region-ops", type=int, default=24,
                    help="Maximum legal BCIR region size in ops")
parser.add_argument("--max-region-inputs", type=int, default=8,
                    help="Maximum legal BCIR region input count")
parser.add_argument("--max-region-outputs", type=int, default=2,
                    help="Maximum legal BCIR region output count")
parser.add_argument("--effectiveness-snr", type=float, default=16.0,
                    help="Low SNR used for behavior-difference probes of grafted algorithms")
parser.add_argument("--effectiveness-samples", type=int, default=8,
                    help="Number of low-SNR probe samples used for graft behavior validation (>5 recommended)")
parser.add_argument("--probe-nr", type=int, default=16,
                    help="Nr (receive antennas) for behavior probe MIMO samples")
parser.add_argument("--probe-nt", type=int, default=16,
                    help="Nt (transmit antennas) for behavior probe MIMO samples")
parser.add_argument("--probe-timeout", type=float, default=3.0,
                    help="Per-call timeout (seconds) for behavior probe function execution")
parser.add_argument("--effective-margin", type=float, default=0.01,
                    help="Required rough score improvement over the host to count as performance-effective")
parser.add_argument("--precise-topk", type=int, default=50,
                    help="How many effective graft samples get precise SER evaluation each generation")
parser.add_argument("--precise-target-errors", type=int, default=100,
                    help="Stop precise SER evaluation after at least this many symbol errors")
parser.add_argument("--precise-max-symbols", type=int, default=200000,
                    help="Safety cap on evaluated symbols during precise SER estimation")
parser.add_argument("--precise-timeout", type=float, default=20.0,
                    help="Per-genome timeout for precise SER estimation")
parser.add_argument("--progress", action="store_true",
                    help="Show tqdm progress bars for proposal generation and evaluation")
parser.add_argument("--verbose", action="store_true",
                    help="Show Python warnings and detailed exception diagnostics")
parser.add_argument("--no-host-mask", action="store_true",
                    help="Disable the 4-layer host region mask (legacy permissive sampling + greedy repair)")
parser.add_argument("--no-donor-mask", action="store_true",
                    help="Disable the 4-layer donor region mask (Layer D1 prefilter + D2 output + D4 cut/STOP)")
# ── GNN training-objective / RL flags ───────────────────────────────
parser.add_argument("--lambda-rl", type=float, default=1.0,
                    help="Weight on the REINFORCE policy-gradient term in _train_step.")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="Coefficient on per-step action entropy bonus.")
parser.add_argument("--value-loss-weight", type=float, default=0.5,
                    help="Weight on the critic value-loss (V vs reward).")
parser.add_argument("--scorer-score-weight", type=float, default=0.1,
                    help="Weight on the legacy MSE pair-score regression head.")
parser.add_argument("--scorer-reasonable-weight", type=float, default=1.0,
                    help="Weight on the binary 'reasonable graft' BCE head.")
parser.add_argument("--scorer-behavior-weight", type=float, default=0.5,
                    help="Weight on the behavior-change-rate regression head.")
parser.add_argument("--scorer-perf-weight", type=float, default=0.2,
                    help="Weight on the performance-improvement head.")
parser.add_argument("--failed-replay-weight", type=float, default=1.0,
                    help="Sample-weight multiplier for failed (negative) experiences.")
parser.add_argument("--training-objective", type=str, default="reasonable_first",
                    choices=["reasonable_first", "performance_first"],
                    help="Training objective ordering: reasonable_first prioritises legality.")
parser.add_argument("--ckpt-interval", type=int, default=10,
                    help="Checkpoint every N generations")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default=None,
                    help="Path to GNN checkpoint to resume from")
parser.add_argument("--viz-grafts-per-gen", type=int, default=3,
                    help="How many graft samples to render as ASCII dataflow each "
                         "generation (host/donor/grafted). 0 disables.")
parser.add_argument("--viz-color", action="store_true", default=True,
                    help="Emit ANSI colour escape codes in graft visualisations.")
parser.add_argument("--no-viz-color", dest="viz_color", action="store_false",
                    help="Disable ANSI colour codes in graft visualisations.")
parser.add_argument("--subprocess-eval", action="store_true", default=True,
                    help="Run each genome evaluation inside a kill-able worker "
                         "subprocess.  Required to keep auto-repaired grafts "
                         "(which can hang inside numpy) from saturating CPU.")
parser.add_argument("--no-subprocess-eval", dest="subprocess_eval",
                    action="store_false",
                    help="Use in-process threaded evaluation (fast but unsafe "
                         "with auto-repair-prone grafts).")
parser.add_argument("--micro-pop-size", type=int, default=32,
                    help="Micro-population size per slot (was 8). Larger = better "
                         "type-aware GP exploration but slower per macro generation.")
parser.add_argument("--micro-generations", type=int, default=3,
                    help="Micro-evolution generations per macro generation (was 1).")
parser.add_argument("--micro-mutation-rate", type=float, default=0.6,
                    help="Per-individual mutation probability (was 0.3).")
args = parser.parse_args()

if not args.verbose:
    warnings.filterwarnings("ignore")
    logging.captureWarnings(False)
    class _QuietModeFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.levelno < logging.WARNING

    _quiet_filter = _QuietModeFilter()
    for _handler in logging.getLogger().handlers:
        _handler.addFilter(_quiet_filter)
    try:
        np.seterr(all="ignore")
    except Exception:
        pass

# ── Output directory ──────────────────────────────────────────────────────
out_dir = Path("results/gnn_training")
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "training_log.jsonl"
top50_log_path = out_dir / "top50_grafts.log"
viz_log_path = out_dir / "graft_visualizations.log"

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


_BEHAVIOR_PROBE_CACHE: dict[tuple[int, int, float, int, int], list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]] = {}


def _nearest_constellation_symbols(x_hat: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    x_hat = np.asarray(x_hat)
    out = np.empty(x_hat.shape[0], dtype=complex)
    for i in range(x_hat.shape[0]):
        dists = np.abs(constellation - x_hat[i]) ** 2
        out[i] = constellation[int(np.argmin(dists))]
    return out


def _build_behavior_probe_samples(
    *,
    nr: int,
    nt: int,
    snr_db: float,
    n_samples: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    key = (nr, nt, float(snr_db), int(n_samples), int(seed))
    cached = _BEHAVIOR_PROBE_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(seed)
    constellation = qam16_constellation()
    probes = [
        generate_mimo_sample(nr, nt, constellation, snr_db, rng) + (constellation.copy(),)
        for _ in range(n_samples)
    ]
    _BEHAVIOR_PROBE_CACHE[key] = probes
    return probes


def _collect_return_slice(ir) -> tuple[set[str], set[str]]:
    """Backward slice from observable sinks.

    Sinks are {return, branch, store, set_item}: a value/op is "live"
    if it can affect any of these. Treating ``branch`` inputs as sinks
    is essential for looping algorithms — otherwise the loop counter,
    convergence test, and entire while-body get classified as dead even
    though they obviously control what the function eventually returns.
    See evolution/gnn_pattern_matcher._compute_return_slice_values
    (kept in sync).
    """
    SINKS = ("return", "branch", "store", "set_item")
    slice_values: set[str] = set()
    slice_ops: set[str] = set()
    pending: list[str] = list(ir.return_values)
    for op in ir.ops.values():
        if op.opcode in SINKS:
            pending.extend(op.inputs)
    while pending:
        vid = pending.pop()
        if vid in slice_values:
            continue
        value = ir.values.get(vid)
        if value is None:
            continue
        slice_values.add(vid)
        def_op_id = value.def_op
        if def_op_id and def_op_id in ir.ops:
            def_op = ir.ops[def_op_id]
            slice_ops.add(def_op_id)
            pending.extend(def_op.inputs)
    return slice_values, slice_ops


def _call_with_timeout(fn, args, timeout: float):
    """Run fn(*args) in a daemon thread with a wall-clock timeout.

    Materialized detector code from evolution can contain non-terminating loops;
    daemon threads ensure abandoned executions do not block process exit.
    """
    out: dict = {}

    def _runner():
        try:
            out["result"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001
            out["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"materialized detector exceeded {timeout:.1f}s")
    if "error" in out:
        raise out["error"]
    return out.get("result")


def _probe_pair_in_subprocess(
    host_fn,
    child_fn,
    probes: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
    *,
    timeout: float,
) -> dict[str, object]:
    """Run behavior probes in a killable subprocess.

    Generated detector code can loop forever or pin CPU.  A thread-only
    timeout returns control to the caller but leaves the worker thread
    running in the trainer process.  This subprocess wrapper contains
    those runaway threads and kills the whole worker on deadline.
    """
    host_source = getattr(host_fn, "_materialized_source", None)
    child_source = getattr(child_fn, "_materialized_source", None)
    host_name = getattr(host_fn, "_materialized_name", getattr(host_fn, "__name__", None))
    child_name = getattr(child_fn, "_materialized_name", getattr(child_fn, "__name__", None))
    if not host_source or not child_source or not host_name or not child_name:
        total_changed = 0
        total_symbols = 0
        n_timeouts = 0
        for H, _x_true, y, sigma2, constellation in probes:
            try:
                host_out = _call_with_timeout(
                    host_fn, (H, y, sigma2, constellation), timeout=timeout,
                )
                child_out = _call_with_timeout(
                    child_fn, (H, y, sigma2, constellation), timeout=timeout,
                )
                host_sym = _nearest_constellation_symbols(np.asarray(host_out), constellation)
                child_sym = _nearest_constellation_symbols(np.asarray(child_out), constellation)
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, TimeoutError):
                    n_timeouts += 1
                return {
                    "ok": False,
                    "exception_repr": f"{type(exc).__name__}: {exc}",
                    "n_timeouts": n_timeouts,
                    "total_changed": 0,
                    "total_symbols": 0,
                }
            if host_sym.shape != child_sym.shape:
                return {
                    "ok": False,
                    "exception_repr": f"shape_mismatch: host={host_sym.shape} child={child_sym.shape}",
                    "n_timeouts": n_timeouts,
                    "total_changed": 0,
                    "total_symbols": 0,
                }
            total_changed += int(np.sum(np.abs(host_sym - child_sym) > 1e-6))
            total_symbols += len(host_sym)
        return {
            "ok": True,
            "exception_repr": None,
            "n_timeouts": n_timeouts,
            "total_changed": total_changed,
            "total_symbols": total_symbols,
        }

    helper = Path(__file__).with_name("evolution") / "behavior_probe_worker.py"
    payload = {
        "host_source": host_source,
        "host_func_name": host_name,
        "child_source": child_source,
        "child_func_name": child_name,
        "probes": probes,
        "timeout": float(timeout),
    }
    # The helper enforces per-call timeouts internally and exits at the
    # first exception.  The outer timeout is a hard wall-clock guard for
    # worker startup/compile failures or native-code stalls.
    outer_timeout = max(5.0, float(timeout) * 2.0 * max(len(probes), 1) + 5.0)
    with tempfile.TemporaryDirectory(prefix="alphadetect_probe_") as td:
        in_path = Path(td) / "probe_in.pkl"
        out_path = Path(td) / "probe_out.pkl"
        with open(in_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            proc = subprocess.run(
                [sys.executable, str(helper), str(in_path), str(out_path)],
                cwd=str(Path(__file__).resolve().parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=outer_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "exception_repr": f"TimeoutError: behavior probe worker exceeded {outer_timeout:.1f}s",
                "n_timeouts": 1,
                "total_changed": 0,
                "total_symbols": 0,
            }
        if proc.returncode != 0 and not out_path.exists():
            return {
                "ok": False,
                "exception_repr": f"RuntimeError: behavior probe worker exited {proc.returncode}",
                "n_timeouts": 0,
                "total_changed": 0,
                "total_symbols": 0,
            }
        if not out_path.exists():
            return {
                "ok": False,
                "exception_repr": "RuntimeError: behavior probe worker produced no result",
                "n_timeouts": 0,
                "total_changed": 0,
                "total_symbols": 0,
            }
        with open(out_path, "rb") as fh:
            return pickle.load(fh)


def _compile_materialized_callable(genome: AlgorithmGenome) -> tuple[Callable | None, str | None, str | None]:
    try:
        source = materialize(genome)
    except Exception as exc:
        return None, None, f"materialize failed: {exc}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        try:
            fn = materialize_to_callable(genome)
        except Exception as exc:
            return None, source, f"compile failed: {exc}"

    syntax_warnings = [
        str(w.message)
        for w in caught
        if issubclass(w.category, SyntaxWarning)
    ]
    if syntax_warnings:
        return None, source, "; ".join(syntax_warnings)
    try:
        setattr(fn, "_materialized_source", source)
        setattr(fn, "_materialized_name", getattr(fn, "__name__", genome.algo_id))
    except Exception:
        pass
    return fn, source, None


def _analyse_effective_graft(
    genome: AlgorithmGenome,
    fit: FitnessResult,
    engine: AlgorithmEvolutionEngine,
    host_fitness_map: dict[str, FitnessResult],
    *,
    host_callable_cache: dict[str, tuple[Callable | None, str | None]] | None = None,
    host_fitness_cache: dict[str, FitnessResult] | None = None,
) -> dict:
    result: dict[str, object] = {
        "is_grafted": bool(genome.graft_history),
        "graft_lineage": "",
        "n_graft_ops": 0,
        "n_graft_ops_in_slice": 0,
        "n_slots": 0,
        "slot_pop_sizes": {},
        "structural_ok": False,
        "behavior_ok": False,
        "performance_ok": False,
        "no_exception": False,
        "effective": False,
        "reasonable": False,
        "behavior_change_rate": 0.0,
        "host_algo_id": None,
        "host_score": None,
        "child_score": fit.composite_score(),
        "quality_verdict": "NOT_GRAFTED",
        "compile_issue": None,
        "source": None,
    }
    _PROFILE = os.environ.get("ALPHADETECT_PROFILE_DONOR") in ("1", "true", "yes")
    _prof_compile_s = 0.0
    _prof_host_compile_s = 0.0
    _prof_probe_s = 0.0
    _prof_n_probes = 0
    _prof_n_timeouts = 0
    if not genome.graft_history:
        if _PROFILE:
            result["_prof_compile_s"] = 0.0
            result["_prof_host_compile_s"] = 0.0
            result["_prof_probe_s"] = 0.0
            result["_prof_n_probes"] = 0
            result["_prof_n_timeouts"] = 0
        return result

    result["graft_lineage"] = " | ".join(
        f"{gr.host_algo_id}<-{gr.donor_algo_id}"
        for gr in genome.graft_history
    )

    ir = genome.structural_ir
    grafted_ops = [op for op in ir.ops.values() if op.attrs.get("grafted")]
    result["n_graft_ops"] = len(grafted_ops)
    result["n_slots"] = len(find_algslot_ops(ir))
    for sid, pop in genome.slot_populations.items():
        result["slot_pop_sizes"][sid] = len(pop.variants)

    slice_values, slice_ops = _collect_return_slice(ir)
    grafted_in_slice = [
        op for op in grafted_ops
        if op.id in slice_ops or any(out_vid in slice_values for out_vid in op.outputs)
    ]
    result["n_graft_ops_in_slice"] = len(grafted_in_slice)
    result["structural_ok"] = len(grafted_in_slice) > 0

    host_algo_id = genome.metadata.get("graft_host_algo_id")
    if host_algo_id is None and genome.graft_history:
        host_algo_id = genome.graft_history[-1].host_algo_id
    result["host_algo_id"] = host_algo_id

    host_genome = engine.resolve_genome(str(host_algo_id)) if host_algo_id else None
    host_score = genome.metadata.get("graft_host_score")
    if host_algo_id in host_fitness_map:
        host_score = host_fitness_map[host_algo_id].composite_score()
    elif host_genome is not None and host_fitness_cache is not None and host_algo_id is not None:
        cached_fit = host_fitness_cache.get(str(host_algo_id))
        if cached_fit is None:
            cached_fit = engine.evaluator.evaluate(host_genome)
            host_fitness_cache[str(host_algo_id)] = cached_fit
        host_score = cached_fit.composite_score()

    result["host_score"] = host_score
    if host_score is not None:
        result["performance_ok"] = fit.composite_score() < (float(host_score) - args.effective_margin)

    # Cheap structural check first; everything else (compile + probes) is
    # paid only for structurally-OK grafts.
    if not result["structural_ok"]:
        result["quality_verdict"] = "STRUCTURAL_FAIL"
        if _PROFILE:
            result["_prof_compile_s"] = _prof_compile_s
            result["_prof_host_compile_s"] = _prof_host_compile_s
            result["_prof_probe_s"] = _prof_probe_s
            result["_prof_n_probes"] = _prof_n_probes
            result["_prof_n_timeouts"] = _prof_n_timeouts
        return result

    # Always run behavior probes for structurally-OK grafts (even if
    # performance_ok is False) so we can distinguish:
    #   - EXCEPTION       : compile or probe call raised
    #   - BEHAVIOR_FAIL   : ran cleanly but produced byte-identical output
    #   - REASONABLE      : behavior changed AND no exceptions, but score <= host
    #   - EFFECTIVE       : reasonable AND beats host by --effective-margin
    raised_exception = False
    if host_genome is not None:
        _t0 = time.perf_counter() if _PROFILE else 0.0
        child_fn, source, child_issue = _compile_materialized_callable(genome)
        if _PROFILE:
            _prof_compile_s = time.perf_counter() - _t0
        if host_callable_cache is not None and str(host_algo_id) in host_callable_cache:
            host_fn, host_issue = host_callable_cache[str(host_algo_id)]
        else:
            _t1 = time.perf_counter() if _PROFILE else 0.0
            host_fn, _, host_issue = _compile_materialized_callable(host_genome)
            if _PROFILE:
                _prof_host_compile_s = time.perf_counter() - _t1
            if host_callable_cache is not None:
                host_callable_cache[str(host_algo_id)] = (host_fn, host_issue)
        result["compile_issue"] = child_issue or host_issue
        result["source"] = source
        if child_fn is None or host_fn is None:
            raised_exception = True
            result["exception_repr"] = result.get("compile_issue") or "compile returned None"
        else:
            probes = _build_behavior_probe_samples(
                nr=args.probe_nr,
                nt=args.probe_nt,
                snr_db=args.effectiveness_snr,
                n_samples=max(args.effectiveness_samples, 6),
                seed=args.seed + 1001,
            )
            _t_probe = time.perf_counter() if _PROFILE else 0.0
            _n_timeouts_local = 0
            if _PROFILE:
                _prof_n_probes += len(probes)
            probe_result = _probe_pair_in_subprocess(
                host_fn,
                child_fn,
                probes,
                timeout=args.probe_timeout,
            )
            total_changed = int(probe_result.get("total_changed", 0) or 0)
            total_symbols = int(probe_result.get("total_symbols", 0) or 0)
            _n_timeouts_local = int(probe_result.get("n_timeouts", 0) or 0)
            if not bool(probe_result.get("ok", False)):
                raised_exception = True
                result["exception_repr"] = str(
                    probe_result.get("exception_repr")
                    or "behavior probe subprocess failed"
                )
                # ── DEBUG: dump the failing source for offline inspection.
                try:
                    import os as _os
                    _dbg_dir = _os.environ.get("ALPHADETECT_GRAFT_EXC_DUMP")
                    if _dbg_dir and source:
                        _os.makedirs(_dbg_dir, exist_ok=True)
                        _safe_id = str(getattr(genome, "algo_id", "anon"))[:32]
                        _exc_name = str(result["exception_repr"]).split(":", 1)[0]
                        _path = _os.path.join(_dbg_dir, f"{_exc_name}_{_safe_id}.py")
                        with open(_path, "w", encoding="utf-8") as _fh:
                            _fh.write(f"# {result['exception_repr']}\n")
                            _fh.write(f"# host={host_algo_id} algo={genome.algo_id}\n\n")
                            _fh.write(source)
                except Exception:
                    pass
            if _PROFILE:
                _prof_probe_s = time.perf_counter() - _t_probe
                _prof_n_timeouts = _n_timeouts_local
            if total_symbols > 0:
                change_rate = total_changed / total_symbols
                result["behavior_change_rate"] = change_rate
                result["behavior_ok"] = change_rate > 0.0
    else:
        result["compile_issue"] = "host genome unresolved"
        raised_exception = True

    result["no_exception"] = not raised_exception
    result["reasonable"] = (
        bool(result["structural_ok"])
        and bool(result["behavior_ok"])
        and bool(result["no_exception"])
    )
    result["effective"] = bool(result["reasonable"]) and bool(result["performance_ok"])

    if _PROFILE:
        result["_prof_compile_s"] = _prof_compile_s
        result["_prof_host_compile_s"] = _prof_host_compile_s
        result["_prof_probe_s"] = _prof_probe_s
        result["_prof_n_probes"] = _prof_n_probes
        result["_prof_n_timeouts"] = _prof_n_timeouts

    if result["effective"]:
        result["quality_verdict"] = "EFFECTIVE"
    elif result["reasonable"]:
        result["quality_verdict"] = "REASONABLE"
    elif raised_exception:
        result["quality_verdict"] = "EXCEPTION"
    else:
        result["quality_verdict"] = "BEHAVIOR_FAIL"
    return result


# ══════════════════════════════════════════════════════════════════════════
# ASCII data-flow visualisation of grafted samples
# ══════════════════════════════════════════════════════════════════════════

def _log_graft_visualizations(
    gen: int,
    graft_samples: list[tuple[AlgorithmGenome, FitnessResult]],
    analyses: dict[str, dict],
    snr_db: float,
    log_file: Path,
    engine: AlgorithmEvolutionEngine,
    *,
    n_per_gen: int,
    color: bool,
) -> int:
    """Render a small sample of host/donor/graft IRs as ASCII dataflow.

    Picks ``n_per_gen`` graft children, preferring those that passed the
    three-fold ``effective`` check, then back-filling from the remaining
    samples sorted by composite score.  For each pick, writes a
    side-by-side dataflow listing of the host IR (with replaced region
    in red), the grafted child IR (with inlined donor ops in green),
    and -- when resolvable -- the donor IR (in cyan).

    Returns the number of grafts actually rendered.
    """
    if n_per_gen <= 0 or not graft_samples:
        return 0

    try:
        from algorithm_ir.visualize import (
            build_visible_ir,
            render_graft_visualization,
            visibility_stats,
        )
    except Exception as exc:
        if args.verbose:
            logger.warning("Graft visualiser unavailable: %s", exc)
        return 0

    # Rank: effective first, then by composite score ascending
    effective = [
        (g, f) for g, f in graft_samples
        if analyses.get(g.algo_id, {}).get("effective")
    ]
    rest = [
        (g, f) for g, f in graft_samples
        if not analyses.get(g.algo_id, {}).get("effective")
    ]
    rest.sort(key=lambda x: x[1].composite_score())
    picks = (effective + rest)[:n_per_gen]

    rendered = 0
    visible_log_file = log_file.with_name(log_file.stem + ".visible.log")
    with open(log_file, "a", encoding="utf-8") as fh, \
            open(visible_log_file, "a", encoding="utf-8") as fh_vis:
        header = (f"\n{'#'*80}\n"
                  f"# GENERATION {gen} | SNR={snr_db:.0f}dB | "
                  f"{len(picks)} graft visualisations "
                  f"({len(effective)} effective available)\n"
                  f"{'#'*80}\n\n")
        fh.write(header)
        fh_vis.write(header)
        fh_vis.write(
            "# NOTE: this file shows the *GNN-visible* IR (trivial ops -- "
            "const, get_attr, assign, trivial-phi -- are removed).\n"
            "# The full IR is in the sibling file 'graft_visualizations.log'.\n\n"
        )

        for rank, (genome, fit) in enumerate(picks, 1):
            analysis = analyses.get(genome.algo_id, {})
            host_algo_id = analysis.get("host_algo_id") or genome.metadata.get("graft_host_algo_id")
            donor_algo_id = genome.metadata.get("graft_donor_algo_id")

            host_genome = engine.resolve_genome(str(host_algo_id)) if host_algo_id else None
            donor_genome = engine.resolve_genome(str(donor_algo_id)) if donor_algo_id else None
            grafted_ir = genome.structural_ir
            host_ir = host_genome.structural_ir if host_genome is not None else None
            donor_ir = donor_genome.structural_ir if donor_genome is not None else None

            # Reconstruct the replaced region: original host op IDs are
            # preserved by graft_general(); donor ops use fresh IDs.
            # So the region == host op IDs missing from the grafted IR.
            region_op_ids: list[str] = []
            if host_ir is not None and grafted_ir is not None:
                grafted_op_ids = set(grafted_ir.ops.keys())
                region_op_ids = [
                    op_id for op_id in host_ir.ops.keys()
                    if op_id not in grafted_op_ids
                ]

            class _Region:
                pass

            region_proxy = _Region()
            region_proxy.op_ids = region_op_ids
            region_proxy.entry_values = []
            region_proxy.exit_values = []

            verdict = analysis.get("quality_verdict", "?")
            ser = fit.metrics.get("ser", float("inf"))
            score = fit.composite_score()
            host_score = analysis.get("host_score")
            host_score_repr = (
                f"{float(host_score):.6f}" if host_score is not None else "n/a"
            )
            change_rate = float(analysis.get("behavior_change_rate", 0.0))

            # Per-IR visibility stats (Plan B): how many ops survive
            # the GNN visibility filter for each IR in this triple.
            def _vis_str(label: str, ir) -> str:
                if ir is None:
                    return f"{label}=n/a"
                tot, vis, hid, pct = visibility_stats(ir)
                return f"{label}=tot{tot}/vis{vis}/hid{hid}({pct:.0f}%)"

            vis_summary = "  ".join([
                _vis_str("host", host_ir),
                _vis_str("donor", donor_ir),
                _vis_str("graft", grafted_ir),
            ])

            header_lines = (
                f"\n{'='*80}\n"
                f"[#{rank}] child={genome.algo_id}  host={host_algo_id}  "
                f"donor={donor_algo_id}\n"
                f"        verdict={verdict}  child_score={score:.6f}  "
                f"host_score={host_score_repr}  child_SER={ser:.6f}  "
                f"behavior_change={change_rate:.3f}  region_size={len(region_op_ids)}\n"
                f"        visibility: {vis_summary}\n"
                f"{'='*80}\n"
            )
            fh.write(header_lines)
            fh_vis.write(header_lines)

            if host_ir is None:
                msg = (f"  <host genome '{host_algo_id}' not resolvable; "
                       f"skipping visualisation>\n")
                fh.write(msg)
                fh_vis.write(msg)
                continue

            # ---- Full IR (original log) ------------------------------
            try:
                viz = render_graft_visualization(
                    host_ir=host_ir,
                    region=region_proxy,
                    donor_ir=donor_ir,
                    grafted_ir=grafted_ir,
                    color=color,
                    show_consumers=True,
                )
            except Exception as exc:
                fh.write(f"  <render failed: {exc}>\n")
            else:
                fh.write(viz)
                fh.write("\n")

            # ---- GNN-visible IR view (parallel log) ------------------
            try:
                host_vis = build_visible_ir(host_ir)
                donor_vis = build_visible_ir(donor_ir) if donor_ir is not None else None
                grafted_vis = build_visible_ir(grafted_ir) if grafted_ir is not None else None
                visible_region_ids = [
                    op_id for op_id in region_op_ids
                    if op_id in host_vis.ops
                ]
                visible_region_proxy = _Region()
                visible_region_proxy.op_ids = visible_region_ids
                visible_region_proxy.entry_values = []
                visible_region_proxy.exit_values = []
                viz_vis = render_graft_visualization(
                    host_ir=host_vis,
                    region=visible_region_proxy,
                    donor_ir=donor_vis,
                    grafted_ir=grafted_vis,
                    color=color,
                    show_consumers=True,
                )
            except Exception as exc:
                fh_vis.write(f"  <visible render failed: {exc}>\n")
            else:
                fh_vis.write(viz_vis)
                fh_vis.write("\n")

            rendered += 1

    return rendered


def _get_host_algo_id(genome: AlgorithmGenome) -> str | None:
    """Extract host algo_id from a grafted genome, or None if not grafted."""
    if not genome.graft_history:
        return None
    host_id = genome.metadata.get("graft_host_algo_id")
    if host_id is not None:
        return host_id
    return genome.graft_history[-1].host_algo_id


def _log_effective_grafts(
    gen: int,
    graft_samples: list[tuple[AlgorithmGenome, FitnessResult]],
    population: list[AlgorithmGenome],
    fitness: list[FitnessResult],
    snr_db: float,
    log_file: Path,
    engine: AlgorithmEvolutionEngine,
    *,
    show_progress: bool = False,
) -> dict:
    evaluated_pairs = list(graft_samples)
    evaluated_pairs.sort(key=lambda x: x[1].composite_score())
    host_fitness_map = {g.algo_id: f for g, f in zip(population, fitness)}

    # ── profiling: graft eval quality-classification phase ──────────
    _PROFILE = os.environ.get("ALPHADETECT_PROFILE_DONOR") in ("1", "true", "yes")
    _t_phase = time.perf_counter() if _PROFILE else 0.0

    # ── Pre-populate host caches (serial, ~1-2s for all hosts) ─────
    host_callable_cache: dict[str, tuple[Callable | None, str | None]] = {}
    host_fitness_cache: dict[str, FitnessResult] = {}
    _t_prep = time.perf_counter() if _PROFILE else 0.0
    for genome, _fit in evaluated_pairs:
        host_algo_id = _get_host_algo_id(genome)
        if host_algo_id and host_algo_id not in host_callable_cache:
            host_genome = engine.resolve_genome(str(host_algo_id))
            if host_genome is not None:
                host_fn, _, host_issue = _compile_materialized_callable(host_genome)
                host_callable_cache[host_algo_id] = (host_fn, host_issue)
                if host_algo_id not in host_fitness_cache:
                    try:
                        cached_fit = engine.evaluator.evaluate(host_genome)
                    except Exception:
                        cached_fit = FitnessResult(float("inf"), False, {})
                    host_fitness_cache[host_algo_id] = cached_fit
    _prof_prep_s = (time.perf_counter() - _t_prep) if _PROFILE else 0.0

    # ── Parallel quality-classification of all grafts ───────────────
    def _analyse_one(args):
        genome, fit = args
        return genome.algo_id, _analyse_effective_graft(
            genome, fit, engine, host_fitness_map,
            host_callable_cache=host_callable_cache,
            host_fitness_cache=host_fitness_cache,
        )

    analyses: dict[str, dict] = {}
    n_workers = min(8, max(1, len(evaluated_pairs)))
    _pairs_for_executor = list(evaluated_pairs)  # snapshot
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_analyse_one, (g, f)): g.algo_id for g, f in _pairs_for_executor}
        if show_progress:
            import tqdm as _tqdm_module
            _iter = _tqdm_module.tqdm(
                as_completed(futures), total=len(futures),
                desc="Classify graft quality", leave=False,
            )
        else:
            _iter = as_completed(futures)
        for future in _iter:
            try:
                algo_id, analysis = future.result()
                analyses[algo_id] = analysis
            except Exception:
                pass  # Should not happen; fall through

    # ── profiling: aggregate stats ──────────────────────────────────
    _prof_compile_s = 0.0
    _prof_probe_s = 0.0
    _prof_host_compile_s = 0.0
    _prof_n_timeouts = 0
    _prof_n_probes_run = 0
    _prof_n_struct_fail = 0
    _prof_per_graft: dict[str, dict] = {}
    _prof_probes_per_graft: list[int] = []
    if _PROFILE:
        for genome, _fit in evaluated_pairs:
            a = analyses.get(genome.algo_id, {})
            verdict = str(a.get("quality_verdict", "?"))
            _prof_per_graft[genome.algo_id] = {
                "t": 0.0, "verdict": verdict,
                "compile": a.get("_prof_compile_s", 0.0),
                "host_compile": a.get("_prof_host_compile_s", 0.0),
                "probes": a.get("_prof_probe_s", 0.0),
                "n_probes": a.get("_prof_n_probes", 0),
                "n_timeouts": a.get("_prof_n_timeouts", 0),
            }
            _prof_compile_s += a.get("_prof_compile_s", 0.0)
            _prof_host_compile_s += a.get("_prof_host_compile_s", 0.0)
            _prof_probe_s += a.get("_prof_probe_s", 0.0)
            _prof_n_timeouts += a.get("_prof_n_timeouts", 0)
            _prof_n_probes_run += a.get("_prof_n_probes", 0)
            _prof_probes_per_graft.append(a.get("_prof_n_probes", 0))
            if verdict == "STRUCTURAL_FAIL":
                _prof_n_struct_fail += 1

    effective_pairs = [
        (g, f) for g, f in evaluated_pairs
        if analyses[g.algo_id]["effective"]
    ]
    reasonable_pairs = [
        (g, f) for g, f in evaluated_pairs
        if analyses[g.algo_id]["reasonable"]
    ]
    structural_fail = sum(1 for g, _ in evaluated_pairs if analyses[g.algo_id]["quality_verdict"] == "STRUCTURAL_FAIL")
    behavior_fail = sum(1 for g, _ in evaluated_pairs if analyses[g.algo_id]["quality_verdict"] == "BEHAVIOR_FAIL")
    exception_fail = sum(1 for g, _ in evaluated_pairs if analyses[g.algo_id]["quality_verdict"] == "EXCEPTION")
    # Performance-fail = reasonable but not effective (behaved differently
    # without exceptions, but did not strictly beat host).
    performance_fail = max(0, len(reasonable_pairs) - len(effective_pairs))

    precise_results: dict[str, FitnessResult] = {}
    if effective_pairs:
        shortlist = effective_pairs[:max(1, min(args.precise_topk, len(effective_pairs)))]
        precise_fits = engine.evaluator.evaluate_precise_batch(
            [g for g, _ in shortlist],
            target_errors=args.precise_target_errors,
            max_symbols=args.precise_max_symbols,
            timeout_sec=args.precise_timeout,
            desc="Precise SER",
        )
        precise_results = {
            genome.algo_id: precise_fit
            for (genome, _), precise_fit in zip(shortlist, precise_fits)
        }

    if _PROFILE:
        _dt_phase = time.perf_counter() - _t_phase
        import numpy as _np
        _t_grafts = [v["t"] for v in _prof_per_graft.values()]
        _t_sorted = sorted(_t_grafts, reverse=True)
        logger.info(
            "GRAFT EVAL QUALITY PROFILE\n"
            "======================================================================\n"
            "  TOTAL phase time:              %8.2fs  (n=%d grafts, workers=%d)\n"
            "  pre-populate host caches:      %8.2fs\n"
            "  compile (child materialize):   %8.2fs\n"
            "  compile (host, cached):        %8.2fs\n"
            "  behavior probes:               %8.2fs  (%d probes run, %d timeouts)\n"
            "  structural_fail (fast path):   %d\n"
            "--- per-graft (all times ms) ---\n"
            "  mean=%6.0f  median=%6.0f  max=%6.0f  min=%6.0f\n"
            "  top-10 slowest grafts (ms): %s\n"
            "  probes-per-graft: mean=%.1f  max=%d  (excluding structural_fail)\n"
            "======================================================================",
            _dt_phase, len(evaluated_pairs), n_workers,
            _prof_prep_s,
            _prof_compile_s,
            _prof_host_compile_s,
            _prof_probe_s, _prof_n_probes_run, _prof_n_timeouts,
            _prof_n_struct_fail,
            _np.mean(_t_grafts) * 1000, _np.median(_t_grafts) * 1000,
            max(_t_grafts) * 1000, min(_t_grafts) * 1000,
            [f"{t*1000:.0f}" for t in _t_sorted[:10]],
            _np.mean(_prof_probes_per_graft) if _prof_probes_per_graft else 0,
            max(_prof_probes_per_graft) if _prof_probes_per_graft else 0,
        )

    effective_ser = [fit.metrics.get("ser", float("inf")) for _, fit in effective_pairs]
    precise_ser = [fit.metrics.get("ser", float("inf")) for fit in precise_results.values()]

    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(f"\n{'='*80}\n")
        fh.write(
            f"GENERATION {gen} | SNR={snr_db:.0f}dB | "
            f"{len(evaluated_pairs)} evaluated graft samples | "
            f"{len(effective_pairs)} effective | "
            f"{len(reasonable_pairs)} reasonable\n"
        )
        fh.write(f"{'='*80}\n\n")
        fh.write(
            "SUMMARY: "
            f"{len(effective_pairs)} effective, "
            f"{len(reasonable_pairs)} reasonable, "
            f"{structural_fail} structural_fail, "
            f"{behavior_fail} behavior_fail, "
            f"{exception_fail} exception, "
            f"{performance_fail} performance_fail (reasonable but not effective)\n"
        )
        # ── Exception type breakdown (when behavior probes raised). ──
        from collections import Counter as _Counter
        exc_counter: _Counter = _Counter()
        exc_examples: dict[str, str] = {}
        for g, _ in evaluated_pairs:
            a = analyses[g.algo_id]
            if a.get("quality_verdict") != "EXCEPTION":
                continue
            repr_ = a.get("exception_repr") or "?:unknown"
            head = repr_.split(":", 1)[0]
            exc_counter[head] += 1
            exc_examples.setdefault(head, repr_)
        if exc_counter:
            fh.write("EXCEPTION BREAKDOWN:\n")
            for k, v in exc_counter.most_common():
                fh.write(f"  {v:4d}  {k:32s}  e.g. {exc_examples[k][:160]}\n")
            top_summary = ", ".join(
                f"{k}={v}" for k, v in exc_counter.most_common(5)
            )
            if args.verbose:
                logger.info("  Graft exceptions (gen %d): %s", gen, top_summary)
                for k, v in exc_counter.most_common(3):
                    logger.info("    e.g. %s: %s", k, exc_examples[k][:200])
        fh.write(
            "EFFECTIVE SER STATS (all evaluated graft samples): "
            f"best={_fmt_metric(min(effective_ser) if effective_ser else None)} | "
            f"median={_fmt_metric(float(np.median(effective_ser)) if effective_ser else None)} | "
            f"mean={_fmt_metric(float(np.mean(effective_ser)) if effective_ser else None)} | "
            f"n={len(effective_ser)}\n"
        )
        fh.write(
            "PRECISE SER STATS (effective shortlist): "
            f"best={_fmt_metric(min(precise_ser) if precise_ser else None)} | "
            f"median={_fmt_metric(float(np.median(precise_ser)) if precise_ser else None)} | "
            f"mean={_fmt_metric(float(np.mean(precise_ser)) if precise_ser else None)} | "
            f"n={len(precise_ser)}\n\n"
        )

        fh.write("ALL EFFECTIVE GRAFT SAMPLES (full source)\n")
        fh.write(f"{'-'*80}\n\n")
        for rank, (genome, fit) in enumerate(effective_pairs, 1):
            analysis = analyses[genome.algo_id]
            ser = fit.metrics.get("ser", float("inf"))
            score = fit.composite_score()
            precise_fit = precise_results.get(genome.algo_id)
            source = analysis.get("source")
            if source is None:
                try:
                    source = materialize(genome)
                except Exception as exc:
                    source = f"<materialize failed: {exc}>"

            fh.write(
                f"--- #{rank} | {genome.algo_id} | SER={ser:.6f} | "
                f"score={score:.6f} | verdict={analysis['quality_verdict']} ---\n"
            )
            fh.write(f"  Lineage: {analysis['graft_lineage']}\n")
            fh.write(
                f"  Effective checks: structural={analysis['structural_ok']} "
                f"behavior={analysis['behavior_ok']} performance={analysis['performance_ok']}\n"
            )
            fh.write(
                f"  Graft ops: {analysis['n_graft_ops']} "
                f"(slice-connected: {analysis['n_graft_ops_in_slice']})\n"
            )
            fh.write(
                f"  Behavior change rate @{args.effectiveness_snr:.0f}dB "
                f"({max(args.effectiveness_samples, 6)} probes, "
                f"nr={args.probe_nr} nt={args.probe_nt} timeout={args.probe_timeout:.1f}s): "
                f"{float(analysis['behavior_change_rate']):.4f}\n"
            )
            fh.write(f"  Host score: {analysis['host_score']} | Child score: {analysis['child_score']}\n")
            if analysis.get("compile_issue"):
                fh.write(f"  Compile issue: {analysis['compile_issue']}\n")
            if precise_fit is not None:
                fh.write(
                    f"  Precise SER: {precise_fit.metrics.get('ser', float('inf')):.6f} "
                    f"(errors={int(precise_fit.metrics.get('symbol_errors', 0))}, "
                    f"symbols={int(precise_fit.metrics.get('symbols_total', 0))})\n"
                )
            fh.write(f"  Slots: {analysis['n_slots']} | Slot pops: {analysis['slot_pop_sizes']}\n")
            fh.write(f"  Parents: {genome.parent_ids}\n")
            fh.write("  Source:\n")
            for line in str(source).split("\n"):
                fh.write(f"    {line}\n")
            fh.write("\n")
        if not effective_pairs:
            fh.write("  <none>\n\n")

        # ── IR visualizations for ALL reasonable grafts ─────────────────
        # Reasonable = structural_ok + behavior_ok + no_exception.
        # Render host/donor/grafted IR side-by-side so every graft that
        # changes behavior can be inspected at the IR level.
        _has_viz = False
        try:
            from algorithm_ir.visualize import (
                build_visible_ir,
                render_graft_visualization,
            )
            _has_viz = True
        except Exception:
            pass

        if _has_viz and reasonable_pairs:
            fh.write("\nALL REASONABLE GRAFT IR VISUALIZATIONS\n")
            fh.write(f"{'-'*80}\n\n")
            for rank, (genome, fit) in enumerate(reasonable_pairs, 1):
                analysis = analyses.get(genome.algo_id, {})
                host_algo_id = analysis.get("host_algo_id") or genome.metadata.get("graft_host_algo_id")
                donor_algo_id = genome.metadata.get("graft_donor_algo_id")
                host_genome = engine.resolve_genome(str(host_algo_id)) if host_algo_id else None
                donor_genome = engine.resolve_genome(str(donor_algo_id)) if donor_algo_id else None
                grafted_ir = genome.structural_ir
                host_ir = host_genome.structural_ir if host_genome is not None else None
                donor_ir = donor_genome.structural_ir if donor_genome is not None else None

                region_op_ids: list[str] = []
                if host_ir is not None and grafted_ir is not None:
                    host_op_ids = set(host_ir.ops.keys())
                    grafted_op_ids = set(grafted_ir.ops.keys())
                    for op_id in host_op_ids - grafted_op_ids:
                        if op_id not in host_op_ids:
                            continue
                        op = host_ir.ops.get(op_id)
                        if op is not None:
                            region_op_ids.append(op_id)

                ser = fit.metrics.get("ser", float("inf"))
                fh.write(
                    f"--- #{rank} REASONABLE | {genome.algo_id} | SER={ser:.6f} | "
                    f"host={host_algo_id} donor={donor_algo_id} "
                    f"graft_ops={analysis.get('n_graft_ops', '?')} "
                    f"change_rate={float(analysis.get('behavior_change_rate', 0)):.4f} ---\n"
                )

                if host_ir is not None and grafted_ir is not None:
                    try:
                        class _Region:
                            pass
                        region_proxy = _Region()
                        region_proxy.op_ids = region_op_ids
                        region_proxy.entry_values = []
                        region_proxy.exit_values = []
                        viz = render_graft_visualization(
                            host_ir=host_ir,
                            region=region_proxy,
                            donor_ir=donor_ir,
                            grafted_ir=grafted_ir,
                            color=False,
                            show_consumers=True,
                        )
                        fh.write(viz)
                        fh.write("\n")
                    except Exception as exc:
                        fh.write(f"  <render failed: {exc}>\n")
                fh.write("\n")

    return ({
        "n_graft_samples": len(evaluated_pairs),
        "n_effective": len(effective_pairs),
        "n_reasonable": len(reasonable_pairs),
        "n_structural_fail": structural_fail,
        "n_behavior_fail": behavior_fail,
        "n_exception": exception_fail,
        "n_performance_fail": performance_fail,
        "effective_best_ser": min(effective_ser) if effective_ser else None,
        "effective_median_ser": float(np.median(effective_ser)) if effective_ser else None,
        "effective_mean_ser": float(np.mean(effective_ser)) if effective_ser else None,
        "effective_precise_best_ser": min(precise_ser) if precise_ser else None,
        "effective_precise_median_ser": float(np.median(precise_ser)) if precise_ser else None,
        "effective_precise_mean_ser": float(np.mean(precise_ser)) if precise_ser else None,
        "n_effective_precise": len(precise_results),
    }, analyses, evaluated_pairs)


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
    subprocess: bool | None = None,
) -> MIMOFitnessEvaluator:
    """Create a MIMO evaluator at the given SNR.

    If ``subprocess`` (or the global ``--subprocess-eval`` flag) is true,
    the returned evaluator runs each genome inside a kill-able worker
    process.  This is required for graft evaluation because
    auto-repaired grafts can compile to code that hangs inside numpy
    indefinitely; without process isolation the leaked OS threads
    saturate the CPU and starve the GNN training loop.
    """
    cfg = MIMOEvalConfig(
        Nr=16, Nt=16, mod_order=16,
        snr_db_list=[snr_db],
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        complexity_weight=0.01,
        batch_workers=batch_workers,
        show_progress=args.progress,
        seed=args.seed,
    )
    use_sp = bool(getattr(args, "subprocess_eval", False)) if subprocess is None else bool(subprocess)
    if use_sp:
        from evolution.subprocess_evaluator import SubprocessMIMOEvaluator
        return SubprocessMIMOEvaluator(cfg)
    return MIMOFitnessEvaluator(cfg)


gnn_matcher = GNNPatternMatcher(
    max_proposals_per_gen=args.proposals,
    top_k_pairs=args.top_k,
    min_region_size=1,
    max_region_size=args.max_region_ops,
    max_boundary_outputs=args.max_boundary_outputs,
    max_cut_values=args.max_cut_values,
    max_region_ops=args.max_region_ops,
    max_region_inputs=args.max_region_inputs,
    max_region_outputs=args.max_region_outputs,
    lr=1e-3,
    buffer_size=args.buffer_size,
    train_interval=args.train_interval,
    train_steps=args.train_steps,
    warmstart_generations=args.warmstart_gens,
    pair_temperature=args.pair_temp,
    pair_exploration=args.pair_eps,
    region_exploration=args.region_eps,
    donor_exploration=args.donor_eps,
    enable_batched_proposals=args.proposal_batch,
    proposal_batch_size=args.proposal_batch_size,
    show_progress=args.progress,
    enable_host_mask=not args.no_host_mask,
    enable_donor_mask=not args.no_donor_mask,
    lambda_rl=args.lambda_rl,
    entropy_coef=args.entropy_coef,
    value_loss_weight=args.value_loss_weight,
    scorer_score_weight=args.scorer_score_weight,
    scorer_reasonable_weight=args.scorer_reasonable_weight,
    scorer_behavior_weight=args.scorer_behavior_weight,
    scorer_perf_weight=args.scorer_perf_weight,
    failed_replay_weight=args.failed_replay_weight,
    training_objective=args.training_objective,
)

if args.resume and Path(args.resume).exists():
    ckpt = torch.load(args.resume, map_location=gnn_matcher.device, weights_only=False)
    gnn_matcher.encoder.load_state_dict(ckpt["encoder"], strict=False)
    gnn_matcher.scorer.load_state_dict(ckpt["scorer"], strict=False)
    if "critic" in ckpt and hasattr(gnn_matcher, "critic"):
        try:
            gnn_matcher.critic.load_state_dict(ckpt["critic"], strict=False)
        except Exception as exc:
            if args.verbose:
                logger.warning("Critic state restore skipped: %s", exc)
    boundary_state = (
        ckpt.get("boundary_region_policy")
        or ckpt.get("region_proposer")
        or ckpt.get("donor_region_selector")
    )
    if boundary_state is not None:
        gnn_matcher.boundary_region_policy.load_state_dict(boundary_state, strict=False)
    if "optimizer" in ckpt:
        try:
            gnn_matcher.optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            if args.verbose:
                logger.warning("Optimizer state restore skipped: %s", exc)
    logger.info("Resumed GNN from checkpoint: %s", args.resume)

evo_config = AlgorithmEvolutionConfig(
    pool_size=args.pool_size,
    n_generations=1,  # we drive the loop ourselves
    seed=args.seed,
    micro_pop_size=args.micro_pop_size,
    micro_generations=args.micro_generations,
    micro_mutation_rate=args.micro_mutation_rate,
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
engine.show_progress = args.progress
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
        "critic": gnn_matcher.critic.state_dict() if hasattr(gnn_matcher, "critic") else {},
        "boundary_region_policy": gnn_matcher.boundary_region_policy.state_dict(),
        "region_proposer": gnn_matcher.region_proposer.state_dict(),
        "donor_region_selector": gnn_matcher.donor_region_selector.state_dict(),
        "optimizer": gnn_matcher.optimizer.state_dict(),
    }, path)
    return path


def _fmt_metric(value: object) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


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
        if args.verbose:
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
    n_grafted_survivors = sum(1 for g in engine.population if "grafted" in g.tags)
    dispatch_case_counts = dict(getattr(engine, "_dispatch_case_counts", {}) or {})
    n_single = dispatch_case_counts.get("single_path", 0)
    n_stale = dispatch_case_counts.get("stale_region", 0)
    n_failed = sum(
        v for k, v in dispatch_case_counts.items()
        if k not in {"single_path", "stale_region",
                     "typed_bind_used", "typed_bind_skipped"}
    )
    n_typed_used = dispatch_case_counts.get("typed_bind_used", 0)
    n_typed_skipped = dispatch_case_counts.get("typed_bind_skipped", 0)
    logger.info(
        "  graft (single-IR): success=%d stale_region=%d failed=%d "
        "typed_bind=%d/%d (cum)",
        n_single, n_stale, n_failed,
        n_typed_used, n_typed_used + n_typed_skipped,
    )

    # ── Slot micro-evolution telemetry ───────────────────────────────
    slot_evo = (engine.last_generation_stats or {}).get("slot_evo") or {}
    if slot_evo:
        bd_count = max(1, slot_evo.get("best_delta_count", 0))
        bd_mean = slot_evo.get("best_delta_sum", 0.0) / bd_count if slot_evo.get("best_delta_count", 0) else 0.0
        logger.info(
            "  slot-evo: attempted=%d validated=%d evaluated=%d improved=%d "
            "(apply_fail=%d val_fail=%d eval_fail=%d noop_beh=%d skip_no_sids=%d skip_no_var=%d) "
            "best_delta_mean=%.4f",
            slot_evo.get("n_attempted", 0),
            slot_evo.get("n_validated", 0),
            slot_evo.get("n_evaluated", 0),
            slot_evo.get("n_improved", 0),
            slot_evo.get("n_apply_failed", 0),
            slot_evo.get("n_validate_failed", 0),
            slot_evo.get("n_eval_failed", 0),
            slot_evo.get("n_noop_behavior", 0),
            slot_evo.get("skipped_no_sids", 0),
            slot_evo.get("skipped_no_variants", 0),
            bd_mean,
        )
        # S6: per-cause failure breakdown — only emit if any non-zero so
        # we don't spam the log when the rejected causes haven't been
        # populated yet (e.g. running with an older engine build).
        cause_fields = (
            ("signature_fail", "n_apply_signature_failed"),
            ("validator_fail", "n_apply_validator_failed"),
            ("codegen_fail", "n_eval_codegen_failed"),
            ("runtime_exc", "n_eval_runtime_exception"),
            ("timeout", "n_eval_timeout"),
            ("shape_err", "n_eval_shape_error"),
            ("ser_bad", "n_eval_ser_bad"),
        )
        cause_total = sum(int(slot_evo.get(field, 0)) for _, field in cause_fields)
        if cause_total > 0:
            cause_str = " ".join(f"{label}={int(slot_evo.get(field, 0))}"
                                 for label, field in cause_fields)
            logger.info("  slot-evo causes: %s", cause_str)

    if best_ser < best_ser_ever:
        best_ser_ever = best_ser

    elapsed = time.perf_counter() - t0

    # ── Log top-50 grafted individuals ─────────────────────────────────
    graft_stats, graft_analyses, graft_evaluated_pairs = _log_effective_grafts(
        gen, engine.last_graft_sample_pool, engine.population, engine.fitness,
        current_snr, top50_log_path, engine,
        show_progress=args.progress,
    )

    # ── Stage-6: backfill rich per-proposal outcomes into the GNN ──
    try:
        analyses_by_pid: dict[str, dict] = {}
        for genome, _fit in graft_evaluated_pairs:
            analysis = graft_analyses.get(genome.algo_id)
            if not analysis:
                continue
            # Only attribute this child's analysis to the graft event
            # that actually produced it in the CURRENT generation —
            # otherwise ancestor proposals get their reward overwritten
            # by descendant outcomes (label pollution).
            current_gen_grafts = [
                gr for gr in genome.graft_history
                if getattr(gr, "generation", None) == gen
            ]
            target_gr = current_gen_grafts[-1] if current_gen_grafts else None
            if target_gr is None:
                continue
            pid = getattr(target_gr, "proposal_id", None)
            if pid:
                analyses_by_pid[pid] = analysis
        if analyses_by_pid and hasattr(gnn_matcher, "backfill_outcomes"):
            n_back = gnn_matcher.backfill_outcomes(analyses_by_pid)
            if n_back:
                logger.info("  Backfilled %d outcomes with rich reward.", n_back)
    except Exception as exc:
        if args.verbose:
            logger.warning("backfill_outcomes failed at gen %d: %s", gen, exc)

    # ── Render ASCII data-flow visualisations of selected grafts ──────
    if args.viz_grafts_per_gen > 0:
        try:
            n_viz = _log_graft_visualizations(
                gen,
                graft_evaluated_pairs,
                graft_analyses,
                current_snr,
                viz_log_path,
                engine,
                n_per_gen=args.viz_grafts_per_gen,
                color=args.viz_color,
            )
            if n_viz:
                logger.info("  Wrote %d graft visualisation(s) to %s", n_viz, viz_log_path)
        except Exception as exc:
            if args.verbose:
                logger.warning("Graft visualisation failed at gen %d: %s", gen, exc)

    # ── SNR curriculum: escalate when EITHER (a) GNN learned meaningful
    # grafts at this SNR, OR (b) the population has converged so the
    # current SNR offers no headroom for improvement.  Convergence is
    # detected when the median equals the best for several recent gens
    # — this means every survivor is a clone of the best, so any
    # improvement signal is impossible to extract at this SNR.
    snr_changed = False
    gens_at_current_snr = sum(1 for r in phase_history if r.get("snr_db") == current_snr)
    n_effective = graft_stats.get("n_effective", 0)
    frac_effective_grafts = n_effective / max(graft_stats.get("n_graft_samples", 0), 1) if graft_stats.get("n_graft_samples", 0) > 0 else 0.0

    # Convergence: median == best across the last few gens at current SNR.
    recent_at_snr = [r for r in phase_history if r.get("snr_db") == current_snr]
    converged = (
        len(recent_at_snr) >= 3
        and all(
            abs((r.get("median_ser") or 0.0) - (r.get("best_ser") or 0.0)) < 1e-6
            for r in recent_at_snr[-3:]
        )
    )

    should_escalate = (
        current_snr < args.snr_target
        and (
            (gens_at_current_snr >= 20 and frac_effective_grafts >= args.phase_thresh)
            or (gens_at_current_snr >= 5 and converged)
        )
    )
    if should_escalate:
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
            "*** SNR escalation: %.0fdB → %.0fdB (%d gens at %.0fdB, %.0f%% effective) ***",
            old_snr, current_snr, gens_at_current_snr, old_snr, frac_effective_grafts * 100,
        )

    # ── Logging ────────────────────────────────────────────────────────
    record = {
        "gen": gen,
        "snr_db": current_snr,
        "best_ser": best_ser,
        "median_ser": median_ser,
        "best_ser_ever": best_ser_ever,
        "frac_good": frac_good,
        "frac_effective_grafts": round(frac_effective_grafts, 3),
        "gens_at_snr": gens_at_current_snr,
        "n_grafted_survivors": n_grafted_survivors,
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
        "good=%.0f%% | graft_samples=%d effective=%d reasonable=%d (structural_fail=%d, behavior_fail=%d, exception=%d, perf_fail=%d) | survivors=%d | %.1fs",
        gen, current_snr, best_ser, median_ser,
        frac_good * 100, graft_stats.get("n_graft_samples", 0),
        graft_stats.get("n_effective", 0),
        graft_stats.get("n_reasonable", 0),
        graft_stats.get("n_structural_fail", 0),
        graft_stats.get("n_behavior_fail", 0),
        graft_stats.get("n_exception", 0),
        graft_stats.get("n_performance_fail", 0),
        n_grafted_survivors,
        elapsed,
    )
    logger.info(
        "  Effective graft SER (all samples): rough_best=%s rough_median=%s rough_mean=%s | precise_best=%s precise_median=%s precise_mean=%s (n=%s)",
        _fmt_metric(graft_stats.get("effective_best_ser")),
        _fmt_metric(graft_stats.get("effective_median_ser")),
        _fmt_metric(graft_stats.get("effective_mean_ser")),
        _fmt_metric(graft_stats.get("effective_precise_best_ser")),
        _fmt_metric(graft_stats.get("effective_precise_median_ser")),
        _fmt_metric(graft_stats.get("effective_precise_mean_ser")),
        graft_stats.get("n_effective_precise", 0),
    )
    matcher_stats = gnn_matcher.get_stats()
    last_proposals = matcher_stats.get("last_proposals", {})
    logger.info(
        "  Train stats: matched=%s score=%.4f baseline=%.4f | proposals=%s pairs=%s/%s | region_ops=%.2f inputs=%.2f outputs=%.2f cut=%.2f invalid=%.3f | graft_eval=%s kept=%s warmstart=%s",
        matcher_stats.get("last_train", {}).get("matched_samples"),
        matcher_stats.get("last_train", {}).get("mean_graft_score", float("nan")),
        matcher_stats.get("reward_baseline", 0.0),
        last_proposals.get("proposals_built"),
        last_proposals.get("pair_candidates_selected"),
        last_proposals.get("pair_candidates_scored"),
        last_proposals.get("mean_region_ops", 0.0),
        last_proposals.get("mean_region_inputs", 0.0),
        last_proposals.get("mean_region_outputs", 0.0),
        last_proposals.get("effective_cut_size", 0.0),
        last_proposals.get("invalid_region_rate", 0.0),
        engine.last_generation_stats.get("graft_evaluated"),
        engine.last_generation_stats.get("graft_kept"),
        engine.last_generation_stats.get("warmstart_active"),
    )
    logger.info(
        "  Host region: mask=%s attempts=%s built=%s validate_passes=%s validity_rate=%.3f | invalid_buckets=%s",
        last_proposals.get("host_use_mask", False),
        last_proposals.get("host_sampler_attempts", 0),
        last_proposals.get("host_built_regions", 0),
        last_proposals.get("host_validate_passes", 0),
        last_proposals.get("host_validity_rate", 0.0),
        last_proposals.get("invalid_regions", {}),
    )
    logger.info(
        "  Donor region: mask=%s attempts=%s built=%s validate_passes=%s validity_rate=%.3f end_to_end=%.3f",
        last_proposals.get("donor_use_mask", False),
        last_proposals.get("donor_sampler_attempts", 0),
        last_proposals.get("donor_built_regions", 0),
        last_proposals.get("donor_validate_passes", 0),
        last_proposals.get("donor_validity_rate", 0.0),
        last_proposals.get("end_to_end_validity_rate", 0.0),
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
        if args.verbose:
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
        if args.verbose:
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
