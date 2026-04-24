"""Slot-population micro-evolution on the canonical flat IR.

Phase H+3 — replaces the legacy ``materialize_with_override`` path that
operated on AlgSlot ops (which no longer exist after the Phase G
single-IR refactor) with a graft-based path that uses
``_provenance.from_slot_id`` annotations to locate slot regions in the
flat IR and ``graft_general`` to splice in candidate variant bodies.

Pipeline for ONE slot variant evaluation::

    genome.ir (flat, validated, annotated)
        │
        ├── map_pop_key_to_from_slot_ids(genome, "lmmse.regularizer")
        │   → {"_slot_regularizer_op_24"}                            (ann lookup)
        │
        ├── collect_slot_region(genome.ir, slot_anno_ids)
        │   → RewriteRegion R                                        (region build)
        │
        ├── apply_slot_variant(genome, "lmmse.regularizer", variant_ir)
        │     ├── GraftProposal(donor_ir=variant_ir, region=R, contract=None)
        │     ├── graft_general(genome.ir, proposal)
        │     │       (uses Phase H+1 typed binder for arg alignment)
        │     └── validate_function_ir(artifact.ir)                  (Phase H+2 gate)
        │   → flat_ir  OR  None on rejection
        │
        ├── emit_python_source(flat_ir)
        │   → source string
        │
        └── evaluator.evaluate_source_quick(source, fn_name, ...)
            → SER scalar

The single-IR invariant is preserved: ``genome.ir`` is the only
canonical IR, and slot variants are only spliced in transiently for
evaluation. ``pop.best_idx`` records which variant won; the variant
itself is stored in ``pop.variants[best_idx]`` and re-spliced on demand.
"""
from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.ir.validator import validate_function_ir
from algorithm_ir.region.selector import RewriteRegion
from algorithm_ir.regeneration.codegen import emit_python_source

if TYPE_CHECKING:
    from evolution.pool_types import AlgorithmGenome, SlotPopulation

logger = logging.getLogger(__name__)

__all__ = [
    "map_pop_key_to_from_slot_ids",
    "collect_slot_region",
    "apply_slot_variant",
    "evaluate_slot_variant",
    "perturb_constants_in_ir",
    "step_slot_population",
    "SlotMicroStats",
]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class SlotMicroStats:
    """Telemetry for one slot micro-generation step."""

    slot_pop_key: str
    n_attempted: int = 0          # children proposed (mutations + crossovers)
    n_validated: int = 0          # passed validate_function_ir post-graft
    n_evaluated: int = 0          # actually got a finite SER
    n_improved: int = 0           # SER < parent SER
    best_before: float = float("inf")
    best_after: float = float("inf")
    n_apply_failed: int = 0       # graft_general failed / returned None
    n_validate_failed: int = 0    # validation rejected the spliced IR
    n_eval_failed: int = 0        # subprocess returned 1.0 / inf

    def as_dict(self) -> dict:
        return {
            "slot_pop_key": self.slot_pop_key,
            "n_attempted": self.n_attempted,
            "n_validated": self.n_validated,
            "n_evaluated": self.n_evaluated,
            "n_improved": self.n_improved,
            "best_before": self.best_before,
            "best_after": self.best_after,
            "n_apply_failed": self.n_apply_failed,
            "n_validate_failed": self.n_validate_failed,
            "n_eval_failed": self.n_eval_failed,
            "best_delta": self.best_after - self.best_before,
        }


# ---------------------------------------------------------------------------
# M1: slot region discovery
# ---------------------------------------------------------------------------

def map_pop_key_to_from_slot_ids(genome: "AlgorithmGenome",
                                 pop_key: str) -> set[str]:
    """Return the set of distinct ``_provenance.from_slot_id`` annotations
    in ``genome.ir`` that correspond to the slot population keyed by
    ``pop_key``.

    The annotation format produced by FII inlining is
    ``_slot_<short_name>_<call_op_id>`` — one per call site of the slot
    helper. The pop key format is ``<algo>.<short_name>``. The short name
    is the last dotted segment of pop_key; we accept any annotation that
    starts with ``_slot_<short>_`` or equals ``short`` exactly.
    """
    short = pop_key.split(".")[-1]
    candidates: set[str] = set()
    for op in genome.ir.ops.values():
        prov = op.attrs.get("_provenance") if op.attrs else None
        if not isinstance(prov, dict):
            continue
        sid = prov.get("from_slot_id")
        if not isinstance(sid, str):
            continue
        if sid.startswith(f"_slot_{short}_") or sid == short:
            candidates.add(sid)
    return candidates


def collect_slot_region(ir: FunctionIR,
                        from_slot_ids: set[str]) -> RewriteRegion | None:
    """Build a ``RewriteRegion`` from FII-tagged slot ops.

    Two boundary adjustments distinguish this from a naive
    "all-tagged-ops" region:

    1. **Drop input-snapshot assigns.** FII inlining tags ops like
       ``__fii_lmmse_G_1 = G`` (an ``assign`` op whose only input is a
       value defined OUTSIDE the tagged set, i.e. one of the slot
       helper's call-site arguments). Removing such an assign would
       leave its output value (consumed by downstream loop phis as a
       carry-in) dangling.  We therefore exclude these snapshot
       assigns from the region; their outputs become the slot's actual
       entry values for the donor body.

    2. **Absorb trailing output assigns.** FII emits a final
       ``output_var = <slot_internal_value>`` assign that is **not**
       tagged with ``from_slot_id`` (because the inliner attributes it
       to the call-site rather than the helper body). Without absorbing
       it, the assign's input becomes orphaned after the tagged ops are
       deleted. We absorb any ``assign`` op in the same block whose
       only input is region-defined.
    """
    region_ops: list[str] = []
    for oid, op in ir.ops.items():
        prov = op.attrs.get("_provenance") if op.attrs else None
        if not isinstance(prov, dict):
            continue
        sid = prov.get("from_slot_id")
        if isinstance(sid, str) and sid in from_slot_ids:
            region_ops.append(oid)
    if not region_ops:
        return None

    # Pass 1: drop tagged assigns whose inputs are all from outside the
    # tagged set — these are input-snapshot assigns whose outputs feed
    # downstream loop phis. Removing them would invalidate those phis.
    tagged_set = set(region_ops)
    tagged_defined: set[str] = set()
    for oid in region_ops:
        tagged_defined.update(ir.ops[oid].outputs)
    filtered: list[str] = []
    for oid in region_ops:
        op = ir.ops[oid]
        if op.opcode == "assign" and op.inputs and not any(
            v in tagged_defined for v in op.inputs
        ):
            # Input-snapshot assign — leave it in the host IR.
            continue
        filtered.append(oid)
    region_ops = filtered
    if not region_ops:
        return None

    region_set = set(region_ops)
    region_block_ids = {ir.ops[oid].block_id for oid in region_ops if oid in ir.ops}
    region_defined: set[str] = set()
    for oid in region_ops:
        region_defined.update(ir.ops[oid].outputs)

    # Pass 2: trailing-assign closure. Absorb un-tagged assigns whose
    # only input is region-defined (the FII-emitted output assign).
    PASSTHROUGH_OPCODES = {"assign"}
    changed = True
    while changed:
        changed = False
        for oid, op in ir.ops.items():
            if oid in region_set:
                continue
            if op.opcode not in PASSTHROUGH_OPCODES:
                continue
            if op.block_id not in region_block_ids:
                continue
            if not op.inputs:
                continue
            if not all(v in region_defined for v in op.inputs):
                continue
            region_set.add(oid)
            region_ops.append(oid)
            region_defined.update(op.outputs)
            changed = True

    region_used: set[str] = set()
    for oid in region_ops:
        op = ir.ops[oid]
        region_used.update(op.inputs)
    entry_values = sorted(v for v in region_used if v not in region_defined)
    exit_values: list[str] = []
    for vid in sorted(region_defined):
        val = ir.values.get(vid)
        if val is None:
            continue
        for use in val.use_ops:
            if use not in region_set:
                exit_values.append(vid)
                break
    block_ids = sorted(region_block_ids)
    return RewriteRegion(
        region_id=f"slot_region_{uuid.uuid4().hex[:6]}",
        op_ids=region_ops,
        block_ids=block_ids,
        entry_values=entry_values,
        exit_values=exit_values,
        read_set=entry_values,
        write_set=exit_values,
        state_carriers=[],
        schedule_anchors={},
        allows_new_state=False,
    )


def pick_real_exit_value(ir: FunctionIR,
                         region: RewriteRegion) -> str | None:
    """Heuristic to identify the slot region's "true" output value.

    Strategy:

    - If only one exit value, return it.
    - Else among exit values, prefer the one whose def op has the
      highest position in its block (the slot helper's last
      computation typically writes the result).
    - Tie-break: prefer the exit value with the most non-phi outside
      uses (the "real" output is consumed by computational ops, not
      just loop-carry phis).
    """
    if not region.exit_values:
        return None
    if len(region.exit_values) == 1:
        return region.exit_values[0]
    region_set = set(region.op_ids)

    def block_pos(vid: str) -> int:
        val = ir.values.get(vid)
        if val is None or val.def_op is None:
            return -1
        op = ir.ops.get(val.def_op)
        if op is None:
            return -1
        block = ir.blocks.get(op.block_id)
        if block is None or val.def_op not in block.op_ids:
            return -1
        return block.op_ids.index(val.def_op)

    def non_phi_outside_uses(vid: str) -> int:
        val = ir.values.get(vid)
        if val is None:
            return 0
        n = 0
        for use_oid in val.use_ops:
            if use_oid in region_set:
                continue
            use_op = ir.ops.get(use_oid)
            if use_op is None:
                continue
            if use_op.opcode == "phi":
                continue
            n += 1
        return n

    ranked = sorted(
        region.exit_values,
        key=lambda vid: (non_phi_outside_uses(vid), block_pos(vid)),
        reverse=True,
    )
    return ranked[0]


# ---------------------------------------------------------------------------
# M2: variant application via graft_general
# ---------------------------------------------------------------------------

def apply_slot_variant(genome: "AlgorithmGenome",
                       pop_key: str,
                       variant_ir: FunctionIR) -> FunctionIR | None:
    """Splice ``variant_ir`` into the slot region identified by ``pop_key``
    in ``genome.ir`` and return the resulting flat IR.

    Returns ``None`` on any failure (region missing, graft failed,
    validation failed). Never mutates ``genome``.
    """
    if variant_ir is None:
        return None

    sids = map_pop_key_to_from_slot_ids(genome, pop_key)
    if not sids:
        return None

    region = collect_slot_region(genome.ir, sids)
    if region is None:
        return None

    # Identify the SINGLE "real" exit value of the slot. The donor
    # variant has exactly one return value; we want it bound to the
    # one host value that downstream computational ops actually
    # consume. Without this pinning, graft_general's positional
    # bind_donor_returns_to_host_exits would map the donor's return
    # to whichever exit value sorts first alphabetically, leaving the
    # actual downstream consumer reading from a freshly-removed op.
    real_exit = pick_real_exit_value(genome.ir, region)
    if real_exit is None:
        return None

    # Build a minimal BoundaryContract that pins exit_values to
    # [real_exit] only. Empty port_signature lists make the strict
    # signature compatibility check trivially succeed (vacuous all()),
    # so graft_general will use these normalized ports as positional
    # binding targets — which is exactly what we want for slots whose
    # arg names match the helper signature by construction.
    from algorithm_ir.region.contract import BoundaryContract
    contract = BoundaryContract(
        contract_id=f"slot_contract_{pop_key}_{uuid.uuid4().hex[:6]}",
        region_id=region.region_id,
        input_ports=list(region.entry_values),
        output_ports=[real_exit],
        normalized_input_ports=list(region.entry_values),
        normalized_output_ports=[real_exit],
        port_signature={"inputs": [], "outputs": []},
        port_order_evidence={},
        readable_slots=[],
        writable_slots=[],
        new_state_policy={"allowed": False, "state_carriers": []},
        reconnect_points={},
        invariants={},
    )

    # Lazy imports to avoid cycles.
    from algorithm_ir.grafting.graft_general import graft_general
    from evolution.pool_types import GraftProposal

    proposal = GraftProposal(
        proposal_id=f"slot_apply_{pop_key}_{uuid.uuid4().hex[:6]}",
        host_algo_id=genome.algo_id,
        region=region,
        contract=contract,
        donor_algo_id=None,
        donor_ir=variant_ir,
        donor_region=None,
        dependency_overrides=[],
    )
    try:
        artifact = graft_general(genome.ir, proposal)
    except Exception as exc:
        logger.debug("apply_slot_variant: graft_general raised: %r", exc)
        return None
    if artifact is None or getattr(artifact, "ir", None) is None:
        return None

    errs = validate_function_ir(artifact.ir)
    if errs:
        logger.debug("apply_slot_variant: post-graft validation rejected "
                     "%s (%d errs): %s", pop_key, len(errs), errs[:2])
        return None
    return artifact.ir


# ---------------------------------------------------------------------------
# M2': end-to-end evaluation
# ---------------------------------------------------------------------------

def _func_name_from_ir(ir: FunctionIR) -> str:
    name = getattr(ir, "name", None) or "detector"
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(name))
    if not safe or safe[0].isdigit():
        safe = "_" + safe
    return safe


def _source_compiles_with_resolved_names(source: str) -> bool:
    """AST-level sanity gate.

    Parses ``source`` and verifies, per top-level function, that every
    Name read inside the body has a binding before its first use
    (function arg, assignment LHS, augmented assignment LHS, for-target,
    or import). Globals (``np``, ``scipy``, builtins, etc.) are accepted
    via a permissive allow-list. This catches the common splice failure
    where the slot region drops the loop pre-header and downstream loop
    code references variables that no longer exist.
    """
    import ast
    import builtins as _builtins
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    GLOBALS_OK = set(dir(_builtins)) | {
        "np", "numpy", "scipy", "sp", "math", "complex", "float",
        "int", "bool", "list", "dict", "tuple", "set", "range",
        "len", "abs", "min", "max", "sum", "any", "all", "zip",
        "enumerate", "map", "filter", "sorted", "reversed",
        "isinstance", "type", "print",
    }
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        bound: set[str] = set(GLOBALS_OK)
        for a in fn.args.args:
            bound.add(a.arg)
        for a in fn.args.kwonlyargs:
            bound.add(a.arg)
        if fn.args.vararg:
            bound.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            bound.add(fn.args.kwarg.arg)

        # Pre-scan ALL targets in the function (including inside loops
        # and conditionals) so that order-independent assignments are
        # treated as bound. We only flag names that are NEVER assigned
        # anywhere in the function body.
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    for n in ast.walk(tgt):
                        if isinstance(n, ast.Name):
                            bound.add(n.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    bound.add(node.target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    bound.add(node.target.id)
            elif isinstance(node, ast.For):
                for n in ast.walk(node.target):
                    if isinstance(n, ast.Name):
                        bound.add(n.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add(alias.asname or alias.name.split(".")[0])

        # Now scan all Loaded names; reject if any is not bound.
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in bound:
                    return False
    return True





def evaluate_slot_variant(genome: "AlgorithmGenome",
                          pop_key: str,
                          variant_ir: FunctionIR,
                          *,
                          evaluator: Any,
                          n_trials: int = 8,
                          timeout_sec: float = 1.0,
                          snr_db: float = 16.0) -> tuple[float, str | None]:
    """Splice the variant in, materialize, evaluate via subprocess.

    Returns ``(ser, source_or_None)``. ``ser`` is in [0, 1]; ``inf`` on
    any failure. ``source`` is the materialized Python source (returned
    so the caller can log / inspect winners); ``None`` if any earlier
    step failed.
    """
    flat_ir = apply_slot_variant(genome, pop_key, variant_ir)
    if flat_ir is None:
        return float("inf"), None
    try:
        source = emit_python_source(flat_ir)
    except Exception as exc:
        logger.debug("evaluate_slot_variant: codegen failed: %r", exc)
        return float("inf"), None
    if not _source_compiles_with_resolved_names(source):
        # The IR validator sometimes accepts artifacts whose generated
        # source contains undefined names (e.g. when a slot region
        # contains loop-internal definitions that the trailing-assign
        # closure can't capture). Reject these statically here so we
        # don't waste a subprocess call.
        logger.debug("evaluate_slot_variant: source has undefined names "
                     "(%s)", pop_key)
        return float("inf"), source
    fn_name = _func_name_from_ir(flat_ir)

    eval_quick = getattr(evaluator, "evaluate_source_quick", None)
    if eval_quick is None:
        # In-process fallback: build a transient genome with the spliced
        # IR and call evaluator.evaluate(). This path supports tests
        # using plain MIMOFitnessEvaluator.
        try:
            from copy import copy as _shallow_copy
            transient = _shallow_copy(genome)
            transient.ir = flat_ir
            res = evaluator.evaluate(transient)
            ser = float(getattr(res, "metrics", {}).get("ser", float("inf")))
            if not np.isfinite(ser):
                return float("inf"), source
            return ser, source
        except Exception as exc:
            logger.debug("evaluate_slot_variant: in-proc fallback failed: %r", exc)
            return float("inf"), source
    try:
        ser = float(eval_quick(
            source, fn_name,
            algo_id=f"{genome.algo_id}_slot_{pop_key}",
            n_trials=int(n_trials),
            timeout_sec=float(timeout_sec),
            snr_db=float(snr_db),
        ))
    except Exception as exc:
        logger.debug("evaluate_slot_variant: subprocess eval failed: %r", exc)
        return float("inf"), source
    if not np.isfinite(ser):
        return float("inf"), source
    return ser, source


# ---------------------------------------------------------------------------
# M3: typed mutation (constant perturbation only for first cut)
# ---------------------------------------------------------------------------

def perturb_constants_in_ir(ir: FunctionIR,
                            rng: np.random.Generator,
                            *,
                            scale: float = 0.1,
                            min_const: float = 1e-6,
                            prob: float = 0.5) -> FunctionIR:
    """Return a deep copy of ``ir`` with numeric ``const`` op literals
    perturbed by Gaussian multiplicative noise.

    Each numeric const is independently perturbed with probability
    ``prob``; the new value is ``v * (1 + rng.normal(0, scale))``.
    Boolean / int / non-numeric / None literals are left untouched.

    Always validates before returning. Returns the (deep-copied)
    original ``ir`` on validation failure.
    """
    new_ir = deepcopy(ir)
    n_perturbed = 0
    for op in new_ir.ops.values():
        if op.opcode != "const":
            continue
        lit = op.attrs.get("literal", None) if op.attrs else None
        if lit is None or isinstance(lit, bool):
            continue
        if not isinstance(lit, (int, float)):
            continue
        if rng.random() >= prob:
            continue
        # Don't perturb integer indices / counters that look like ints.
        if isinstance(lit, int) and not isinstance(lit, bool):
            # Integers used as iteration counts / array sizes — leave alone.
            continue
        noise = float(rng.normal(0.0, scale))
        new_val = float(lit) * (1.0 + noise)
        if not np.isfinite(new_val):
            continue
        if abs(new_val) < min_const and lit != 0:
            new_val = float(np.sign(new_val) or 1.0) * min_const
        op.attrs["literal"] = new_val
        n_perturbed += 1
    errs = validate_function_ir(new_ir)
    if errs:
        logger.debug("perturb_constants: validation rejected "
                     "(%d errs); returning input copy", len(errs))
        return deepcopy(ir)
    return new_ir


# ---------------------------------------------------------------------------
# M4: one micro-generation step on a single slot population
# ---------------------------------------------------------------------------

def step_slot_population(genome: "AlgorithmGenome",
                         pop_key: str,
                         pop: "SlotPopulation",
                         *,
                         evaluator: Any,
                         rng: np.random.Generator,
                         n_children: int = 4,
                         n_trials: int = 8,
                         timeout_sec: float = 1.0,
                         snr_db: float = 16.0,
                         max_pop_size: int = 16,
                         perturb_scale: float = 0.1) -> SlotMicroStats:
    """Run one (μ+λ) micro-generation on ``pop``.

    Steps:
      1. Ensure baseline fitness for the current best variant exists.
      2. Generate ``n_children`` mutated children by perturbing constants
         of randomly picked parents.
      3. Splice + materialize + evaluate each child via subprocess.
      4. Append survivors to ``pop.variants`` / ``pop.fitness``.
      5. Recompute ``pop.best_idx``; truncate to ``max_pop_size``.

    Mutates ``pop`` in place. Returns ``SlotMicroStats``.
    """
    stats = SlotMicroStats(slot_pop_key=pop_key)

    # Skip if no variants at all.
    if not pop.variants:
        return stats

    # Skip if the genome has no annotations matching this pop_key.
    sids = map_pop_key_to_from_slot_ids(genome, pop_key)
    if not sids:
        return stats

    # Bootstrap baseline fitness for the current best.
    bi = pop.best_idx
    if bi >= len(pop.fitness):
        pop.fitness = list(pop.fitness) + [float("inf")] * (
            len(pop.variants) - len(pop.fitness)
        )
    if not np.isfinite(pop.fitness[bi]):
        baseline_ser, _ = evaluate_slot_variant(
            genome, pop_key, pop.variants[bi],
            evaluator=evaluator,
            n_trials=n_trials, timeout_sec=timeout_sec, snr_db=snr_db,
        )
        pop.fitness[bi] = baseline_ser
    stats.best_before = pop.fitness[bi]

    # Generate children via constant perturbation.
    n_parents = len(pop.variants)
    for _ in range(n_children):
        stats.n_attempted += 1
        # Pick parent biased toward better fitness.
        finite = [(i, f) for i, f in enumerate(pop.fitness) if np.isfinite(f)]
        if finite:
            # Tournament of 2.
            i1, i2 = rng.integers(0, len(finite), size=2)
            cand = finite[i1] if finite[i1][1] <= finite[i2][1] else finite[i2]
            parent_idx = cand[0]
        else:
            parent_idx = int(rng.integers(0, n_parents))
        parent = pop.variants[parent_idx]
        child = perturb_constants_in_ir(parent, rng, scale=perturb_scale)
        # Quick sanity: variant arg signature must match parent.
        if list(child.arg_values) != list(parent.arg_values):
            continue
        # Splice + validate.
        flat_ir = apply_slot_variant(genome, pop_key, child)
        if flat_ir is None:
            stats.n_apply_failed += 1
            continue
        # The validation already happens inside apply_slot_variant,
        # so reaching here means validation passed.
        stats.n_validated += 1
        # Evaluate.
        try:
            source = emit_python_source(flat_ir)
        except Exception:
            stats.n_eval_failed += 1
            continue
        eval_quick = getattr(evaluator, "evaluate_source_quick", None)
        if eval_quick is None:
            stats.n_eval_failed += 1
            continue
        try:
            ser = float(eval_quick(
                source, _func_name_from_ir(flat_ir),
                algo_id=f"{genome.algo_id}_slot_{pop_key}",
                n_trials=int(n_trials),
                timeout_sec=float(timeout_sec),
                snr_db=float(snr_db),
            ))
        except Exception:
            stats.n_eval_failed += 1
            continue
        if not np.isfinite(ser) or ser >= 1.0:
            stats.n_eval_failed += 1
            # Still record so we don't keep regenerating the same garbage.
            pop.variants.append(child)
            pop.fitness.append(float("inf"))
            continue
        stats.n_evaluated += 1
        pop.variants.append(child)
        pop.fitness.append(ser)
        if ser < stats.best_before - 1e-9:
            stats.n_improved += 1
        if pop.source_variants is not None:
            pop.source_variants.append(None)

    # Recompute best.
    if pop.fitness:
        best_i, best_f = 0, pop.fitness[0]
        for i, f in enumerate(pop.fitness):
            if f < best_f:
                best_f = f
                best_i = i
        pop.best_idx = best_i
    stats.best_after = pop.fitness[pop.best_idx] if pop.fitness else float("inf")

    # Truncate (keep the first variant — typically the default — plus the
    # top (max_pop_size - 1) by fitness).
    if len(pop.variants) > max_pop_size:
        # Always keep index 0 (the compiled default).
        keep = {0}
        sorted_idx = sorted(range(len(pop.variants)),
                            key=lambda i: pop.fitness[i])
        for i in sorted_idx:
            if len(keep) >= max_pop_size:
                break
            keep.add(i)
        keep_sorted = sorted(keep)
        new_variants = [pop.variants[i] for i in keep_sorted]
        new_fitness = [pop.fitness[i] for i in keep_sorted]
        new_sources: list[str | None] = []
        if pop.source_variants:
            for i in keep_sorted:
                if i < len(pop.source_variants):
                    new_sources.append(pop.source_variants[i])
                else:
                    new_sources.append(None)
        pop.variants = new_variants
        pop.fitness = new_fitness
        if new_sources:
            pop.source_variants = new_sources
        # Re-find best in trimmed pop.
        best_i = min(range(len(pop.fitness)), key=lambda i: pop.fitness[i])
        pop.best_idx = best_i
        stats.best_after = pop.fitness[best_i]

    return stats


# ---------------------------------------------------------------------------
# Genome-level commit: replace genome.ir with the best-variant splice
# ---------------------------------------------------------------------------

def commit_best_variants_to_ir(genome: "AlgorithmGenome") -> bool:
    """For each slot population in ``genome``, splice in
    ``pop.variants[pop.best_idx]`` and update ``genome.ir`` accordingly.

    Returns True if any commit succeeded, False otherwise. After this
    call, ``genome.ir`` is the post-best-variants flat IR; any subsequent
    ``materialize(genome)`` will reflect the winning variants directly.
    """
    if not genome.slot_populations:
        return False
    changed = False
    for pop_key, pop in genome.slot_populations.items():
        if not pop.variants:
            continue
        if pop.best_idx >= len(pop.variants):
            continue
        # If best_idx==0 AND it's the default, skip (no change).
        if pop.best_idx == 0:
            continue
        best_variant = pop.variants[pop.best_idx]
        new_ir = apply_slot_variant(genome, pop_key, best_variant)
        if new_ir is None:
            continue
        genome.ir = new_ir
        changed = True
        # After commit, the default becomes the new best_idx=0 conceptually,
        # but we keep the population intact as history; we just reset
        # best_idx so subsequent micro-gens compare against this winner.
        # Move winning variant to slot 0 to preserve "default == current".
        winner = pop.variants[pop.best_idx]
        winner_fit = pop.fitness[pop.best_idx]
        winner_src = (pop.source_variants[pop.best_idx]
                      if pop.source_variants
                      and pop.best_idx < len(pop.source_variants) else None)
        # Swap into position 0.
        pop.variants[0], pop.variants[pop.best_idx] = winner, pop.variants[0]
        pop.fitness[0], pop.fitness[pop.best_idx] = winner_fit, pop.fitness[0]
        if pop.source_variants and pop.best_idx < len(pop.source_variants):
            pop.source_variants[0], pop.source_variants[pop.best_idx] = (
                winner_src, pop.source_variants[0]
            )
        pop.best_idx = 0
    return changed
