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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.ir.validator import validate_function_ir
from algorithm_ir.region.selector import RewriteRegion

if TYPE_CHECKING:
    from evolution.pool_types import AlgorithmGenome, SlotPopulation

logger = logging.getLogger(__name__)

__all__ = [
    "map_pop_key_to_from_slot_ids",
    "collect_slot_region",
    "apply_slot_variant",
    "evaluate_slot_variant",
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
    skipped_no_sids: int = 0      # pop_key has no matching from_slot_id (annotation lost)
    skipped_no_variants: int = 0  # pop has zero variants

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
            "skipped_no_sids": self.skipped_no_sids,
            "skipped_no_variants": self.skipped_no_variants,
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

    # R2: route through the unified region resolver. Provenance tier is
    # the production path; binding tier is reserved for future explicit
    # bindings.
    from evolution.gp.region_resolver import resolve_slot_region
    info = resolve_slot_region(genome, pop_key)
    if info is None:
        return None

    sids = set(info.sids)
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

    # Re-annotate newly introduced ops so the slot stays discoverable
    # by ``map_pop_key_to_from_slot_ids`` on subsequent micro-gens. The
    # graft inlines fresh ops without _provenance.from_slot_id matching
    # the original pop_key; without re-annotation, the next call to
    # ``apply_slot_variant`` for the same pop_key would find no sids and
    # silently no-op (the bug that froze slot-evo from gen ~4 onward).
    anchor_sid = next(iter(sids))
    pre_op_ids = set(genome.ir.ops.keys())
    for op_id, op in artifact.ir.ops.items():
        if op_id in pre_op_ids:
            continue
        prov = op.attrs.setdefault("_provenance", {})
        prov.setdefault("from_slot_id", anchor_sid)
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


def evaluate_slot_variant(genome: "AlgorithmGenome",
                          pop_key: str,
                          variant_ir: FunctionIR,
                          *,
                          evaluator: Any,
                          n_trials: int = 8,
                          timeout_sec: float = 1.0,
                          snr_db: float = 16.0) -> tuple[float, None]:
    """Splice the variant in, hand the spliced FunctionIR to the
    evaluator, return ``(ser, None)``.

    The single-representation principle (Phase H+4): this function does
    NOT inspect Python source. The evaluator is the only component
    permitted to materialise IR -> source -> exec.

    Returns ``(ser, None)`` (the second tuple slot is preserved for
    backward-compatible call sites; it is always ``None`` now).
    """
    flat_ir = apply_slot_variant(genome, pop_key, variant_ir)
    if flat_ir is None:
        return float("inf"), None

    # The evaluator owns source emission. We only ever pass FunctionIR.
    eval_ir = getattr(evaluator, "evaluate_ir_quick", None)
    if eval_ir is not None:
        try:
            ser = float(eval_ir(
                flat_ir,
                algo_id=f"{genome.algo_id}_slot_{pop_key}",
                n_trials=int(n_trials),
                timeout_sec=float(timeout_sec),
                snr_db=float(snr_db),
            ))
        except Exception as exc:
            logger.debug("evaluate_slot_variant: evaluate_ir_quick raised: %r", exc)
            return float("inf"), None
        if not np.isfinite(ser):
            return float("inf"), None
        return ser, None

    # Subprocess evaluator: we are forced to emit source here because
    # the evaluator's wire-format is a string. This is still consistent
    # with the principle: source emission happens at the EVALUATION
    # boundary, not inside any GP or selection logic.
    eval_quick = getattr(evaluator, "evaluate_source_quick", None)
    if eval_quick is not None:
        from algorithm_ir.regeneration.codegen import emit_python_source
        try:
            source = emit_python_source(flat_ir)
        except Exception as exc:
            logger.debug("evaluate_slot_variant: codegen failed: %r", exc)
            return float("inf"), None
        try:
            ser = float(eval_quick(
                source, _func_name_from_ir(flat_ir),
                algo_id=f"{genome.algo_id}_slot_{pop_key}",
                n_trials=int(n_trials),
                timeout_sec=float(timeout_sec),
                snr_db=float(snr_db),
            ))
        except Exception as exc:
            logger.debug("evaluate_slot_variant: subprocess eval failed: %r", exc)
            return float("inf"), None
        if not np.isfinite(ser):
            return float("inf"), None
        return ser, None

    # In-process fallback: build a transient genome with the spliced
    # IR and call evaluator.evaluate(). The evaluator handles all
    # codegen internally.
    try:
        from copy import copy as _shallow_copy
        transient = _shallow_copy(genome)
        transient.ir = flat_ir
        res = evaluator.evaluate(transient)
        ser = float(getattr(res, "metrics", {}).get("ser", float("inf")))
    except Exception as exc:
        logger.debug("evaluate_slot_variant: in-proc fallback failed: %r", exc)
        return float("inf"), None
    if not np.isfinite(ser):
        return float("inf"), None
    return ser, None


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
                         max_pop_size: int = 16) -> SlotMicroStats:
    """Run one (μ+λ) micro-generation on ``pop`` via the typed GP framework.

    Phase H+4: there is only one path — the typed GP MicroPopulation.
    The legacy constant-perturbation kernel and the ``use_typed_gp``
    switch have been removed.
    """
    from evolution.gp.population import micro_population_step
    return micro_population_step(
        genome, pop_key, pop,
        evaluator=evaluator, rng=rng,
        n_children=n_children, n_trials=n_trials,
        timeout_sec=timeout_sec, snr_db=snr_db,
        max_pop_size=max_pop_size,
    )


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
        # Move winning variant to slot 0 to preserve "default == current".
        winner = pop.variants[pop.best_idx]
        winner_fit = pop.fitness[pop.best_idx]
        pop.variants[0], pop.variants[pop.best_idx] = winner, pop.variants[0]
        pop.fitness[0], pop.fitness[pop.best_idx] = winner_fit, pop.fitness[0]
        pop.best_idx = 0
    return changed
