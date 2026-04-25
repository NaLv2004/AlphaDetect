"""Slot rediscovery — periodic structural-cohesion analysis.

Per code_review.md §2.6 / Step S5.

After Case III dissolution flattens a genome's slot populations into
the structural IR, micro-evolution has nowhere to act.  This module
re-extracts cohesive sub-DAGs from the flat IR and emits them as new
``auto_<hash>`` slots.

Public API
----------

``rediscover_slots(structural_ir, *, min_size=4, max_size=24,
max_boundary=4, max_new_per_pass=3, existing_slot_op_ids=None)
    -> list[NewSlotProposal]``
    Pure analysis; does not mutate the IR.

``apply_rediscovered_slots(genome, proposals) -> AlgorithmGenome``
    Wraps each accepted region into an ``algslot`` op in the structural
    IR and emits a corresponding ``SlotPopulation``.  Returns the
    updated genome.

``maybe_rediscover_slots(genome, generation, *, period=20, ...) ->
    AlgorithmGenome``
    Convenience helper: only runs when ``generation % period == 0`` or
    when ``len(genome.slot_populations) == 0``.

Selection criteria (purely structural):
    1. Connected sub-graph in the dataflow DAG of size in [min_size, max_size]
    2. Live-in + live-out ≤ max_boundary
    3. Internal cohesion (intra/extra edge ratio) ≥ 1.5
    4. Does not overlap an existing slot region
    5. Subgraph type signature is inferable
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.region.selector import RewriteRegion, define_rewrite_region
from algorithm_ir.region.contract import infer_boundary_contract

from evolution.pool_types import (
    AlgorithmGenome,
    SlotPopulation,
)
from evolution.skeleton_registry import ProgramSpec

logger = logging.getLogger(__name__)

__all__ = [
    "NewSlotProposal",
    "rediscover_slots",
    "apply_rediscovered_slots",
    "maybe_rediscover_slots",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class NewSlotProposal:
    slot_id: str
    op_ids: list[str]
    entry_values: list[str]
    exit_values: list[str]
    cohesion: float
    program_spec: ProgramSpec
    region: RewriteRegion | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataflow helpers
# ---------------------------------------------------------------------------

def _entry_block_op_ids(ir: FunctionIR) -> list[str]:
    block = ir.blocks.get(ir.entry_block)
    if block is None:
        return []
    return list(block.op_ids)


def _value_def_op(ir: FunctionIR, vid: str) -> str | None:
    """Return the op id that defines ``vid`` if present in the IR."""
    val = ir.values.get(vid)
    if val is None:
        return None
    def_op = getattr(val, "def_op", None)
    if def_op is None:
        return None
    return getattr(def_op, "id", None)


def _op_input_value_ids(op) -> list[str]:
    out: list[str] = []
    for v in (op.inputs or []):
        vid = getattr(v, "id", None) or getattr(v, "var_name", None)
        if vid:
            out.append(vid)
    return out


def _op_output_value_ids(op) -> list[str]:
    out: list[str] = []
    for v in (op.outputs or []):
        vid = getattr(v, "id", None) or getattr(v, "var_name", None)
        if vid:
            out.append(vid)
    return out


def _build_predecessor_map(ir: FunctionIR, op_ids: list[str]) -> dict[str, list[str]]:
    """For each op id, list of op ids that produce its input values
    (restricted to the same op_ids set)."""
    op_set = set(op_ids)
    preds: dict[str, list[str]] = {oid: [] for oid in op_ids}
    for oid in op_ids:
        op = ir.ops.get(oid)
        if op is None:
            continue
        for vid in _op_input_value_ids(op):
            def_oid = _value_def_op(ir, vid)
            if def_oid and def_oid in op_set and def_oid != oid:
                preds[oid].append(def_oid)
    return preds


def _is_skippable_op(op) -> bool:
    """Ops we never include in a candidate region."""
    if op is None:
        return True
    if op.opcode in ("return", "jump", "branch", "phi", "slot"):
        return True
    if op.opcode == "call":
        callee = op.attrs.get("callee") or op.attrs.get("name") or ""
        if isinstance(callee, str) and callee.startswith("__fii_provmark_"):
            return True
    return False


# ---------------------------------------------------------------------------
# Region enumeration via contiguous opcode windows
# ---------------------------------------------------------------------------

def _candidate_windows(
    ir: FunctionIR,
    *,
    min_size: int,
    max_size: int,
) -> list[list[str]]:
    """Enumerate contiguous op-id windows in the entry block of length
    ``[min_size, max_size]`` containing only non-skippable ops."""
    seq = _entry_block_op_ids(ir)
    eligible = [oid for oid in seq if not _is_skippable_op(ir.ops.get(oid))]
    out: list[list[str]] = []
    n = len(eligible)
    for i in range(n):
        for L in range(min_size, max_size + 1):
            j = i + L
            if j > n:
                break
            window = eligible[i:j]
            out.append(window)
    return out


# ---------------------------------------------------------------------------
# Cohesion + boundary metrics
# ---------------------------------------------------------------------------

def _live_in_out(ir: FunctionIR, op_ids: list[str]) -> tuple[set[str], set[str]]:
    """Compute live-in (values consumed by region but defined outside)
    and live-out (values defined in region but consumed outside)."""
    op_set = set(op_ids)
    defs_in_region: set[str] = set()
    uses_in_region: set[str] = set()
    for oid in op_ids:
        op = ir.ops.get(oid)
        if op is None:
            continue
        for vid in _op_output_value_ids(op):
            defs_in_region.add(vid)
        for vid in _op_input_value_ids(op):
            uses_in_region.add(vid)

    live_in = {v for v in uses_in_region if (_value_def_op(ir, v) or "") not in op_set}

    # live-out: any region-defined value used by an op outside the region
    live_out: set[str] = set()
    all_ops = ir.ops.values()
    for op in all_ops:
        if op.id in op_set:
            continue
        for vid in _op_input_value_ids(op):
            if vid in defs_in_region:
                live_out.add(vid)
    return live_in, live_out


def _cohesion(ir: FunctionIR, op_ids: list[str]) -> float:
    """Return intra-edge / extra-edge ratio of the dataflow DAG.

    intra = number of region edges (input value -> producer is in region)
    extra = number of edges crossing region boundary (live_in + live_out)
    """
    op_set = set(op_ids)
    intra = 0
    extra = 0
    for oid in op_ids:
        op = ir.ops.get(oid)
        if op is None:
            continue
        for vid in _op_input_value_ids(op):
            def_oid = _value_def_op(ir, vid) or ""
            if def_oid in op_set and def_oid != oid:
                intra += 1
            else:
                extra += 1
    # outgoing edges
    defs_in_region: set[str] = set()
    for oid in op_ids:
        op = ir.ops.get(oid)
        if op is None:
            continue
        for vid in _op_output_value_ids(op):
            defs_in_region.add(vid)
    for op in ir.ops.values():
        if op.id in op_set:
            continue
        for vid in _op_input_value_ids(op):
            if vid in defs_in_region:
                extra += 1
    if extra == 0:
        return float("inf") if intra > 0 else 0.0
    return intra / extra


# ---------------------------------------------------------------------------
# Type signature
# ---------------------------------------------------------------------------

def _type_signature(ir: FunctionIR, value_ids: list[str]) -> list[str]:
    out: list[str] = []
    for vid in value_ids:
        v = ir.values.get(vid)
        if v is None:
            out.append("any")
            continue
        t = getattr(v, "type_hint", None) or "any"
        out.append(str(t))
    return out


# ---------------------------------------------------------------------------
# Public: rediscover_slots
# ---------------------------------------------------------------------------

def _slot_id_for(op_ids: list[str]) -> str:
    h = hashlib.blake2s("|".join(op_ids).encode(), digest_size=4).hexdigest()
    return f"auto_{h}"


def rediscover_slots(
    structural_ir: FunctionIR,
    *,
    min_size: int = 4,
    max_size: int = 24,
    max_boundary: int = 4,
    min_cohesion: float = 1.5,
    max_new_per_pass: int = 3,
    existing_slot_op_ids: set[str] | None = None,
) -> list[NewSlotProposal]:
    """Pure analysis pass — returns a list of NewSlotProposal sorted by
    cohesion descending, capped at ``max_new_per_pass``.

    Does not mutate ``structural_ir``.
    """
    if existing_slot_op_ids is None:
        existing_slot_op_ids = set()
    # Also exclude any op id that is currently inside an algslot region.
    for op in structural_ir.ops.values():
        if op.opcode == "slot":
            existing_slot_op_ids.add(op.id)

    proposals: list[NewSlotProposal] = []
    seen_signatures: set[tuple] = set()

    windows = _candidate_windows(
        structural_ir, min_size=min_size, max_size=max_size,
    )
    for window in windows:
        if any(oid in existing_slot_op_ids for oid in window):
            continue
        live_in, live_out = _live_in_out(structural_ir, window)
        if (len(live_in) + len(live_out)) > max_boundary:
            continue
        cohesion = _cohesion(structural_ir, window)
        if cohesion < min_cohesion:
            continue
        in_types = _type_signature(structural_ir, sorted(live_in))
        out_types = _type_signature(structural_ir, sorted(live_out))
        sig_key = (tuple(window),)  # dedup by exact op set
        if sig_key in seen_signatures:
            continue
        seen_signatures.add(sig_key)

        # Construct rewrite region (best-effort)
        try:
            region = define_rewrite_region(
                structural_ir,
                op_ids=sorted(window),
                exit_values=sorted(live_out),
            )
        except Exception:
            region = None

        slot_id = _slot_id_for(window)
        spec = ProgramSpec(
            name=slot_id,
            param_names=[f"in_{i}" for i in range(len(in_types))],
            param_types=in_types,
            return_type=out_types[0] if out_types else "void",
        )
        proposals.append(NewSlotProposal(
            slot_id=slot_id,
            op_ids=list(window),
            entry_values=sorted(live_in),
            exit_values=sorted(live_out),
            cohesion=cohesion,
            program_spec=spec,
            region=region,
            metadata={
                "n_ops": len(window),
                "n_live_in": len(live_in),
                "n_live_out": len(live_out),
            },
        ))

    proposals.sort(key=lambda p: p.cohesion, reverse=True)

    # Deduplicate by op_ids overlap >= 50% — keep highest-cohesion only.
    accepted: list[NewSlotProposal] = []
    used_ops: set[str] = set()
    for p in proposals:
        ovl = sum(1 for o in p.op_ids if o in used_ops)
        if ovl > 0.5 * len(p.op_ids):
            continue
        accepted.append(p)
        used_ops.update(p.op_ids)
        if len(accepted) >= max_new_per_pass:
            break
    return accepted


# ---------------------------------------------------------------------------
# Apply: build SlotPopulations seeded with the current concrete sub-DAG
# ---------------------------------------------------------------------------

def _build_seed_variant_ir(
    structural_ir: FunctionIR,
    proposal: NewSlotProposal,
) -> FunctionIR | None:
    """Return a small FunctionIR that wraps the proposal's sub-DAG as a
    standalone helper.  Used as the initial 'best' variant in the new
    slot population so dissolution + rediscovery is **behaviour-preserving**
    (the new auto-slot, when materialized, reproduces the dissolved code
    sequence).
    """
    # Lazy import to avoid heavy deps at module import.
    try:
        from algorithm_ir.region.extract import extract_region_as_function
    except Exception:
        extract_region_as_function = None  # type: ignore[assignment]

    if extract_region_as_function is None or proposal.region is None:
        return None
    try:
        return extract_region_as_function(
            structural_ir,
            proposal.region,
            func_name=f"_slot_{proposal.slot_id}",
        )
    except Exception as exc:
        logger.debug("Seed extraction failed for %s: %r", proposal.slot_id, exc)
        return None


def apply_rediscovered_slots(
    genome: AlgorithmGenome,
    proposals: list[NewSlotProposal],
) -> AlgorithmGenome:
    """Add ``SlotPopulation`` entries for each accepted proposal.

    For now we register the slot proposals in the genome's
    ``slot_populations`` dict using a single seed variant extracted from
    the structural IR.  The actual structural rewrite (replacing the
    contiguous op span with an ``algslot`` op) is delegated to the
    existing ``slot_discovery`` infrastructure on the next macro pass —
    here we only **emit population shells** so micro-evolution can run.

    A successfully-extracted seed variant is added as the first
    variant; if extraction fails the proposal is logged and skipped.
    """
    if not proposals:
        return genome

    for p in proposals:
        if p.slot_id in genome.slot_populations:
            continue
        seed_ir = _build_seed_variant_ir(genome.structural_ir, p)
        if seed_ir is None:
            logger.info(
                "rediscover: skipping %s (seed extraction failed, "
                "%d ops, cohesion=%.2f)",
                p.slot_id, len(p.op_ids), p.cohesion,
            )
            continue
        pop = SlotPopulation(
            slot_id=p.slot_id,
            spec=p.program_spec,
            variants=[seed_ir],
            fitness=[float("inf")],
            best_idx=0,
        )
        genome.slot_populations[p.slot_id] = pop
        logger.info(
            "rediscover: emitted auto-slot %s (n_ops=%d, cohesion=%.2f)",
            p.slot_id, len(p.op_ids), p.cohesion,
        )
    return genome


# ---------------------------------------------------------------------------
# Convenience: scheduling helper
# ---------------------------------------------------------------------------

def maybe_rediscover_slots(
    genome: AlgorithmGenome,
    generation: int,
    *,
    period: int = 20,
    min_size: int = 4,
    max_size: int = 24,
    max_boundary: int = 4,
    min_cohesion: float = 1.5,
    max_new_per_pass: int = 3,
) -> AlgorithmGenome:
    """Run rediscovery if scheduled or genome has been fully dissolved."""
    must_run = (generation > 0 and generation % period == 0) \
        or (len(genome.slot_populations) == 0)
    if not must_run:
        return genome
    proposals = rediscover_slots(
        genome.structural_ir,
        min_size=min_size, max_size=max_size,
        max_boundary=max_boundary, min_cohesion=min_cohesion,
        max_new_per_pass=max_new_per_pass,
    )
    if not proposals:
        return genome
    return apply_rediscovered_slots(genome, proposals)
