"""Graft dispatcher — classify a graft as Case I/II/III and execute.

Per code_review.md §2.5 and Step S4:

A graft proposal selected on the FII (Fully-Inlined IR) is classified
by examining the provenance of every op in the proposal's region:

    Case I   — All ops have the same non-None ``from_slot_id`` (single
               slot), all share the same ``call_site_id``, and none is
               a slot boundary marker. The graft is **back-mapped** to
               the slot's variant IR and the result is appended as a
               new variant in that slot's micro-population.

    Case II  — All ops have ``from_slot_id is None`` (purely structural)
               and none is a boundary marker. The graft is back-mapped
               to ``host_genome.structural_ir`` and applied directly.

    Case III — Mixed (multiple slots, or both structural and in-slot
               ops, or any boundary marker present). For Step S4 these
               are **rejected and logged**. (Case III dissolution is
               implemented in ``slot_dissolution.py`` as Step S5.)

Op-ID back-mapping strategy
---------------------------

The FII IR's op IDs differ from those of either ``structural_ir`` or
the slot variant IRs (each goes through its own compile pipeline). To
map FII op IDs to target op IDs we perform a **positional opcode-
sequence alignment** of each block:

* For Case I, the FII frame body (the ops inside one call site between
  marker_begin and marker_end) corresponds 1-to-1 with the slot variant
  IR's main block ops (excluding the trailing ``return`` op).

* For Case II, the FII main block ops with ``from_slot_id is None``
  (excluding markers) correspond 1-to-1 with ``structural_ir``'s main
  block ops EXCLUDING the ``slot`` (algslot) ops — every algslot op in
  ``structural_ir`` is replaced in FII by a [marker_begin, frame body,
  marker_end] sequence.

The alignment is opcode-equality-checked at every step. If alignment
fails (different lengths or opcode mismatch) the dispatcher returns
``None`` with a diagnostic classification string instead of producing
an unsound graft.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.ir.validator import validate_function_ir
from algorithm_ir.grafting.graft_general import graft_general

from evolution.pool_types import (
    AlgorithmGenome,
    GraftRecord,
    GraftProposal,
    SlotPopulation,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DispatchResult",
    "dispatch_graft",
    "classify_region",
]


# ---------------------------------------------------------------------------
# Dispatch result container
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    child: AlgorithmGenome | None
    case: str  # "case_I" | "case_II" | "case_III_rejected" | "<reason>"
    diagnostic: str = ""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_region(fii_ir: FunctionIR, region_op_ids: list[str]) -> dict[str, Any]:
    """Inspect provenance of every region op and return a summary dict.

    Returns a dict with keys:
        ``case``           : "I" | "II" | "III"
        ``slot_ids``       : set of distinct from_slot_id values (None included if any)
        ``call_site_ids``  : set of distinct call_site_id values
        ``has_boundary``   : True if any op is a slot boundary marker
        ``unique_slot``    : the single slot id if Case I, else None
        ``unique_callsite``: the single call site id if Case I, else None
    """
    slot_ids: set = set()
    call_site_ids: set = set()
    has_boundary = False

    for oid in region_op_ids:
        op = fii_ir.ops.get(oid)
        if op is None:
            return {
                "case": "III",
                "slot_ids": set(),
                "call_site_ids": set(),
                "has_boundary": False,
                "unique_slot": None,
                "unique_callsite": None,
                "reason": "unknown_op_id",
            }
        prov = op.attrs.get("_provenance") or {}
        if prov.get("is_slot_boundary"):
            has_boundary = True
        slot_ids.add(prov.get("from_slot_id"))
        call_site_ids.add(prov.get("call_site_id"))

    if has_boundary:
        case = "III"
    elif slot_ids == {None}:
        case = "II"
    elif None not in slot_ids and len(slot_ids) == 1 and len(call_site_ids) == 1:
        case = "I"
    else:
        case = "III"

    summary = {
        "case": case,
        "slot_ids": slot_ids,
        "call_site_ids": call_site_ids,
        "has_boundary": has_boundary,
        "unique_slot": next(iter(slot_ids)) if case == "I" else None,
        "unique_callsite": next(iter(call_site_ids)) if case == "I" else None,
    }
    return summary


# ---------------------------------------------------------------------------
# Block / op order helpers
# ---------------------------------------------------------------------------

def _entry_block_op_ids(ir: FunctionIR) -> list[str]:
    """Return op ids in declaration order from the entry block only."""
    block = ir.blocks.get(ir.entry_block)
    if block is None:
        return []
    return list(block.op_ids)


def _is_marker_op(ir: FunctionIR, op_id: str) -> bool:
    op = ir.ops.get(op_id)
    if op is None:
        return False
    prov = op.attrs.get("_provenance") or {}
    return bool(prov.get("is_slot_boundary"))


def _frame_op_ids(fii_ir: FunctionIR, call_site_id: int) -> tuple[list[str], list[str]]:
    """Return (frame_body_op_ids, marker_op_ids) for a given call_site.

    The frame body is the contiguous span of ops in the entry block
    sitting between marker_begin and marker_end with the given
    ``call_site_id``. Marker ops themselves are excluded from the body.
    """
    body: list[str] = []
    markers: list[str] = []
    inside = False
    for oid in _entry_block_op_ids(fii_ir):
        op = fii_ir.ops.get(oid)
        if op is None:
            continue
        prov = op.attrs.get("_provenance") or {}
        op_csi = prov.get("call_site_id")
        is_boundary = bool(prov.get("is_slot_boundary"))
        if is_boundary and op_csi == call_site_id:
            kind = prov.get("boundary_kind")
            markers.append(oid)
            if kind == "enter":
                inside = True
            elif kind == "exit":
                inside = False
            continue
        if inside and op_csi == call_site_id:
            body.append(oid)
    return body, markers


def _variant_body_op_ids(variant_ir: FunctionIR) -> list[str]:
    """Return main-block ops of a slot variant IR, excluding the
    trailing ``return`` op.

    The slot helper compiles to a single function whose entry block ends
    in a ``return`` op.  Inlining drops that ``return`` (the helper's
    return value becomes an LHS assignment), so the corresponding FII
    frame body has one fewer op than the variant's entry block.
    """
    ops = _entry_block_op_ids(variant_ir)
    # Strip trailing return op(s)
    while ops:
        last = variant_ir.ops.get(ops[-1])
        if last is not None and last.opcode == "return":
            ops.pop()
            continue
        break
    return ops


def _structural_non_algslot_ops(struct_ir: FunctionIR) -> list[str]:
    """Return main-block ops of structural_ir, excluding algslot (`slot`) ops."""
    out: list[str] = []
    for oid in _entry_block_op_ids(struct_ir):
        op = struct_ir.ops.get(oid)
        if op is None:
            continue
        if op.opcode == "slot":
            continue
        out.append(oid)
    return out


def _structural_ops_with_algslot_marks(struct_ir: FunctionIR) -> list[tuple[str, str]]:
    """Return list of (kind, op_id) for the structural entry block.

    ``kind`` is ``"slot"`` for algslot ops and ``"op"`` for regular ops.
    Used by Case II alignment to identify algslot positions.
    """
    out: list[tuple[str, str]] = []
    for oid in _entry_block_op_ids(struct_ir):
        op = struct_ir.ops.get(oid)
        if op is None:
            continue
        kind = "slot" if op.opcode == "slot" else "op"
        out.append((kind, oid))
    return out


# ---------------------------------------------------------------------------
# Op-ID back-mapping
# ---------------------------------------------------------------------------

def _opcode_aligned(
    src_ir: FunctionIR, src_ops: list[str],
    dst_ir: FunctionIR, dst_ops: list[str],
) -> dict[str, str] | None:
    """Lockstep opcode-sequence alignment: returns ``{src_op_id: dst_op_id}``
    iff lengths match and every position has identical opcode."""
    if len(src_ops) != len(dst_ops):
        return None
    mapping: dict[str, str] = {}
    for s, d in zip(src_ops, dst_ops):
        sop = src_ir.ops.get(s)
        dop = dst_ir.ops.get(d)
        if sop is None or dop is None:
            return None
        if sop.opcode != dop.opcode:
            return None
        mapping[s] = d
    return mapping


def _build_case_I_mapping(
    fii_ir: FunctionIR,
    variant_ir: FunctionIR,
    call_site_id: int,
) -> dict[str, str] | None:
    """Build FII op_id → variant op_id mapping for one slot frame."""
    frame_body, _markers = _frame_op_ids(fii_ir, call_site_id)
    variant_body = _variant_body_op_ids(variant_ir)
    return _opcode_aligned(fii_ir, frame_body, variant_ir, variant_body)


def _build_case_II_mapping(
    fii_ir: FunctionIR,
    struct_ir: FunctionIR,
) -> dict[str, str] | None:
    """Build FII op_id → struct op_id for purely-structural ops.

    Walks the FII entry block. For each marker_begin we expect the
    next algslot op in struct_ir to be the corresponding placeholder;
    we advance the struct pointer past it. We skip ops inside frames
    (they have non-None from_slot_id). For each remaining FII op
    (structural) we align to the next non-algslot op in struct_ir.
    """
    fii_seq = _entry_block_op_ids(fii_ir)
    struct_seq = _structural_ops_with_algslot_marks(struct_ir)

    mapping: dict[str, str] = {}
    s_idx = 0  # index into struct_seq
    inside_frame = False

    for fii_oid in fii_seq:
        op = fii_ir.ops.get(fii_oid)
        if op is None:
            return None
        prov = op.attrs.get("_provenance") or {}
        is_bdry = bool(prov.get("is_slot_boundary"))
        kind = prov.get("boundary_kind")
        from_slot = prov.get("from_slot_id")

        if is_bdry and kind == "enter":
            # advance struct past one algslot op
            advanced = False
            while s_idx < len(struct_seq):
                k, _oid = struct_seq[s_idx]
                if k == "slot":
                    s_idx += 1
                    advanced = True
                    break
                # alignment violation: encountered structural op where
                # an algslot was expected
                return None
            if not advanced:
                return None
            inside_frame = True
            continue

        if is_bdry and kind == "exit":
            inside_frame = False
            continue

        if inside_frame or from_slot is not None:
            # op inside a slot frame: not part of structural mapping
            continue

        # purely structural op — align with next non-algslot op
        # (skipping any algslots that may be interleaved, though in
        # well-formed materialize output algslots are bounded by
        # markers so this skip should rarely fire here)
        while s_idx < len(struct_seq) and struct_seq[s_idx][0] == "slot":
            # An unbounded algslot in struct with no corresponding
            # frame in FII would be a serious shape mismatch.
            return None
        if s_idx >= len(struct_seq):
            return None
        k, s_oid = struct_seq[s_idx]
        if k != "op":
            return None
        s_op = struct_ir.ops.get(s_oid)
        if s_op is None or s_op.opcode != op.opcode:
            return None
        mapping[fii_oid] = s_oid
        s_idx += 1

    return mapping


# ---------------------------------------------------------------------------
# Proposal cloning with remapped region
# ---------------------------------------------------------------------------

def _clone_proposal_with_region(
    proposal: GraftProposal,
    new_region_op_ids: list[str],
) -> GraftProposal:
    """Return a shallow copy of ``proposal`` with a new region op_ids list.

    ``contract`` is dropped (set to None) so graft_general falls back to
    its standard boundary analysis on the remapped region. ``port_mapping``
    is also dropped — its keys reference FII value IDs which do not exist
    in the target IR.
    """
    new_region = deepcopy(proposal.region)
    new_region.op_ids = list(new_region_op_ids)
    # Wipe stale boundary info; graft_general will recompute via
    # find_region_boundary().
    new_region.entry_values = []
    new_region.exit_values = []
    new_region.read_set = []
    new_region.write_set = []

    new_proposal = GraftProposal(
        proposal_id=proposal.proposal_id,
        host_algo_id=proposal.host_algo_id,
        region=new_region,
        contract=None,
        donor_algo_id=proposal.donor_algo_id,
        donor_ir=proposal.donor_ir,
        donor_region=proposal.donor_region,
        dependency_overrides=list(proposal.dependency_overrides),
        port_mapping={},
        confidence=proposal.confidence,
        rationale=(proposal.rationale or "") + " [remapped]",
    )
    return new_proposal


# ---------------------------------------------------------------------------
# Case executors
# ---------------------------------------------------------------------------

def _execute_case_I(
    host_genome: AlgorithmGenome,
    proposal: GraftProposal,
    fii_ir: FunctionIR,
    slot_id: str,
    call_site_id: int,
    generation: int,
    donor_algo_id: str | None,
) -> DispatchResult:
    pop = host_genome.slot_populations.get(slot_id)
    if pop is None:
        return DispatchResult(None, "case_I_no_pop",
                              f"slot {slot_id!r} not in host slot_populations")
    if not pop.variants or pop.best_idx >= len(pop.variants):
        return DispatchResult(None, "case_I_no_variant",
                              f"slot {slot_id!r} has no usable variant")
    variant_ir = pop.variants[pop.best_idx]

    fii_to_var = _build_case_I_mapping(fii_ir, variant_ir, call_site_id)
    if fii_to_var is None:
        return DispatchResult(None, "case_I_unmapped",
                              f"opcode-sequence alignment failed for slot {slot_id!r} "
                              f"call_site={call_site_id}")

    # Map proposal region
    try:
        new_region_op_ids = [fii_to_var[oid] for oid in proposal.region.op_ids]
    except KeyError as e:
        return DispatchResult(None, "case_I_region_outside_frame",
                              f"region op {e!s} not in slot frame")

    new_proposal = _clone_proposal_with_region(proposal, new_region_op_ids)

    try:
        artifact = graft_general(variant_ir, new_proposal)
    except Exception as exc:
        return DispatchResult(None, "case_I_graft_failed", repr(exc))

    new_variant_ir = artifact.ir
    val_errs = validate_function_ir(new_variant_ir)
    if val_errs:
        return DispatchResult(
            None, "case_I_invalid_ir",
            f"validation: {val_errs[:3]}",
        )

    # Build child genome with new variant appended to slot pop
    child = host_genome.clone()
    child.algo_id = AlgorithmGenome._make_id()
    child.generation = generation
    child.parent_ids = [host_genome.algo_id, donor_algo_id or ""]
    child.tags = set(host_genome.tags) | {"grafted", "case_I"}
    child.metadata = dict(host_genome.metadata)
    child.metadata.update({
        "graft_host_algo_id": host_genome.algo_id,
        "graft_donor_algo_id": donor_algo_id,
        "graft_proposal_id": proposal.proposal_id,
        "dispatch_case": "I",
        "dispatch_slot": slot_id,
        "fii_grafted": True,
    })
    child.graft_history = list(host_genome.graft_history) + [
        GraftRecord(
            generation=generation,
            host_algo_id=host_genome.algo_id,
            donor_algo_id=donor_algo_id,
            proposal_id=proposal.proposal_id,
            region_summary=f"case_I:{slot_id}@{call_site_id}",
            new_slots_created=[],
        ),
    ]

    # Append new variant to the slot population in the child
    child_pop = child.slot_populations[slot_id]
    child_pop.variants.append(new_variant_ir)
    child_pop.fitness.append(float("inf"))
    if child_pop.source_variants:
        try:
            from algorithm_ir.regeneration.codegen import emit_python_source
            child_pop.source_variants.append(emit_python_source(new_variant_ir))
        except Exception:
            child_pop.source_variants.append("")
    # best_idx left unchanged; the new variant will be selected only
    # if it survives micro-evolution / fitness re-evaluation.

    return DispatchResult(child, "case_I",
                          f"appended new variant idx={len(child_pop.variants)-1} "
                          f"to slot {slot_id!r}")


def _execute_case_II(
    host_genome: AlgorithmGenome,
    proposal: GraftProposal,
    fii_ir: FunctionIR,
    generation: int,
    donor_algo_id: str | None,
) -> DispatchResult:
    struct_ir = host_genome.structural_ir

    fii_to_struct = _build_case_II_mapping(fii_ir, struct_ir)
    if fii_to_struct is None:
        return DispatchResult(None, "case_II_unmapped",
                              "structural opcode-sequence alignment failed")

    try:
        new_region_op_ids = [fii_to_struct[oid] for oid in proposal.region.op_ids]
    except KeyError as e:
        return DispatchResult(None, "case_II_region_unmapped",
                              f"region op {e!s} not mappable to struct")

    new_proposal = _clone_proposal_with_region(proposal, new_region_op_ids)

    try:
        artifact = graft_general(struct_ir, new_proposal)
    except Exception as exc:
        return DispatchResult(None, "case_II_graft_failed", repr(exc))

    val_errs = validate_function_ir(artifact.ir)
    if val_errs:
        return DispatchResult(
            None, "case_II_invalid_ir",
            f"validation: {val_errs[:3]}",
        )

    child = host_genome.clone()
    child.algo_id = AlgorithmGenome._make_id()
    child.structural_ir = artifact.ir
    child.generation = generation
    child.parent_ids = [host_genome.algo_id, donor_algo_id or ""]
    child.tags = set(host_genome.tags) | {"grafted", "case_II"}
    child.metadata = dict(host_genome.metadata)
    child.metadata.update({
        "graft_host_algo_id": host_genome.algo_id,
        "graft_donor_algo_id": donor_algo_id,
        "graft_proposal_id": proposal.proposal_id,
        "dispatch_case": "II",
        "fii_grafted": False,  # struct-level graft, not flat
    })
    child.graft_history = list(host_genome.graft_history) + [
        GraftRecord(
            generation=generation,
            host_algo_id=host_genome.algo_id,
            donor_algo_id=donor_algo_id,
            proposal_id=proposal.proposal_id,
            region_summary="case_II:structural",
            new_slots_created=list(artifact.new_slot_ids),
        ),
    ]

    # Initialise SlotPopulations for any new slots introduced by the donor.
    for slot_id in artifact.new_slot_ids:
        if slot_id in child.slot_populations:
            continue
        from evolution.skeleton_registry import ProgramSpec
        spec = ProgramSpec(
            name=slot_id,
            param_names=[],
            param_types=[],
            return_type="object",
        )
        child.slot_populations[slot_id] = SlotPopulation(
            slot_id=slot_id,
            spec=spec,
            variants=[],
            fitness=[],
            best_idx=0,
        )

    return DispatchResult(child, "case_II",
                          f"applied to structural_ir; new_slots={artifact.new_slot_ids}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def dispatch_graft(
    host_genome: AlgorithmGenome,
    proposal: GraftProposal,
    fii_ir: FunctionIR,
    *,
    generation: int = 0,
    donor_algo_id: str | None = None,
    accept_case_III: bool = False,
) -> DispatchResult:
    """Classify and execute a graft proposal selected on the FII IR.

    Parameters
    ----------
    host_genome : AlgorithmGenome
        The host algorithm whose FII view ``proposal.region.op_ids``
        reference.
    proposal : GraftProposal
        Proposal whose region.op_ids are FII op IDs.
    fii_ir : FunctionIR
        The host's FII IR (must carry per-op ``_provenance`` attrs).
    generation : int
        Current macro generation (recorded in child metadata).
    donor_algo_id : str | None
        Donor algorithm's ID, recorded in graft_history.
    accept_case_III : bool
        If False (Step S4 default), Case III proposals are rejected
        with diagnostic ``"case_III_rejected"``. Set True once Case III
        dissolution (Step S5) is wired in.

    Returns
    -------
    DispatchResult
    """
    region_op_ids = list(proposal.region.op_ids)
    if not region_op_ids:
        return DispatchResult(None, "empty_region", "proposal has no region ops")

    summary = classify_region(fii_ir, region_op_ids)
    case = summary["case"]

    if case == "I":
        return _execute_case_I(
            host_genome, proposal, fii_ir,
            slot_id=summary["unique_slot"],
            call_site_id=summary["unique_callsite"],
            generation=generation,
            donor_algo_id=donor_algo_id,
        )
    if case == "II":
        return _execute_case_II(
            host_genome, proposal, fii_ir,
            generation=generation,
            donor_algo_id=donor_algo_id,
        )
    # case == "III"
    if not accept_case_III:
        return DispatchResult(
            None, "case_III_rejected",
            f"slots={summary['slot_ids']} call_sites={summary['call_site_ids']} "
            f"has_boundary={summary['has_boundary']}",
        )
    # Step S5: Case III dissolution.
    # Local import to avoid circular import (slot_dissolution imports
    # symbols from this module).
    from evolution.slot_dissolution import dissolve_and_graft
    return dissolve_and_graft(
        host_genome, proposal, fii_ir,
        generation=generation,
        donor_algo_id=donor_algo_id,
    )
