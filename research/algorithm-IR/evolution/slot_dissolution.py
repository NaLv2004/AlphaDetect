"""Slot dissolution for Case III grafts.

Per code_review.md §2.5 / Step S5:

When a graft proposal selected on the Fully-Inlined IR (FII) crosses
slot boundaries (Case III — ``classify_region`` returned ``"III"``),
we cannot back-map it to either a single slot variant (Case I) or to
``structural_ir`` alone (Case II).  Instead we **dissolve** the touched
slots:

    1. Take the current FII view (it already has the touched slots
       inlined with provenance tags).
    2. Apply the GNN-proposed graft directly on the FII.
    3. Promote the result to the new ``structural_ir`` after stripping
       provenance markers (the `__fii_provmark_*` sentinel calls and
       their `_provenance` attrs).
    4. Drop dissolved slots from ``slot_populations``.  Per spec §2.6,
       slot rediscovery (run periodically, every N=20 macro generations)
       will recover new auto-slots from the now-flat structural_ir.

For the first implementation we adopt the **full-dissolution** policy:
when any slot is touched, we commit the whole FII as the new
``structural_ir`` and clear ``slot_populations`` entirely.  This is the
spec-conformant superset of "affected slots are inlined": affected
slots ARE inlined, plus a few more.  Untouched slots will be
re-extracted by the rediscovery pass before they can drift far from
their previous variant pool.

Module API
----------

``dissolve_and_graft(host_genome, proposal, fii_ir, *, generation,
donor_algo_id) -> DispatchResult``
    Mirrors ``graft_dispatch.dispatch_graft`` for Case III.  Returns
    ``DispatchResult(child=..., case="case_III")`` on success, or
    ``DispatchResult(child=None, case="case_III_failed", diagnostic=...)``
    on any failure.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.ir.validator import validate_function_ir
from algorithm_ir.grafting.graft_general import graft_general

from evolution.pool_types import (
    AlgorithmGenome,
    GraftProposal,
    GraftRecord,
)
from evolution.graft_dispatch import (
    DispatchResult,
    classify_region,
    _entry_block_op_ids,  # noqa: F401  (intentional reuse)
)

logger = logging.getLogger(__name__)

__all__ = ["dissolve_and_graft", "strip_provenance_markers"]


# ---------------------------------------------------------------------------
# Marker / provenance scrubbing
# ---------------------------------------------------------------------------

def _is_marker_op(op) -> bool:
    """Return True if ``op`` is a `__fii_provmark_*` sentinel call."""
    if op is None:
        return False
    if op.opcode != "call":
        return False
    callee = op.attrs.get("callee") or op.attrs.get("name") or ""
    return isinstance(callee, str) and callee.startswith("__fii_provmark_")


def strip_provenance_markers(ir: FunctionIR) -> FunctionIR:
    """Return a deep copy of ``ir`` with marker calls and provenance
    attributes removed.

    Called on the post-graft IR before installing it as the new
    ``structural_ir``.  Markers are removed so the resulting IR is
    valid input to ``materialize`` and ``compile_source_to_ir`` in
    downstream passes that don't know about FII sentinels.
    """
    new_ir = deepcopy(ir)

    # Drop marker ops from every block's op list and from new_ir.ops.
    marker_ids: list[str] = []
    for block in new_ir.blocks.values():
        kept: list[str] = []
        for oid in block.op_ids:
            op = new_ir.ops.get(oid)
            if _is_marker_op(op):
                marker_ids.append(oid)
                continue
            kept.append(oid)
        block.op_ids = kept

    for mid in marker_ids:
        op = new_ir.ops.pop(mid, None)
        if op is None:
            continue
        # Drop produced values (markers produce one unused value each).
        for v in getattr(op, "outputs", []) or []:
            vid = getattr(v, "id", None) or getattr(v, "var_name", None)
            if vid and vid in new_ir.values:
                new_ir.values.pop(vid, None)

    # Strip _provenance attribute from remaining ops.
    for op in new_ir.ops.values():
        if isinstance(op.attrs, dict) and "_provenance" in op.attrs:
            op.attrs = {k: v for k, v in op.attrs.items() if k != "_provenance"}

    # Also strip any module-level provenance attrs.
    if hasattr(new_ir, "attrs") and isinstance(new_ir.attrs, dict):
        for k in ("_provenance_map", "_provenance_call_sites"):
            new_ir.attrs.pop(k, None)

    return new_ir


# ---------------------------------------------------------------------------
# Dissolution + graft
# ---------------------------------------------------------------------------

def _filter_region_for_graft(region_op_ids: list[str], fii_ir: FunctionIR) -> list[str]:
    """Drop marker ops from a region — markers must never be part of
    the cut."""
    out: list[str] = []
    for oid in region_op_ids:
        op = fii_ir.ops.get(oid)
        if op is None or _is_marker_op(op):
            continue
        prov = op.attrs.get("_provenance") or {}
        if prov.get("is_slot_boundary"):
            continue
        out.append(oid)
    return out


def dissolve_and_graft(
    host_genome: AlgorithmGenome,
    proposal: GraftProposal,
    fii_ir: FunctionIR,
    *,
    generation: int,
    donor_algo_id: str,
) -> DispatchResult:
    """Dissolve touched slots and apply the Case III graft on the FII.

    The post-dissolution genome has:
      * ``structural_ir = strip_provenance_markers(post_graft_fii)``
      * ``slot_populations = {}``  (empty — to be repopulated by the
        slot rediscovery pass)
      * ``graft_history`` extended with a Case III record
      * ``parent_ids = (host_genome.algo_id, donor_algo_id)``
      * ``generation = generation``

    Returns ``DispatchResult(child=new_genome, case="case_III")`` on
    success.  Any failure (region invalid, graft failure, IR validation
    failure) returns ``DispatchResult(child=None, case="case_III_failed",
    diagnostic=...)``.
    """
    region_ops = _filter_region_for_graft(list(proposal.region), fii_ir)
    if not region_ops:
        return DispatchResult(None, "case_III_failed", "empty_region_after_marker_strip")

    # Re-classify after marker strip — must still be Case III (or II).
    summary = classify_region(fii_ir, region_ops)
    if summary["case"] not in ("II", "III"):
        return DispatchResult(
            None,
            "case_III_failed",
            f"unexpected_case_after_strip:{summary['case']}",
        )

    # Build a proposal whose region targets the FII (no remapping needed —
    # FII op IDs are already the canonical addresses).
    graft_proposal = GraftProposal(
        proposal_id=proposal.proposal_id + "::dissolved",
        host_algo_id=host_genome.algo_id,
        region=region_ops,
        contract=None,
        donor_algo_id=donor_algo_id,
        donor_ir=proposal.donor_ir,
        donor_region=list(proposal.donor_region) if proposal.donor_region else [],
        dependency_overrides=dict(proposal.dependency_overrides or {}),
        port_mapping=None,
        confidence=proposal.confidence,
        rationale=(proposal.rationale or "") + " [dissolved]",
    )

    # Apply graft on FII directly.
    try:
        post_graft = graft_general(
            host_ir=fii_ir,
            proposal=graft_proposal,
        )
    except Exception as exc:
        logger.debug("Case III graft_general failed: %r", exc)
        return DispatchResult(None, "case_III_failed", f"graft_general:{exc!r}")

    if post_graft is None:
        return DispatchResult(None, "case_III_failed", "graft_general_returned_none")

    # Strip markers / provenance and validate.
    new_struct = strip_provenance_markers(post_graft)
    try:
        validate_function_ir(new_struct)
    except Exception as exc:
        logger.debug("Case III post-strip validation failed: %r", exc)
        return DispatchResult(None, "case_III_failed", f"validate:{exc!r}")

    # Build the dissolved child genome.
    child = host_genome.clone()
    child.structural_ir = new_struct
    # Drop ALL slot populations: full-dissolution policy.  Rediscovery
    # will re-emit auto-slots in subsequent passes.
    dissolved_slot_keys = list(child.slot_populations.keys())
    child.slot_populations = {}
    child.parent_ids = (host_genome.algo_id, donor_algo_id)
    child.generation = generation
    # clone() already assigns a fresh id; tag with case_III.
    existing_tags = getattr(host_genome, "tags", None) or set()
    child.tags = set(existing_tags) | {"case_III"}

    summary_str = (
        f"case_III; dissolved_slots={len(dissolved_slot_keys)}; "
        f"region_size={len(region_ops)}; "
        f"slots_touched={sorted(str(s) for s in summary['slot_ids'] if s)}"
    )
    record = GraftRecord(
        generation=generation,
        host_algo_id=host_genome.algo_id,
        donor_algo_id=donor_algo_id,
        proposal_id=graft_proposal.proposal_id,
        region_summary=summary_str,
        new_slots_created=[],
    )
    child.graft_history = list(getattr(host_genome, "graft_history", []) or []) + [record]

    return DispatchResult(child, "case_III")
