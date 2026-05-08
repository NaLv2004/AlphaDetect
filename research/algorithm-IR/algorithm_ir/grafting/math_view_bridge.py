"""MathView <-> op_id bridge for graft pipeline.

This is the **sole** translation layer between the GNN's math-level view
(``algorithm_ir.ir.math_view.MathView``) and the op-id-based graft pipeline
(``algorithm_ir.grafting.graft_general``, ``algorithm_ir.region.selector``,
``evolution.graft_classifier`` etc.).

Design contract (locked, no fallback):
    * GNN sees and selects ``MathNode`` ids only.
    * graft_general / FunctionIR / RewriteRegion stay 100% op-id based.
    * Everything that crosses the boundary goes through the helpers below.

Three public functions:
    * ``expand_math_region_to_op_ids(view, node_ids)`` --
      MathNode set -> frozenset of underlying op_ids, including any J1-dropped
      jump op whose source block lies entirely in the region and whose unique
      successor block is also represented in the region.
    * ``project_op_ids_to_math_nodes(view, op_ids)`` --
      reverse direction; useful e.g. for translating ``slot_meta[k].op_ids``
      into a MathNode set for the GNN to reason over.
    * ``boundary_signature_for_math_region(view, node_ids)`` --
      builds a ``BoundarySignature`` directly from the MathNode region.
      By construction it is identical to the signature produced by
      ``signature_for_region(ir, define_rewrite_region(ir, op_ids=...))`` on
      the expanded op_id set, because we delegate to that exact pipeline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from algorithm_ir.ir.math_view import MathView
    from evolution.graft_classifier import BoundarySignature


__all__ = [
    "expand_math_region_to_op_ids",
    "project_op_ids_to_math_nodes",
    "boundary_signature_for_math_region",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _block_of_op(ir, op_id: str) -> str | None:
    """Return the block_id that contains ``op_id`` (or None)."""
    # FunctionIR.Op may have block_id directly.
    op = ir.ops.get(op_id)
    if op is not None:
        bid = getattr(op, "block_id", None)
        if bid is not None:
            return bid
    # Fallback scan
    for bid, blk in ir.blocks.items():
        if op_id in blk.op_ids:
            return bid
    return None


def _index_jump_target_block(ir, jump_op_id: str) -> str | None:
    """Given a jump op id, return the block_id of its successor."""
    # Source-block successor lookup is sufficient because J1 only fires when
    # the source block has exactly one successor.
    src_bid = _block_of_op(ir, jump_op_id)
    if src_bid is None:
        return None
    blk = ir.blocks.get(src_bid)
    if blk is None:
        return None
    succs = list(getattr(blk, "succs", []) or [])
    if len(succs) != 1:
        return None
    return succs[0]


def _b1_dropped_jumps_inside(view: "MathView", op_ids: set[str]) -> set[str]:
    """Find J1-dropped jump ops that semantically belong to ``op_ids``.

    A dropped jump ``j`` is considered inside the region iff:
      (a) the source block of ``j`` has *all* of its non-jump ops in
          ``op_ids`` (i.e. the region has fully consumed that block); and
      (b) the unique successor block has at least one op in ``op_ids``,
          OR the successor block is the source block itself (self-loop —
          theoretically impossible for J1 but guarded for safety).

    Rationale: if condition (a) fails, the jump still has live consumers
    (other ops in the source block) outside the region, so dropping it
    would leave that block headless. If condition (b) fails, the region
    does not need this jump's control-flow link.
    """
    ir = view.ir
    inside: set[str] = set()
    # view.dropped is a tuple of op_ids that build_math_view marked as
    # dropped. We only consider jumps among those.
    for op_id in view.dropped:
        op = ir.ops.get(op_id)
        if op is None or op.opcode != "jump":
            continue
        src_bid = _block_of_op(ir, op_id)
        if src_bid is None:
            continue
        src_blk = ir.blocks[src_bid]
        # All other (non-jump) ops in the source block must be in op_ids.
        other_ops_in_src = [oid for oid in src_blk.op_ids if oid != op_id]
        if other_ops_in_src and not all(oid in op_ids for oid in other_ops_in_src):
            continue
        # Successor block must be touched by the region (or be the same
        # block, which we skip explicitly).
        tgt_bid = _index_jump_target_block(ir, op_id)
        if tgt_bid is None:
            continue
        if tgt_bid == src_bid:
            inside.add(op_id)
            continue
        tgt_blk = ir.blocks.get(tgt_bid)
        if tgt_blk is None:
            continue
        if any(oid in op_ids for oid in tgt_blk.op_ids):
            inside.add(op_id)
    return inside


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_math_region_to_op_ids(
    view: "MathView", node_ids,
) -> frozenset[str]:
    """Translate a MathNode region into the underlying op_id set.

    Parameters
    ----------
    view : MathView
        The view that produced the node_ids.
    node_ids : Iterable[str]
        MathNode ids selected by the GNN. Boundary nodes (kind=='boundary'
        with no op_ids) are ignored silently because they correspond to
        function-level arguments / returns rather than ops.

    Returns
    -------
    frozenset[str]
        Underlying op_ids covering the entire region, including all
        op_ids absorbed inside each MathNode plus any J1-dropped jumps
        that semantically belong to the region per ``_b1_dropped_jumps_inside``.
    """
    nodes_by_id = {n.node_id: n for n in view.nodes}
    base: set[str] = set()
    for nid in node_ids:
        node = nodes_by_id.get(nid)
        if node is None:
            continue
        base.update(node.op_ids)
    base |= _b1_dropped_jumps_inside(view, base)
    return frozenset(base)


def project_op_ids_to_math_nodes(
    view: "MathView", op_ids,
) -> frozenset[str]:
    """Translate op_ids into the MathNode ids that own them.

    Notes
    -----
    Dropped op_ids (e.g. J1 jumps) have no node and are silently skipped;
    callers should use ``expand_math_region_to_op_ids`` for the round
    trip to recover them.
    """
    out: set[str] = set()
    for op_id in op_ids:
        nid = view.op_id_to_node.get(op_id)
        if nid is not None:
            out.add(nid)
    return frozenset(out)


def boundary_signature_for_math_region(
    view: "MathView", node_ids,
) -> "BoundarySignature":
    """Build the ``BoundarySignature`` of a MathNode region.

    Implementation delegates to the canonical op-id pipeline
    (``define_rewrite_region`` + ``signature_for_region``) on the
    expanded op_id set, so by construction the result is identical to
    what the existing graft pipeline would produce on the same op_id set.
    This is the safest possible correctness guarantee.
    """
    # Local imports to avoid heavy module-level dependencies (and to keep
    # the bridge importable from very early in the pipeline).
    from algorithm_ir.region.selector import define_rewrite_region
    from evolution.graft_classifier import signature_for_region

    op_ids = expand_math_region_to_op_ids(view, node_ids)
    if not op_ids:
        from evolution.graft_classifier import BoundarySignature
        return BoundarySignature(entry_types=(), exit_types=())
    region = define_rewrite_region(view.ir, op_ids=list(op_ids))
    return signature_for_region(view.ir, region)
