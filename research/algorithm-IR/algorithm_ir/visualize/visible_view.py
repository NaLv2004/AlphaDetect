"""GNN-visible IR view (Plan B visibility filter).

Provides :func:`build_visible_ir`, which returns a *new*
:class:`~algorithm_ir.ir.model.FunctionIR` containing only the ops
the GNN encoder actually sees (i.e. trivial ops — ``const``,
``get_attr``, ``assign``, trivial ``phi(x,x)`` — are removed).

The returned IR is purely a *visualization view*: it shares value /
arg / return identifiers with the source IR but is not a valid
executable IR (block.op_ids are trimmed; xDSL fields are empty).
"""
from __future__ import annotations

from algorithm_ir.ir.model import Block, FunctionIR
from algorithm_ir.region.triviality import is_trivial_op


def build_visible_ir(ir: FunctionIR) -> FunctionIR:
    """Return a filtered IR that drops trivial ops from each block.

    Values, function args and return values are preserved verbatim so
    the resulting IR can be passed to ``render_ir_dataflow`` without
    further changes.
    """
    visible_op_ids = {
        op_id for op_id, op in ir.ops.items() if not is_trivial_op(op, ir)
    }
    visible_ops = {op_id: ir.ops[op_id] for op_id in visible_op_ids}

    new_blocks: dict[str, Block] = {}
    for bid, block in ir.blocks.items():
        new_op_ids = [oid for oid in block.op_ids if oid in visible_op_ids]
        new_blocks[bid] = Block(
            id=block.id,
            op_ids=new_op_ids,
            preds=list(block.preds),
            succs=list(block.succs),
        )

    visible = FunctionIR(
        id=ir.id,
        name=ir.name + "_visible",
        arg_values=list(ir.arg_values),
        return_values=list(ir.return_values),
        values=dict(ir.values),
        ops=visible_ops,
        blocks=new_blocks,
        entry_block=ir.entry_block,
        attrs=dict(getattr(ir, "attrs", {}) or {}),
    )
    return visible


def visibility_stats(ir: FunctionIR) -> tuple[int, int, int, float]:
    """Return ``(total, visible, hidden, hidden_pct)`` for ``ir``."""
    total = len(ir.ops)
    if total == 0:
        return (0, 0, 0, 0.0)
    visible = sum(1 for op in ir.ops.values() if not is_trivial_op(op, ir))
    hidden = total - visible
    return (total, visible, hidden, 100.0 * hidden / total)
