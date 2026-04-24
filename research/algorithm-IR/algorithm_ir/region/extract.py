from __future__ import annotations

from algorithm_ir.ir.model import Block, FunctionIR, Op, Value
from algorithm_ir.ir.validator import rebuild_def_use
from algorithm_ir.region.selector import RewriteRegion


def extract_region_ir(func_ir: FunctionIR, region: RewriteRegion) -> FunctionIR:
    """Extract an executable FunctionIR for exactly ``region``.

    The extracted function:
    - contains only region ops plus a synthetic return op
    - uses ``region.entry_values`` as function arguments
    - uses ``region.exit_values`` as function returns
    - performs no dependency expansion beyond the region closure
    """
    block_ids = list(dict.fromkeys(region.block_ids)) or [func_ir.entry_block]
    entry_block_id = block_ids[0]
    new_blocks: dict[str, Block] = {}
    new_ops: dict[str, Op] = {}

    for block_id in block_ids:
        orig_block = func_ir.blocks[block_id]
        selected_ops = [op_id for op_id in orig_block.op_ids if op_id in set(region.op_ids)]
        new_blocks[block_id] = Block(
            id=block_id,
            op_ids=list(selected_ops),
            preds=[pred for pred in orig_block.preds if pred in block_ids],
            succs=[succ for succ in orig_block.succs if succ in block_ids],
            attrs=dict(orig_block.attrs),
        )
        for op_id in selected_ops:
            op = func_ir.ops[op_id]
            new_ops[op_id] = Op(
                id=op.id,
                opcode=op.opcode,
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                block_id=block_id,
                source_span=op.source_span,
                attrs=dict(op.attrs),
            )

    return_block_id = block_ids[-1]
    return_op_id = f"{region.region_id}_return"
    new_ops[return_op_id] = Op(
        id=return_op_id,
        opcode="return",
        inputs=list(region.exit_values),
        outputs=[],
        block_id=return_block_id,
        attrs={"synthetic": True},
    )
    new_blocks[return_block_id].op_ids.append(return_op_id)

    kept_value_ids = set(region.entry_values)
    for op_id in region.op_ids:
        op = func_ir.ops[op_id]
        kept_value_ids.update(op.inputs)
        kept_value_ids.update(op.outputs)
    kept_value_ids.update(region.exit_values)

    new_values: dict[str, Value] = {}
    for value_id in kept_value_ids:
        orig = func_ir.values[value_id]
        new_values[value_id] = Value(
            id=orig.id,
            name_hint=orig.name_hint,
            type_hint=orig.type_hint,
            source_span=orig.source_span,
            def_op=None,
            use_ops=[],
            attrs=dict(orig.attrs),
        )

    extracted = FunctionIR(
        id=f"{func_ir.id}_extract_{region.region_id}",
        name=f"{func_ir.name}_extract",
        arg_values=list(region.entry_values),
        return_values=list(region.exit_values),
        values=new_values,
        ops=new_ops,
        blocks=new_blocks,
        entry_block=entry_block_id,
        attrs=dict(func_ir.attrs) if func_ir.attrs else {},
    )
    rebuild_def_use(extracted)
    return extracted
