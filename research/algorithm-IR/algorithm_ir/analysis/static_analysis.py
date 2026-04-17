from __future__ import annotations

from algorithm_ir.ir.model import FunctionIR


def def_use_edges(func_ir: FunctionIR) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for op in func_ir.ops.values():
        for inp in op.inputs:
            def_op = func_ir.values[inp].def_op
            if def_op is not None:
                edges.add((def_op, op.id))
    return edges


def block_uses(func_ir: FunctionIR, block_id: str) -> set[str]:
    block = func_ir.blocks[block_id]
    used: set[str] = set()
    for op_id in block.op_ids:
        used.update(func_ir.ops[op_id].inputs)
    return used

