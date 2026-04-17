from __future__ import annotations

from .model import FunctionIR


ALLOWED_OPCODES = {
    "const",
    "assign",
    "binary",
    "unary",
    "compare",
    "phi",
    "call",
    "get_attr",
    "set_attr",
    "get_item",
    "set_item",
    "build_list",
    "build_dict",
    "append",
    "pop",
    "iter_init",
    "iter_next",
    "branch",
    "jump",
    "return",
}


def validate_function_ir(func_ir: FunctionIR) -> list[str]:
    errors: list[str] = []
    if func_ir.entry_block not in func_ir.blocks:
        errors.append(f"Missing entry block {func_ir.entry_block}")

    for block_id, block in func_ir.blocks.items():
        for pred in block.preds:
            if pred not in func_ir.blocks:
                errors.append(f"Block {block_id} has unknown pred {pred}")
        for succ in block.succs:
            if succ not in func_ir.blocks:
                errors.append(f"Block {block_id} has unknown succ {succ}")
        for op_id in block.op_ids:
            if op_id not in func_ir.ops:
                errors.append(f"Block {block_id} references missing op {op_id}")

    for op_id, op in func_ir.ops.items():
        if op.opcode not in ALLOWED_OPCODES:
            errors.append(f"Unsupported opcode {op.opcode} for {op_id}")
        if op.block_id not in func_ir.blocks:
            errors.append(f"Op {op_id} references unknown block {op.block_id}")
        for value_id in op.inputs + op.outputs:
            if value_id not in func_ir.values:
                errors.append(f"Op {op_id} references missing value {value_id}")

    for value_id, value in func_ir.values.items():
        if value.def_op is not None and value.def_op not in func_ir.ops:
            errors.append(f"Value {value_id} references missing def op {value.def_op}")
        for use_op in value.use_ops:
            if use_op not in func_ir.ops:
                errors.append(f"Value {value_id} references missing use op {use_op}")

    return errors

