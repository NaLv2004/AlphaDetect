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
    "build_tuple",
    "build_dict",
    "build_slice",
    "append",
    "pop",
    "iter_init",
    "iter_next",
    "branch",
    "jump",
    "return",
    "slot",
}


def rebuild_def_use(func_ir: FunctionIR) -> FunctionIR:
    """Rebuild SSA def/use bookkeeping from ops.

    This function is intentionally strict and deterministic:
    - every output value gets exactly one ``def_op`` if the producing op exists
    - every input edge contributes exactly one entry to ``use_ops``
    - stale/missing bookkeeping is overwritten from structural truth
    """
    for value in func_ir.values.values():
        value.def_op = None
        value.use_ops = []

    for op_id, op in func_ir.ops.items():
        for value_id in op.outputs:
            value = func_ir.values.get(value_id)
            if value is not None:
                value.def_op = op_id

    for op_id, op in func_ir.ops.items():
        for value_id in op.inputs:
            value = func_ir.values.get(value_id)
            if value is not None:
                value.use_ops.append(op_id)

    return func_ir


def validate_def_use(func_ir: FunctionIR) -> list[str]:
    """Validate that def/use metadata matches the op graph exactly."""
    errors: list[str] = []

    expected_defs: dict[str, str | None] = {value_id: None for value_id in func_ir.values}
    expected_uses: dict[str, list[str]] = {value_id: [] for value_id in func_ir.values}

    for op_id, op in func_ir.ops.items():
        for value_id in op.outputs:
            if value_id not in func_ir.values:
                errors.append(f"Op {op_id} outputs missing value {value_id}")
                continue
            prev = expected_defs[value_id]
            if prev is not None and prev != op_id:
                errors.append(
                    f"Value {value_id} has multiple defining ops: {prev}, {op_id}"
                )
            expected_defs[value_id] = op_id
        for value_id in op.inputs:
            if value_id not in func_ir.values:
                errors.append(f"Op {op_id} inputs missing value {value_id}")
                continue
            expected_uses[value_id].append(op_id)

    for value_id, value in func_ir.values.items():
        if value.def_op != expected_defs[value_id]:
            errors.append(
                f"Value {value_id} def_op mismatch: stored={value.def_op} "
                f"expected={expected_defs[value_id]}"
            )
        if list(value.use_ops) != expected_uses[value_id]:
            errors.append(
                f"Value {value_id} use_ops mismatch: stored={list(value.use_ops)} "
                f"expected={expected_uses[value_id]}"
            )

    return errors


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

    errors.extend(validate_def_use(func_ir))
    return errors
