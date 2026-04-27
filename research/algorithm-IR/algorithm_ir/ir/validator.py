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
    errors.extend(validate_slot_meta(func_ir))
    return errors


def validate_slot_meta(func_ir: FunctionIR) -> list[str]:
    """Verify FunctionIR.slot_meta is internally consistent.

    Invariants:
      * every op tagged ``slot_id=K`` is recorded in ``slot_meta[K].op_ids``
        (innermost-tag rule);
      * every input value of slot K is defined OUTSIDE ``slot_full_op_ids(K)``
        OR is a function arg;
      * every output value of slot K is defined INSIDE ``slot_full_op_ids(K)``;
      * parent chain has no cycles, and every parent key exists in slot_meta.
    """
    errors: list[str] = []
    sm = getattr(func_ir, "slot_meta", None) or {}
    if not sm:
        return errors

    # Parent-chain integrity.
    for key, meta in sm.items():
        if meta.parent is not None and meta.parent not in sm:
            errors.append(f"slot {key} has unknown parent {meta.parent}")
    for key in sm:
        seen = {key}
        cur = sm[key].parent
        depth = 0
        while cur is not None:
            if cur in seen:
                errors.append(f"slot {key} has parent cycle through {cur}")
                break
            seen.add(cur)
            cur = sm.get(cur).parent if cur in sm else None
            depth += 1
            if depth > len(sm) + 1:
                errors.append(f"slot {key} parent chain too deep")
                break

    # Innermost-tag rule.
    tagged: dict[str, list[str]] = {}
    for oid, op in func_ir.ops.items():
        sid = op.attrs.get("slot_id") if op.attrs else None
        if sid:
            tagged.setdefault(sid, []).append(oid)
    for key, oids in tagged.items():
        if key not in sm:
            errors.append(f"op tagged slot_id={key} but slot_meta has no entry")
            continue
        declared = set(sm[key].op_ids)
        extra = set(oids) - declared
        missing = declared - set(oids)
        if extra:
            errors.append(
                f"slot {key}: ops {sorted(extra)} tagged but not in op_ids"
            )
        if missing:
            errors.append(
                f"slot {key}: op_ids contains {sorted(missing)} not present/tagged in IR"
            )

    arg_set = set(func_ir.arg_values)
    for key, meta in sm.items():
        full_ops = func_ir.slot_full_op_ids(key)
        # Inputs: defined outside or arg.
        for vid in meta.inputs:
            v = func_ir.values.get(vid)
            if v is None:
                errors.append(f"slot {key}: input value {vid} unknown")
                continue
            if v.def_op and v.def_op in full_ops:
                errors.append(
                    f"slot {key}: input {vid} defined inside slot by {v.def_op}"
                )
            elif v.def_op is None and vid not in arg_set:
                errors.append(
                    f"slot {key}: input {vid} has no def and is not a func arg"
                )
        # Outputs: defined inside.
        for vid in meta.outputs:
            v = func_ir.values.get(vid)
            if v is None:
                errors.append(f"slot {key}: output value {vid} unknown")
                continue
            if v.def_op is None or v.def_op not in full_ops:
                errors.append(
                    f"slot {key}: output {vid} not defined inside slot (def_op={v.def_op})"
                )
    return errors
