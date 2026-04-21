from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.func import FuncOp


SourceSpan = tuple[int, int, int, int] | None


@dataclass
class Value:
    id: str
    name_hint: str | None = None
    type_hint: str | None = None
    source_span: SourceSpan = None
    def_op: str | None = None
    use_ops: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Op:
    id: str
    opcode: str
    inputs: list[str]
    outputs: list[str]
    block_id: str
    source_span: SourceSpan = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Block:
    id: str
    op_ids: list[str] = field(default_factory=list)
    preds: list[str] = field(default_factory=list)
    succs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


class FunctionIR:
    """
    Thin wrapper over an xDSL ModuleOp containing a single function.

    Provides dict-based access to values/ops/blocks for backward compat
    with the interpreter, region selector, grafting engine, etc.
    The native xDSL module is always the source of truth.
    """

    def __init__(
        self,
        *,
        id: str,
        name: str,
        arg_values: list[str],
        return_values: list[str],
        values: dict[str, Value],
        ops: dict[str, Op],
        blocks: dict[str, Block],
        entry_block: str,
        attrs: dict[str, Any] | None = None,
        xdsl_module=None,
        xdsl_func: FuncOp | None = None,
        xdsl_op_map: dict[str, Any] | None = None,
        xdsl_block_map: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.arg_values = arg_values
        self.return_values = return_values
        self.values = values
        self.ops = ops
        self.blocks = blocks
        self.entry_block = entry_block
        self.attrs = attrs or {}
        self.xdsl_module = xdsl_module
        self.xdsl_func = xdsl_func
        self.xdsl_op_map = xdsl_op_map or {}
        self.xdsl_block_map = xdsl_block_map or {}

    @classmethod
    def from_xdsl(cls, module) -> "FunctionIR":
        """
        Build a FunctionIR from an xDSL ModuleOp.

        Supports both the new typed dialect ops (AlgConst, AlgBinary, …)
        and legacy UnregisteredOp+payload ops (for backward compat during migration).
        """
        func = next(iter(module.ops))

        # Detect whether we're dealing with new-style or legacy ops
        # New-style: func has "alg_id" metadata attribute set by the new frontend
        from xdsl.dialects.builtin import StringAttr as _SA
        is_new_style = (
            func.attributes.get("alg_id") is not None
            and isinstance(func.attributes.get("alg_id"), _SA)
        )

        if is_new_style:
            return cls._from_typed_xdsl(module, func)
        else:
            return cls._from_legacy_xdsl(module, func)

    @classmethod
    def _from_typed_xdsl(cls, module, func) -> "FunctionIR":
        """Build FunctionIR from new typed AlgDialect ops."""
        from algorithm_ir.ir.dialect import (
            AlgAppend, AlgAssign, AlgBinary, AlgBranch, AlgBuildDict,
            AlgBuildList, AlgBuildTuple, AlgCall, AlgCompare, AlgConst,
            AlgGetAttr, AlgGetItem, AlgIterInit, AlgIterNext, AlgJump,
            AlgPhi, AlgPop, AlgReturn, AlgSetAttr, AlgSetItem, AlgSlot,
            AlgUnary,
        )
        from xdsl.dialects.builtin import StringAttr

        # Extract function-level metadata from func attributes
        func_meta = {}
        for key in ("alg_id", "alg_arg_values", "alg_return_values",
                     "alg_entry_block", "alg_filename", "alg_source"):
            attr = func.attributes.get(key)
            if attr is not None and isinstance(attr, StringAttr):
                func_meta[key] = attr.data

        function_id = func_meta.get("alg_id", f"fn:{func.sym_name.data}")
        arg_value_ids = func_meta.get("alg_arg_values", "").split(",") if func_meta.get("alg_arg_values") else []
        return_value_ids = func_meta.get("alg_return_values", "").split(",") if func_meta.get("alg_return_values") else []
        entry_block_id = func_meta.get("alg_entry_block", "")

        values: dict[str, Value] = {}
        ops: dict[str, Op] = {}
        blocks: dict[str, Block] = {}
        xdsl_op_map: dict[str, Any] = {}
        xdsl_block_map: dict[str, Any] = {}

        # Map xDSL Block objects to their alg block IDs
        block_obj_to_id: dict[int, str] = {}

        # First pass: discover blocks and their IDs from block label attrs
        for xdsl_block in func.body.blocks:
            block_id_attr = None
            # The block ID is stored on ops' "block_id" attribute
            block_ops = list(xdsl_block.ops)
            for op in block_ops:
                block_id_attr = _get_str_attr(op, "block_id")
                if block_id_attr:
                    break
                # Also check legacy payload for block_id
                try:
                    payload = _payload_from_operation(op)
                    block_id_attr = payload.get("block_id")
                    if block_id_attr:
                        break
                except Exception:
                    pass
            if block_id_attr is None:
                # Fallback: use object id
                block_id_attr = f"b_auto_{id(xdsl_block)}"
            block_obj_to_id[id(xdsl_block)] = block_id_attr
            xdsl_block_map[block_id_attr] = xdsl_block
            blocks[block_id_attr] = Block(id=block_id_attr)

        if not entry_block_id and blocks:
            entry_block_id = next(iter(blocks))

        # Build arg values
        for arg_id_str in arg_value_ids:
            if not arg_id_str:
                continue
            # Arg metadata stored in func attributes
            arg_meta_attr = func.attributes.get(f"alg_arg_{arg_id_str}")
            if arg_meta_attr is not None and isinstance(arg_meta_attr, StringAttr):
                import ast as py_ast
                meta = py_ast.literal_eval(arg_meta_attr.data)
                values[arg_id_str] = Value(
                    id=arg_id_str,
                    name_hint=meta.get("name_hint"),
                    type_hint=meta.get("type_hint"),
                    source_span=meta.get("source_span"),
                    attrs=_denormalize(meta.get("attrs", {})),
                )
            else:
                values[arg_id_str] = Value(id=arg_id_str)

        # Second pass: build ops and values from xDSL ops
        for xdsl_block in func.body.blocks:
            block_id = block_obj_to_id[id(xdsl_block)]
            for xdsl_op in xdsl_block.ops:
                # Try typed extraction first, then legacy
                op_data = _extract_op_data(xdsl_op, block_id, block_obj_to_id)
                if op_data is None:
                    # Might be a legacy UnregisteredOp from the rewriter
                    op_data = _extract_legacy_op_data(xdsl_op, block_id)
                if op_data is None:
                    continue
                op_id, op, output_values = op_data
                ops[op_id] = op
                xdsl_op_map[op_id] = xdsl_op
                blocks[block_id].op_ids.append(op_id)
                for v in output_values:
                    values[v.id] = v

        # Build def-use edges
        for op_id, op in ops.items():
            for value_id in op.inputs:
                if value_id in values:
                    values[value_id].use_ops.append(op_id)

        # Build block preds/succs from control flow ops
        for op_id, op in ops.items():
            if op.opcode == "branch":
                true_target = op.attrs.get("true")
                false_target = op.attrs.get("false")
                if true_target:
                    _link_blocks(blocks, op.block_id, true_target)
                if false_target:
                    _link_blocks(blocks, op.block_id, false_target)
            elif op.opcode == "jump":
                target = op.attrs.get("target")
                if target:
                    _link_blocks(blocks, op.block_id, target)

        func_attrs: dict[str, Any] = {}
        if func_meta.get("alg_filename"):
            func_attrs["filename"] = func_meta["alg_filename"]
        if func_meta.get("alg_source"):
            func_attrs["source"] = func_meta["alg_source"]
        func_attrs["xdsl_module"] = module

        # Generate xdsl_text
        from algorithm_ir.ir.xdsl_bridge import render_xdsl_module
        func_attrs["xdsl_text"] = render_xdsl_module(module)

        return cls(
            id=function_id,
            name=func.sym_name.data,
            arg_values=arg_value_ids,
            return_values=return_value_ids,
            values=values,
            ops=ops,
            blocks=blocks,
            entry_block=entry_block_id,
            attrs=func_attrs,
            xdsl_module=module,
            xdsl_func=func,
            xdsl_op_map=xdsl_op_map,
            xdsl_block_map=xdsl_block_map,
        )

    @classmethod
    def _from_legacy_xdsl(cls, module, func) -> "FunctionIR":
        """Build FunctionIR from legacy UnregisteredOp + payload ops."""
        import ast as py_ast
        func_payload = _payload_from_operation(func)

        values: dict[str, Value] = {}
        ops: dict[str, Op] = {}
        blocks: dict[str, Block] = {}
        xdsl_op_map: dict[str, Any] = {}
        xdsl_block_map: dict[str, Any] = {}
        block_obj_to_id: dict[int, str] = {}

        for block in func.body.blocks:
            block_ops = list(block.ops)
            label_op = block_ops[0]
            block_payload = _payload_from_operation(label_op)
            block_id = block_payload["id"]
            xdsl_block_map[block_id] = block
            block_obj_to_id[id(block)] = block_id
            blocks[block_id] = Block(
                id=block_id,
                preds=list(block_payload.get("preds", [])),
                succs=list(block_payload.get("succs", [])),
                attrs=dict(block_payload.get("attrs", {})),
            )

        for arg_meta in func_payload["arg_meta"]:
            values[arg_meta["id"]] = Value(
                id=arg_meta["id"],
                name_hint=arg_meta.get("name_hint"),
                type_hint=arg_meta.get("type_hint"),
                source_span=arg_meta.get("source_span"),
                attrs=dict(arg_meta.get("attrs", {})),
            )

        for block in func.body.blocks:
            block_ops = list(block.ops)
            block_id = block_obj_to_id[id(block)]
            for xdsl_op in block_ops[1:]:
                op_payload = _payload_from_operation(xdsl_op)
                op_id = op_payload["id"]
                op = Op(
                    id=op_id,
                    opcode=op_payload["opcode"],
                    inputs=list(op_payload["inputs"]),
                    outputs=list(op_payload["outputs"]),
                    block_id=block_id,
                    source_span=op_payload.get("source_span"),
                    attrs=dict(op_payload.get("attrs", {})),
                )
                ops[op_id] = op
                xdsl_op_map[op_id] = xdsl_op
                blocks[block_id].op_ids.append(op_id)
                for value_meta in op_payload.get("output_meta", []):
                    values[value_meta["id"]] = Value(
                        id=value_meta["id"],
                        name_hint=value_meta.get("name_hint"),
                        type_hint=value_meta.get("type_hint"),
                        source_span=value_meta.get("source_span"),
                        def_op=op_id,
                        attrs=dict(value_meta.get("attrs", {})),
                    )

        for op_id, op in ops.items():
            for value_id in op.inputs:
                if value_id in values:
                    values[value_id].use_ops.append(op_id)

        attrs = dict(func_payload.get("attrs", {}))
        attrs["xdsl_module"] = module
        from algorithm_ir.ir.xdsl_bridge import render_xdsl_module
        attrs["xdsl_text"] = render_xdsl_module(module)

        return cls(
            id=func_payload["id"],
            name=func_payload["name"],
            arg_values=list(func_payload["arg_values"]),
            return_values=list(func_payload["return_values"]),
            values=values,
            ops=ops,
            blocks=blocks,
            entry_block=func_payload["entry_block"],
            attrs=attrs,
            xdsl_module=module,
            xdsl_func=func,
            xdsl_op_map=xdsl_op_map,
            xdsl_block_map=xdsl_block_map,
        )

    def clone(self) -> "FunctionIR":
        """Deep-clone at the dict level so grafted ops are preserved.

        The previous implementation used ``from_xdsl(self.xdsl_module.clone())``
        which silently discarded any dict-level modifications (inlined ops,
        removed ops, etc.) that were not reflected back into the xDSL module.
        """
        return self._dict_level_clone()

    def __deepcopy__(self, memo):
        return self._dict_level_clone()

    def _dict_level_clone(self) -> "FunctionIR":
        """Clone by copying Python-level dicts (ops/values/blocks).

        This preserves all mutations made by graft_general() and other
        dict-level IR transformations, unlike from_xdsl() which only
        sees the original (stale) xDSL module.
        """
        from copy import deepcopy as _dc
        # We must avoid recursion: _dc on a Value/Op/Block won't trigger
        # FunctionIR.__deepcopy__ because they are plain dataclasses.
        memo: dict = {}  # fresh memo to avoid cross-contamination
        return FunctionIR(
            id=self.id,
            name=self.name,
            arg_values=list(self.arg_values),
            return_values=list(self.return_values),
            values={
                k: Value(
                    id=v.id,
                    name_hint=v.name_hint,
                    type_hint=v.type_hint,
                    source_span=v.source_span,
                    def_op=v.def_op,
                    use_ops=list(v.use_ops),
                    attrs=dict(v.attrs),
                )
                for k, v in self.values.items()
            },
            ops={
                k: Op(
                    id=o.id,
                    opcode=o.opcode,
                    inputs=list(o.inputs),
                    outputs=list(o.outputs),
                    block_id=o.block_id,
                    source_span=o.source_span,
                    attrs=dict(o.attrs),
                )
                for k, o in self.ops.items()
            },
            blocks={
                k: Block(
                    id=b.id,
                    op_ids=list(b.op_ids),
                    preds=list(b.preds),
                    succs=list(b.succs),
                    attrs=dict(b.attrs),
                )
                for k, b in self.blocks.items()
            },
            entry_block=self.entry_block,
            attrs=dict(self.attrs) if self.attrs else {},
            # xdsl_module is intentionally NOT cloned — it is stale
            # after dict-level edits and only kept for reference.
            xdsl_module=self.xdsl_module,
            xdsl_op_map=dict(self.xdsl_op_map) if self.xdsl_op_map else {},
            xdsl_block_map=dict(self.xdsl_block_map) if self.xdsl_block_map else {},
        )


@dataclass
class ModuleIR:
    functions: dict[str, FunctionIR]
    global_values: dict[str, Value]
    attrs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers for typed dialect ops
# ---------------------------------------------------------------------------

def _is_typed_dialect_op(op) -> bool:
    """Check if an op is a new-style typed AlgDialect op (not UnregisteredOp)."""
    return op.name.startswith("alg.") and not hasattr(op, '_unregistered_op_name')


def _get_str_attr(op, name: str) -> str | None:
    """Get a string attribute from an xDSL op, returning None if missing."""
    from xdsl.dialects.builtin import StringAttr
    attr = op.attributes.get(name)
    if attr is not None and isinstance(attr, StringAttr):
        return attr.data
    return None


def _extract_op_data(xdsl_op, block_id: str, block_obj_to_id: dict[int, str]):
    """
    Extract (op_id, Op, [Value, ...]) from a typed AlgDialect xDSL operation.
    Returns None if the op should be skipped.
    """
    from algorithm_ir.ir.dialect import (
        AlgAppend, AlgAssign, AlgBinary, AlgBranch, AlgBuildDict,
        AlgBuildList, AlgBuildTuple, AlgCall, AlgCompare, AlgConst,
        AlgGetAttr, AlgGetItem, AlgIterInit, AlgIterNext, AlgJump,
        AlgPhi, AlgPop, AlgReturn, AlgSetAttr, AlgSetItem, AlgSlot,
        AlgUnary,
    )
    from xdsl.dialects.builtin import StringAttr, IntegerAttr
    import ast as py_ast

    op_id = _get_str_attr(xdsl_op, "op_id")
    if op_id is None:
        return None

    source_span_str = _get_str_attr(xdsl_op, "source_span")
    source_span = py_ast.literal_eval(source_span_str) if source_span_str else None

    # Extra attrs stored as "alg_attrs" JSON string
    extra_attrs_str = _get_str_attr(xdsl_op, "alg_attrs")
    extra_attrs: dict = py_ast.literal_eval(extra_attrs_str) if extra_attrs_str else {}

    # Get output value metadata
    output_values: list[Value] = []
    output_ids: list[str] = []
    output_meta_str = _get_str_attr(xdsl_op, "output_meta")
    if output_meta_str:
        output_meta_list = py_ast.literal_eval(output_meta_str)
        for meta in output_meta_list:
            vid = meta["id"]
            output_ids.append(vid)
            raw_attrs = meta.get("attrs", {})
            output_values.append(Value(
                id=vid,
                name_hint=meta.get("name_hint"),
                type_hint=meta.get("type_hint"),
                source_span=meta.get("source_span"),
                def_op=op_id,
                attrs=_denormalize(raw_attrs),
            ))

    # Get input value IDs
    input_ids_str = _get_str_attr(xdsl_op, "input_ids")
    input_ids: list[str] = py_ast.literal_eval(input_ids_str) if input_ids_str else []

    # Determine opcode from op name
    opcode = xdsl_op.name.removeprefix("alg.")

    # Build attrs dict based on op type
    attrs: dict[str, Any] = dict(extra_attrs)

    if isinstance(xdsl_op, AlgConst):
        literal_str = _get_str_attr(xdsl_op, "alg_literal")
        if literal_str is not None:
            try:
                raw = py_ast.literal_eval(literal_str)
                attrs["literal"] = _denormalize(raw)
            except (ValueError, SyntaxError):
                attrs["literal"] = literal_str
        name_str = _get_str_attr(xdsl_op, "alg_name")
        if name_str:
            attrs["name"] = name_str
    elif isinstance(xdsl_op, AlgAssign):
        if xdsl_op.var_name is not None:
            attrs["target"] = xdsl_op.var_name.data
    elif isinstance(xdsl_op, AlgBinary):
        attrs["operator"] = xdsl_op.operator.data
    elif isinstance(xdsl_op, AlgUnary):
        attrs["operator"] = xdsl_op.operator.data
    elif isinstance(xdsl_op, AlgCompare):
        attrs["operators"] = xdsl_op.operators.data.split(",")
    elif isinstance(xdsl_op, AlgPhi):
        attrs["sources"] = xdsl_op.sources.data.split(",")
        if xdsl_op.var_name is not None:
            attrs["var_name"] = xdsl_op.var_name.data
    elif isinstance(xdsl_op, AlgCall):
        attrs["n_args"] = xdsl_op.n_args.value.data
    elif isinstance(xdsl_op, (AlgGetAttr, AlgSetAttr)):
        attrs["attr"] = xdsl_op.attr_name.data
    elif isinstance(xdsl_op, AlgBranch):
        for succ_block in xdsl_op.successors:
            succ_id = block_obj_to_id.get(id(succ_block))
            if succ_id:
                if "true" not in attrs:
                    attrs["true"] = succ_id
                else:
                    attrs["false"] = succ_id
    elif isinstance(xdsl_op, AlgJump):
        target_id = block_obj_to_id.get(id(xdsl_op.target))
        if target_id:
            attrs["target"] = target_id
        # Check for loop_backedge
        loop_backedge_str = _get_str_attr(xdsl_op, "loop_backedge")
        if loop_backedge_str == "true":
            attrs["loop_backedge"] = True
    elif isinstance(xdsl_op, AlgSlot):
        attrs["slot_id"] = xdsl_op.slot_id.data
        if xdsl_op.slot_kind is not None:
            attrs["slot_kind"] = xdsl_op.slot_kind.data

    op = Op(
        id=op_id,
        opcode=opcode,
        inputs=input_ids,
        outputs=output_ids,
        block_id=block_id,
        source_span=source_span,
        attrs=attrs,
    )
    return op_id, op, output_values


def _link_blocks(blocks: dict[str, Block], src: str, dst: str) -> None:
    if src in blocks and dst in blocks:
        if dst not in blocks[src].succs:
            blocks[src].succs.append(dst)
        if src not in blocks[dst].preds:
            blocks[dst].preds.append(src)


def _denormalize(value: Any) -> Any:
    """
    Recursively reconstruct callables/modules from their serialized dicts.
    Compatible with the __callable__ protocol from xdsl_bridge._denormalize_payload.
    """
    if isinstance(value, list):
        return [_denormalize(item) for item in value]
    if isinstance(value, dict):
        if "__callable__" in value:
            import importlib
            module_name = value.get("module", "builtins")
            attr_name = value["__callable__"]
            try:
                module = importlib.import_module(module_name)
                return getattr(module, attr_name)
            except Exception:
                return attr_name
        return {key: _denormalize(val) for key, val in value.items()}
    return value


# ---------------------------------------------------------------------------
# Helpers for legacy UnregisteredOp + payload ops
# ---------------------------------------------------------------------------

def _payload_from_attr(data: str) -> dict[str, Any]:
    import ast as py_ast
    from algorithm_ir.ir.xdsl_bridge import _denormalize_payload
    return _denormalize_payload(py_ast.literal_eval(data))


def _payload_from_operation(operation) -> dict[str, Any]:
    if hasattr(operation, "_alg_payload"):
        return getattr(operation, "_alg_payload")
    return _payload_from_attr(operation.attributes["payload"].data)


def _extract_legacy_op_data(xdsl_op, block_id: str):
    """
    Try extracting op data from an UnregisteredOp with payload attribute.
    Used for ops inserted by the rewriter during grafting.
    """
    import ast as py_ast
    try:
        payload = _payload_from_operation(xdsl_op)
    except (KeyError, AttributeError):
        return None

    op_id = payload.get("id")
    if op_id is None:
        return None

    output_values: list[Value] = []
    output_ids: list[str] = list(payload.get("outputs", []))
    for meta in payload.get("output_meta", []):
        vid = meta["id"]
        output_values.append(Value(
            id=vid,
            name_hint=meta.get("name_hint"),
            type_hint=meta.get("type_hint"),
            source_span=meta.get("source_span"),
            def_op=op_id,
            attrs=dict(meta.get("attrs", {})),
        ))

    op = Op(
        id=op_id,
        opcode=payload["opcode"],
        inputs=list(payload.get("inputs", [])),
        outputs=output_ids,
        block_id=block_id,
        source_span=payload.get("source_span"),
        attrs=dict(payload.get("attrs", {})),
    )
    return op_id, op, output_values
