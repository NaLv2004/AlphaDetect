from __future__ import annotations

import importlib
from ast import literal_eval
from io import StringIO
from typing import Any

from xdsl.dialects.builtin import ModuleOp, StringAttr, f64, i1, i64
from xdsl.dialects.builtin import UnregisteredOp
from xdsl.dialects.func import FuncOp
from xdsl.ir import Block as XBlock
from xdsl.ir import Region
from xdsl.printer import Printer


def lower_legacy_function_to_xdsl(
    *,
    function_id: str,
    name: str,
    arg_values: list[str],
    return_values: list[str],
    values: dict[str, Any],
    ops: dict[str, Any],
    blocks: dict[str, Any],
    entry_block: str,
    attrs: dict[str, Any],
) -> ModuleOp:
    block_map = {block_id: XBlock() for block_id in blocks}
    for block_id, block in blocks.items():
        label_payload = {
            "id": block_id,
            "attrs": getattr(block, "attrs", {}),
            "preds": list(getattr(block, "preds", [])),
            "succs": list(getattr(block, "succs", [])),
        }
        label_op = UnregisteredOp.with_name("alg.block_label").create(
            attributes={"payload": StringAttr(repr(label_payload))}
        )
        setattr(label_op, "_alg_payload", label_payload)
        block_map[block_id].add_op(label_op)
        for op_id in block.op_ids:
            op = ops[op_id]
            op_cls = UnregisteredOp.with_name(f"alg.{op.opcode}")
            op_payload = {
                "id": op.id,
                "opcode": op.opcode,
                "block_id": op.block_id,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "source_span": op.source_span,
                "attrs": dict(op.attrs),
                "output_meta": [_value_payload(values[value_id]) for value_id in op.outputs],
            }
            successors = [block_map[successor] for successor in _successor_ids(op.opcode, op.attrs)]
            result_types = [_type_to_xdsl_attr(values[value_id].type_hint) for value_id in op.outputs]
            xdsl_op = op_cls.create(
                result_types=result_types,
                attributes={"payload": StringAttr(repr(_normalize_payload(op_payload)))},
                successors=successors,
            )
            setattr(xdsl_op, "_alg_payload", op_payload)
            block_map[block_id].add_op(xdsl_op)

    region = Region(list(block_map.values()))
    func_payload = {
        "id": function_id,
        "name": name,
        "arg_values": list(arg_values),
        "return_values": list(return_values),
        "entry_block": entry_block,
        "attrs": dict(attrs),
        "arg_meta": [_value_payload(values[value_id]) for value_id in arg_values],
    }
    arg_types = [_type_to_xdsl_attr(values[value_id].type_hint) for value_id in arg_values]
    return_types = [_type_to_xdsl_attr(values[value_id].type_hint) for value_id in return_values]
    func = FuncOp(name, (arg_types, return_types), region)
    func.attributes["payload"] = StringAttr(repr(_normalize_payload(func_payload)))
    setattr(func, "_alg_payload", func_payload)
    return ModuleOp([func], attributes={"legacy_function": StringAttr(name)})


def render_xdsl_module(module: ModuleOp) -> str:
    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_op(module)
    return stream.getvalue()


def create_xdsl_op_from_payload(
    *,
    opcode: str,
    payload: dict[str, Any],
    result_type_hints: list[str | None],
    successors: list[XBlock] | None = None,
):
    op_cls = UnregisteredOp.with_name(f"alg.{opcode}")
    xdsl_op = op_cls.create(
        result_types=[_type_to_xdsl_attr(type_hint) for type_hint in result_type_hints],
        attributes={"payload": StringAttr(repr(_normalize_payload(payload)))},
        successors=successors or [],
    )
    setattr(xdsl_op, "_alg_payload", payload)
    return xdsl_op


def payload_from_xdsl_attr(operation) -> dict[str, Any]:
    if hasattr(operation, "_alg_payload"):
        return getattr(operation, "_alg_payload")
    payload_attr = operation.attributes["payload"]
    return _denormalize_payload(literal_eval(payload_attr.data))


def _value_payload(value) -> dict[str, Any]:
    return {
        "id": value.id,
        "name_hint": value.name_hint,
        "type_hint": value.type_hint,
        "source_span": value.source_span,
        "attrs": dict(value.attrs),
    }


def _type_to_xdsl_attr(type_hint: str | None):
    if type_hint in {"bool"}:
        return i1
    if type_hint in {"int"}:
        return i64
    if type_hint in {"float"}:
        return f64
    if type_hint in {"complex", "tuple", "list", "dict", "object", "str", "none", None}:
        return StringAttr(type_hint or "object")
    return StringAttr(type_hint)


def _successor_ids(opcode: str, attrs: dict[str, Any]) -> list[str]:
    if opcode not in {"branch", "jump"}:
        return []
    targets = []
    for key in ("true", "false", "target"):
        target = attrs.get(key)
        if isinstance(target, str):
            targets.append(target)
    return targets


def _normalize_payload(value: Any):
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    if isinstance(value, tuple):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, list):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_payload(val) for key, val in value.items()}
    if callable(value):
        return {
            "__callable__": getattr(value, "__name__", repr(value)),
            "module": getattr(value, "__module__", "builtins"),
        }
    return repr(value)


def _denormalize_payload(value: Any):
    if isinstance(value, list):
        return [_denormalize_payload(item) for item in value]
    if isinstance(value, dict):
        if "__callable__" in value:
            module_name = value.get("module", "builtins")
            attr_name = value["__callable__"]
            try:
                module = importlib.import_module(module_name)
                return getattr(module, attr_name)
            except Exception:
                return attr_name
        return {key: _denormalize_payload(val) for key, val in value.items()}
    return value
