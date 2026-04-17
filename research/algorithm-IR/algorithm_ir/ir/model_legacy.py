from __future__ import annotations

import ast as py_ast
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
        func = next(iter(module.ops))
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
        return FunctionIR.from_xdsl(self.xdsl_module.clone())

    def __deepcopy__(self, memo):
        return self.clone()


@dataclass
class ModuleIR:
    functions: dict[str, FunctionIR]
    global_values: dict[str, Value]
    attrs: dict[str, Any] = field(default_factory=dict)


def _payload_from_attr(data: str) -> dict[str, Any]:
    from algorithm_ir.ir.xdsl_bridge import _denormalize_payload

    return _denormalize_payload(py_ast.literal_eval(data))


def _payload_from_operation(operation) -> dict[str, Any]:
    if hasattr(operation, "_alg_payload"):
        return getattr(operation, "_alg_payload")
    return _payload_from_attr(operation.attributes["payload"].data)
