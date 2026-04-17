from __future__ import annotations

import ast
import builtins
import itertools
from dataclasses import dataclass
from types import SimpleNamespace
from types import FunctionType
from typing import Any

from algorithm_ir.frontend.ast_parser import ParsedFunction, parse_function, source_span
from algorithm_ir.frontend.cfg_builder import CFGBlock, link_blocks
from algorithm_ir.ir.model import Block, FunctionIR, Op, Value
from algorithm_ir.ir.type_info import (
    combine_binary_type_info,
    type_hint_from_info,
    type_info_for_python_value,
    unify_type_infos,
)
from algorithm_ir.ir.xdsl_bridge import lower_legacy_function_to_xdsl


SUPPORTED_AST = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.AugAssign,
    ast.Expr,
    ast.Return,
    ast.If,
    ast.While,
    ast.For,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
)


@dataclass
class BuildState:
    parsed: ParsedFunction
    values: dict[str, Value]
    ops: dict[str, Op]
    cfg: dict[str, CFGBlock]
    current_block: str
    entry_block: str
    return_values: list[str]
    global_constants: dict[str, str]
    name_versions: dict[str, int]
    name_env: dict[str, str]


class IRBuilder:
    def __init__(self, parsed: ParsedFunction):
        self.parsed = parsed
        self.value_counter = itertools.count()
        self.op_counter = itertools.count()
        self.block_counter = itertools.count()
        entry = self._new_block_id("entry")
        cfg = {entry: CFGBlock(id=entry)}
        self.state = BuildState(
            parsed=parsed,
            values={},
            ops={},
            cfg=cfg,
            current_block=entry,
            entry_block=entry,
            return_values=[],
            global_constants={},
            name_versions={},
            name_env={},
        )
        self.function_id = f"fn:{parsed.tree.name}"

    def build(self) -> FunctionIR:
        self._assert_supported(self.parsed.tree)
        arg_values: list[str] = []
        for arg in self.parsed.tree.args.args:
            value_id = self._new_value(
                name_hint=arg.arg,
                type_hint="object",
                source=source_span(arg),
                attrs={"role": "arg", "type_info": {"kind": "object"}},
            )
            self.state.name_versions[arg.arg] = 0
            self.state.name_env[arg.arg] = value_id
            arg_values.append(value_id)

        for stmt in self.parsed.tree.body:
            if self._block_terminated(self.state.current_block):
                unreachable = self._new_block("dead")
                self.state.current_block = unreachable
            self._compile_stmt(stmt)

        if not self._block_terminated(self.state.current_block):
            none_value = self._compile_constant(None, self.parsed.tree)
            self._emit("return", [none_value], [], self.parsed.tree, attrs={})
            self.state.return_values.append(none_value)

        blocks = {
            block_id: Block(
                id=cfg_block.id,
                op_ids=[
                    op_id
                    for op_id, op in self.state.ops.items()
                    if op.block_id == block_id
                ],
                preds=list(cfg_block.preds),
                succs=list(cfg_block.succs),
                attrs={},
            )
            for block_id, cfg_block in self.state.cfg.items()
        }

        xdsl_module = lower_legacy_function_to_xdsl(
            function_id=self.function_id,
            name=self.parsed.tree.name,
            arg_values=arg_values,
            return_values=self.state.return_values,
            values=self.state.values,
            ops=self.state.ops,
            blocks=blocks,
            entry_block=self.state.entry_block,
            attrs={"filename": self.parsed.filename, "source": self.parsed.source},
        )
        return FunctionIR.from_xdsl(xdsl_module)

    def _assert_supported(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.expr_context,
                    ast.operator,
                    ast.unaryop,
                    ast.cmpop,
                ),
            ):
                continue
            if not isinstance(child, SUPPORTED_AST):
                raise NotImplementedError(
                    f"Unsupported AST node {type(child).__name__} at {source_span(child)}"
                )

    def _compile_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            value_id = self._compile_expr(stmt.value)
            for target in stmt.targets:
                self._assign_target(target, value_id)
        elif isinstance(stmt, ast.AugAssign):
            current = self._compile_expr(stmt.target)
            rhs = self._compile_expr(stmt.value)
            result = self._emit_expr_op("binary", [current, rhs], stmt, {"operator": type(stmt.op).__name__})
            self._assign_target(stmt.target, result)
        elif isinstance(stmt, ast.Expr):
            self._compile_expr(stmt.value)
        elif isinstance(stmt, ast.Return):
            value_id = self._compile_expr(stmt.value) if stmt.value is not None else self._compile_constant(None, stmt)
            self._emit("return", [value_id], [], stmt, attrs={})
            self.state.return_values.append(value_id)
        elif isinstance(stmt, ast.If):
            self._compile_if(stmt)
        elif isinstance(stmt, ast.While):
            self._compile_while(stmt)
        elif isinstance(stmt, ast.For):
            self._compile_for(stmt)
        else:
            raise NotImplementedError(f"Unsupported statement {type(stmt).__name__}")

    def _compile_if(self, stmt: ast.If) -> None:
        cond = self._compile_expr(stmt.test)
        then_block = self._new_block("if_true")
        else_block = self._new_block("if_false")
        merge_block = self._new_block("if_merge")
        self._emit("branch", [cond], [], stmt, attrs={"true": then_block, "false": else_block})
        link_blocks(self.state.cfg, self.state.current_block, then_block)
        link_blocks(self.state.cfg, self.state.current_block, else_block)

        before_env = dict(self.state.name_env)
        before_versions = dict(self.state.name_versions)

        self.state.current_block = then_block
        self.state.name_env = dict(before_env)
        self.state.name_versions = dict(before_versions)
        for body_stmt in stmt.body:
            self._compile_stmt(body_stmt)
        if not self._block_terminated(self.state.current_block):
            self._emit("jump", [], [], stmt, attrs={"target": merge_block})
            link_blocks(self.state.cfg, self.state.current_block, merge_block)
        then_end = self.state.current_block
        then_env = dict(self.state.name_env)

        self.state.current_block = else_block
        self.state.name_env = dict(before_env)
        self.state.name_versions = dict(before_versions)
        for body_stmt in stmt.orelse:
            self._compile_stmt(body_stmt)
        if not self._block_terminated(self.state.current_block):
            self._emit("jump", [], [], stmt, attrs={"target": merge_block})
            link_blocks(self.state.cfg, self.state.current_block, merge_block)
        else_end = self.state.current_block
        else_env = dict(self.state.name_env)

        self.state.current_block = merge_block
        merged = dict(before_env)
        keys = set(before_env) | set(then_env) | set(else_env)
        for key in keys:
            t_val = then_env.get(key, before_env.get(key))
            e_val = else_env.get(key, before_env.get(key))
            if t_val != e_val and t_val is not None and e_val is not None:
                phi_type = unify_type_infos(self._value_type_info(t_val), self._value_type_info(e_val))
                phi_value = self._new_versioned_name(
                    key,
                    stmt,
                    type_hint=type_hint_from_info(phi_type),
                    attrs={"type_info": phi_type.to_dict()},
                )
                self._emit("phi", [t_val, e_val], [phi_value], stmt, attrs={"sources": [then_end, else_end]})
                merged[key] = phi_value
            elif t_val is not None:
                merged[key] = t_val
            elif e_val is not None:
                merged[key] = e_val
        self.state.name_env = merged

    def _compile_while(self, stmt: ast.While) -> None:
        before_env = dict(self.state.name_env)
        test_block = self._new_block("while_test")
        body_block = self._new_block("while_body")
        exit_block = self._new_block("while_exit")
        self._emit("jump", [], [], stmt, attrs={"target": test_block})
        link_blocks(self.state.cfg, self.state.current_block, test_block)

        self.state.current_block = test_block
        loop_phi_inputs: dict[str, tuple[str, str]] = {}
        self.state.name_env = dict(before_env)
        for name, incoming_value in before_env.items():
            phi_type = unify_type_infos(self._value_type_info(incoming_value))
            phi_out = self._new_versioned_name(
                name,
                stmt,
                type_hint=type_hint_from_info(phi_type),
                attrs={"type_info": phi_type.to_dict()},
            )
            phi_op_id = self._emit(
                "phi",
                [incoming_value, incoming_value],
                [phi_out],
                stmt,
                attrs={"sources": ["loop_entry", "loop_backedge"], "loop_phi": True, "var_name": name},
            )
            loop_phi_inputs[name] = (phi_op_id, incoming_value)
            self.state.name_env[name] = phi_out
        cond = self._compile_expr(stmt.test)
        self._emit("branch", [cond], [], stmt, attrs={"true": body_block, "false": exit_block})
        link_blocks(self.state.cfg, test_block, body_block)
        link_blocks(self.state.cfg, test_block, exit_block)

        self.state.current_block = body_block
        for body_stmt in stmt.body:
            self._compile_stmt(body_stmt)
        body_env = dict(self.state.name_env)
        for name, (phi_op_id, entry_value) in loop_phi_inputs.items():
            backedge_value = body_env.get(name, entry_value)
            phi_op = self.state.ops[phi_op_id]
            if backedge_value == phi_op.outputs[0]:
                backedge_value = entry_value
            phi_op.inputs[1] = backedge_value
            self.state.values[backedge_value].use_ops.append(phi_op_id)
        if not self._block_terminated(self.state.current_block):
            self._emit("jump", [], [], stmt, attrs={"target": test_block, "loop_backedge": True})
            link_blocks(self.state.cfg, self.state.current_block, test_block)

        self.state.current_block = exit_block
        self.state.name_env = {
            name: self.state.ops[phi_op_id].outputs[0]
            for name, (phi_op_id, _) in loop_phi_inputs.items()
        }

    def _compile_for(self, stmt: ast.For) -> None:
        iter_value = self._compile_expr(stmt.iter)
        iter_handle = self._emit_expr_op("iter_init", [iter_value], stmt, {})
        test_block = self._new_block("for_test")
        body_block = self._new_block("for_body")
        exit_block = self._new_block("for_exit")
        self._emit("jump", [], [], stmt, attrs={"target": test_block})
        link_blocks(self.state.cfg, self.state.current_block, test_block)

        self.state.current_block = test_block
        next_value = self._new_value("iter_next", "object", source_span(stmt), {})
        has_next = self._new_value("iter_has_next", "bool", source_span(stmt), {})
        self._emit("iter_next", [iter_handle], [next_value, has_next], stmt, attrs={})
        self._assign_target(stmt.target, next_value)
        self._emit("branch", [has_next], [], stmt, attrs={"true": body_block, "false": exit_block})
        link_blocks(self.state.cfg, test_block, body_block)
        link_blocks(self.state.cfg, test_block, exit_block)

        self.state.current_block = body_block
        for body_stmt in stmt.body:
            self._compile_stmt(body_stmt)
        if not self._block_terminated(self.state.current_block):
            self._emit("jump", [], [], stmt, attrs={"target": test_block, "loop_backedge": True})
            link_blocks(self.state.cfg, self.state.current_block, test_block)
        self.state.current_block = exit_block

    def _assign_target(self, target: ast.AST, value_id: str) -> None:
        if isinstance(target, ast.Name):
            source_value = self.state.values[value_id]
            propagated_attrs: dict[str, Any] = {}
            if "type_info" in source_value.attrs:
                propagated_attrs["type_info"] = source_value.attrs["type_info"]
            new_value = self._new_versioned_name(
                target.id,
                target,
                type_hint=source_value.type_hint or "object",
                attrs=propagated_attrs,
            )
            self._emit("assign", [value_id], [new_value], target, attrs={"target": target.id})
            self.state.name_env[target.id] = new_value
        elif isinstance(target, ast.Attribute):
            owner = self._compile_expr(target.value)
            self._emit("set_attr", [owner, value_id], [], target, attrs={"attr": target.attr})
        elif isinstance(target, ast.Subscript):
            owner = self._compile_expr(target.value)
            index = self._compile_expr(target.slice)
            self._emit("set_item", [owner, index, value_id], [], target, attrs={})
        else:
            raise NotImplementedError(f"Unsupported assignment target {type(target).__name__}")

    def _compile_expr(self, expr: ast.AST) -> str:
        if isinstance(expr, ast.Name):
            if expr.id in self.state.name_env:
                return self.state.name_env[expr.id]
            return self._load_global(expr.id, expr)
        if isinstance(expr, ast.Constant):
            return self._compile_constant(expr.value, expr)
        if isinstance(expr, ast.BinOp):
            lhs = self._compile_expr(expr.left)
            rhs = self._compile_expr(expr.right)
            result_type = combine_binary_type_info(
                type(expr.op).__name__,
                self._value_type_info(lhs),
                self._value_type_info(rhs),
            )
            return self._emit_expr_op(
                "binary",
                [lhs, rhs],
                expr,
                {"operator": type(expr.op).__name__, "type_info": result_type.to_dict()},
                type_hint_from_info(result_type),
            )
        if isinstance(expr, ast.UnaryOp):
            operand = self._compile_expr(expr.operand)
            operand_type = self._value_type_info(operand)
            return self._emit_expr_op(
                "unary",
                [operand],
                expr,
                {"operator": type(expr.op).__name__, "type_info": operand_type.to_dict()},
                type_hint_from_info(operand_type),
            )
        if isinstance(expr, ast.Compare):
            left = self._compile_expr(expr.left)
            rights = [self._compile_expr(comp) for comp in expr.comparators]
            attrs = {"operators": [type(op).__name__ for op in expr.ops]}
            return self._emit_expr_op("compare", [left] + rights, expr, attrs, "bool")
        if isinstance(expr, ast.Call):
            func_value = self._compile_expr(expr.func)
            args = [self._compile_expr(arg) for arg in expr.args]
            return self._emit_expr_op("call", [func_value] + args, expr, {"n_args": len(args)})
        if isinstance(expr, ast.Attribute):
            owner = self._compile_expr(expr.value)
            return self._emit_expr_op("get_attr", [owner], expr, {"attr": expr.attr})
        if isinstance(expr, ast.Subscript):
            owner = self._compile_expr(expr.value)
            index = self._compile_expr(expr.slice)
            return self._emit_expr_op("get_item", [owner, index], expr, {})
        if isinstance(expr, ast.List):
            items = [self._compile_expr(item) for item in expr.elts]
            elem_type = unify_type_infos(*(self._value_type_info(item) for item in items))
            return self._emit_expr_op(
                "build_list",
                items,
                expr,
                {"n_items": len(items), "type_info": {"kind": "list", "elem": elem_type.to_dict(), "shape": [len(items)]}},
                "list",
            )
        if isinstance(expr, ast.Tuple):
            items = [self._compile_expr(item) for item in expr.elts]
            elem_type = unify_type_infos(*(self._value_type_info(item) for item in items))
            return self._emit_expr_op(
                "build_tuple",
                items,
                expr,
                {
                    "n_items": len(items),
                    "type_info": {
                        "kind": "tuple",
                        "elem": elem_type.to_dict(),
                        "arity": len(items),
                        "shape": [len(items)],
                    },
                },
                "tuple",
            )
        if isinstance(expr, ast.Dict):
            inputs: list[str] = []
            key_types = []
            value_types = []
            for key, value in zip(expr.keys, expr.values):
                key_id = self._compile_expr(key)
                value_id = self._compile_expr(value)
                inputs.append(key_id)
                inputs.append(value_id)
                key_types.append(self._value_type_info(key_id))
                value_types.append(self._value_type_info(value_id))
            key_type = unify_type_infos(*key_types)
            value_type = unify_type_infos(*value_types)
            return self._emit_expr_op(
                "build_dict",
                inputs,
                expr,
                {
                    "n_items": len(expr.keys),
                    "type_info": {
                        "kind": "dict",
                        "key": key_type.to_dict(),
                        "value": value_type.to_dict(),
                        "shape": [len(expr.keys)],
                    },
                },
                "dict",
            )
        raise NotImplementedError(f"Unsupported expression {type(expr).__name__}")

    def _compile_constant(self, value: Any, node: ast.AST) -> str:
        type_info = type_info_for_python_value(value)
        type_name = type_hint_from_info(type_info)
        out = self._new_value(
            name_hint=f"const_{type_name}",
            type_hint=type_name,
            source=source_span(node),
            attrs={"literal": value, "type_info": type_info.to_dict()},
        )
        self._emit("const", [], [out], node, attrs={"literal": value})
        return out

    def _emit_expr_op(
        self,
        opcode: str,
        inputs: list[str],
        node: ast.AST,
        attrs: dict[str, Any],
        type_hint: str = "object",
    ) -> str:
        out = self._new_value(
            name_hint=opcode,
            type_hint=type_hint,
            source=source_span(node),
            attrs={"type_info": attrs["type_info"]} if "type_info" in attrs else {},
        )
        self._emit(opcode, inputs, [out], node, attrs=attrs)
        return out

    def _emit(
        self,
        opcode: str,
        inputs: list[str],
        outputs: list[str],
        node: ast.AST,
        attrs: dict[str, Any],
    ) -> str:
        op_id = f"op_{next(self.op_counter)}"
        op = Op(
            id=op_id,
            opcode=opcode,
            inputs=list(inputs),
            outputs=list(outputs),
            block_id=self.state.current_block,
            source_span=source_span(node),
            attrs=dict(attrs),
        )
        self.state.ops[op_id] = op
        for out in outputs:
            self.state.values[out].def_op = op_id
        for inp in inputs:
            self.state.values[inp].use_ops.append(op_id)
        return op_id

    def _new_value(
        self,
        name_hint: str | None,
        type_hint: str | None,
        source: tuple[int, int, int, int] | None,
        attrs: dict[str, Any],
    ) -> str:
        value_id = f"v_{next(self.value_counter)}"
        self.state.values[value_id] = Value(
            id=value_id,
            name_hint=name_hint,
            type_hint=type_hint,
            source_span=source,
            attrs=dict(attrs),
        )
        return value_id

    def _new_versioned_name(
        self,
        name: str,
        node: ast.AST,
        *,
        type_hint: str = "object",
        attrs: dict[str, Any] | None = None,
    ) -> str:
        version = self.state.name_versions.get(name, -1) + 1
        self.state.name_versions[name] = version
        payload = {"var_name": name, "version": version}
        if attrs:
            payload.update(attrs)
        return self._new_value(
            name_hint=f"{name}_{version}",
            type_hint=type_hint,
            source=source_span(node),
            attrs=payload,
        )

    def _value_type_info(self, value_id: str):
        value = self.state.values[value_id]
        payload = value.attrs.get("type_info")
        if isinstance(payload, dict):
            from algorithm_ir.ir.type_info import type_info_from_dict

            return type_info_from_dict(payload)
        if value.type_hint in _TYPE_SENTINELS:
            return type_info_for_python_value(_TYPE_SENTINELS[value.type_hint])
        from algorithm_ir.ir.type_info import TypeInfo

        return TypeInfo("object")

    def _load_global(self, name: str, node: ast.AST) -> str:
        if name in self.state.global_constants:
            return self.state.global_constants[name]
        if name in self.parsed.globals_dict:
            obj = self.parsed.globals_dict[name]
        elif hasattr(builtins, name):
            obj = getattr(builtins, name)
        else:
            raise NameError(f"Unknown name {name} at {source_span(node)}")
        value_id = self._new_value(
            name_hint=name,
            type_hint=type(obj).__name__,
            source=source_span(node),
            attrs={"literal": obj, "scope": "global"},
        )
        self._emit("const", [], [value_id], node, attrs={"literal": obj, "name": name})
        self.state.global_constants[name] = value_id
        return value_id

    def _new_block_id(self, prefix: str) -> str:
        return f"b_{prefix}_{next(self.block_counter)}"

    def _new_block(self, prefix: str) -> str:
        block_id = self._new_block_id(prefix)
        self.state.cfg[block_id] = CFGBlock(id=block_id)
        return block_id

    def _block_terminated(self, block_id: str) -> bool:
        for op in self.state.ops.values():
            if op.block_id == block_id and op.opcode in {"return", "branch", "jump"}:
                return True
        return False


def compile_function_to_ir(fn: FunctionType) -> FunctionIR:
    parsed = parse_function(fn)
    return IRBuilder(parsed).build()


_TYPE_SENTINELS = {
    "bool": False,
    "int": 0,
    "float": 0.0,
    "complex": 0j,
    "str": "",
    "list": [],
    "dict": {},
    "tuple": (),
    "none": None,
}
