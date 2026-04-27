"""
Frontend compiler: Python function → xDSL-native AlgDialect IR.

Fixes over the original ir_builder.py:
  - for-loop now creates phi nodes for modified variables (backedge update)
  - Function parameter type annotations are parsed (int→i64, float→f64, etc.)
  - Module-level attribute access works (e.g. math.sqrt)
"""
from __future__ import annotations

import ast
import builtins
import itertools
import types as pytypes
from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64
from xdsl.dialects.func import FuncOp
from xdsl.ir import Block as XBlock, Region

from algorithm_ir.frontend.ast_parser import ParsedFunction, parse_function, source_span
from algorithm_ir.ir.dialect import (
    AlgAppend,
    AlgAssign,
    AlgBinary,
    AlgBranch,
    AlgBuildDict,
    AlgBuildList,
    AlgBuildTuple,
    AlgCall,
    AlgCompare,
    AlgConst,
    AlgGetAttr,
    AlgGetItem,
    AlgIterInit,
    AlgIterNext,
    AlgJump,
    AlgPhi,
    AlgPop,
    AlgReturn,
    AlgSetAttr,
    AlgSetItem,
    AlgUnary,
)
from algorithm_ir.ir.model import Block, FunctionIR, Op, Value
from algorithm_ir.ir.type_info import (
    TypeInfo,
    combine_binary_type_info,
    type_hint_from_info,
    type_info_for_python_value,
    unify_type_infos,
)
from algorithm_ir.ir.types import AlgType


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
    ast.keyword,
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
    # Extended (S1.0): expression-level control flow and slicing
    ast.IfExp,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Slice,
    # Slot DSL
    ast.With,
    ast.withitem,
)

# Python type annotation name → IR type hint
_ANNOTATION_TYPE_MAP = {
    "int": "int",
    "float": "float",
    "bool": "bool",
    "str": "str",
    "complex": "cx",
    "list": "list<any>",
    "dict": "dict<any>",
    "tuple": "tuple<any>",
    "None": "void",
    # numpy / scipy / typing aliases commonly used in annotations
    "ndarray": "any",
    "NDArray": "any",
    "ArrayLike": "any",
    "Array": "any",
    "Matrix": "mat_cx",
    "Vector": "vec_cx",
    "Scalar": "float",
}

# Heuristic prior on argument name → lattice type, applied when an argument
# has no explicit annotation.  This reflects the conventions used across
# the detector pool (H = channel matrix, y = received vector, sigma2 =
# noise variance, constellation = symbol alphabet, …).  When a name does
# not match the prior, the argument falls back to ``any`` (top of the
# lattice) so downstream lattice ops do not get poisoned by ``object``.
_ARG_NAME_TYPE_PRIOR: dict[str, str] = {
    # Channel / matrix-shaped inputs
    "H": "mat_cx", "Heq": "mat_cx", "H_eq": "mat_cx",
    "G": "mat_cx", "G_inv": "mat_cx", "Ginv": "mat_cx",
    "A": "mat_cx", "B": "mat_cx", "M": "mat_cx",
    "Q": "mat_cx", "R": "mat_cx", "L": "mat_cx", "U": "mat_cx",
    "P": "mat_cx", "Sigma": "mat_cx", "Cov": "mat_cx",
    "F": "mat_cx", "K": "mat_cx",
    # Received / transmitted / estimate vectors
    "y": "vec_cx", "x": "vec_cx", "x_hat": "vec_cx",
    "z": "vec_cx", "r": "vec_cx", "s": "vec_cx",
    "b": "vec_cx", "v": "vec_cx", "w": "vec_cx",
    "x0": "vec_cx", "x_init": "vec_cx",
    # Real-valued vectors
    "p": "vec_f", "d": "vec_f", "g": "vec_f",
    "alpha": "vec_f", "beta": "vec_f", "gamma": "vec_f",
    # Noise / scalar parameters
    "sigma2": "float", "sigma": "float", "noise_var": "float",
    "snr": "float", "snr_db": "float", "tol": "float", "lr": "float",
    "step": "float", "lambda_": "float", "lam": "float",
    "n_iter": "int", "iters": "int", "max_iter": "int",
    "k": "int", "K": "int", "Nt": "int", "Nr": "int", "N": "int",
    # Constellations / alphabets
    "constellation": "vec_cx", "symbols": "vec_cx",
    "alphabet": "vec_cx", "modulation": "vec_cx",
    # Probability / soft information
    "p_table": "prob_table", "probs": "prob_table",
    "llr": "vec_f", "soft": "vec_f",
}


@dataclass
class BuildState:
    parsed: ParsedFunction
    values: dict[str, Value]
    ops: dict[str, Op]
    # Block management (legacy dict + xDSL blocks)
    blocks: dict[str, Block]       # legacy Block objects for FunctionIR compat
    xdsl_blocks: dict[str, XBlock] # actual xDSL blocks
    current_block: str
    entry_block: str
    return_values: list[str]
    global_constants: dict[str, str]
    name_versions: dict[str, int]
    name_env: dict[str, str]
    # Slot stack (innermost last). Each frame: (pop_key, entry_value_ids,
    # output_names, op_ids_collected, parent_pop_key)
    slot_stack: list[dict[str, Any]] = field(default_factory=list)
    # Completed slots (pop_key -> SlotMeta-shaped dict). Order preserved.
    slot_meta: dict[str, dict[str, Any]] = field(default_factory=dict)


class IRBuilder:
    def __init__(self, parsed: ParsedFunction):
        self.parsed = parsed
        self.value_counter = itertools.count()
        self.op_counter = itertools.count()
        self.block_counter = itertools.count()
        entry = self._new_block_id("entry")
        self.state = BuildState(
            parsed=parsed,
            values={},
            ops={},
            blocks={entry: Block(id=entry)},
            xdsl_blocks={entry: XBlock()},
            current_block=entry,
            entry_block=entry,
            return_values=[],
            global_constants={},
            name_versions={},
            name_env={},
        )
        self.function_id = f"fn:{parsed.tree.name}"
        # Map op_id → xDSL op for xdsl_op_map
        self._xdsl_op_map: dict[str, Any] = {}

    def build(self) -> FunctionIR:
        self._assert_supported(self.parsed.tree)
        arg_values: list[str] = []
        for arg in self.parsed.tree.args.args:
            type_hint = self._resolve_annotation(arg)
            type_info = TypeInfo(type_hint)
            value_id = self._new_value(
                name_hint=arg.arg,
                type_hint=type_hint,
                source=source_span(arg),
                attrs={"role": "arg", "type_info": type_info.to_dict(), "var_name": arg.arg, "version": 0},
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
            self._emit_return(none_value, self.parsed.tree)
            self.state.return_values.append(none_value)

        # Build the xDSL module with typed ops
        xdsl_module = self._build_xdsl_module(arg_values)

        # Build FunctionIR from our state (not from xdsl roundtrip)
        return self._build_function_ir(arg_values, xdsl_module)

    def _resolve_annotation(self, arg: ast.arg) -> str:
        """Parse type annotation from ast.arg, return type hint string.

        When no annotation is present, fall back to the argument-name
        prior (`_ARG_NAME_TYPE_PRIOR`) so that detector pool functions
        without type hints still get useful lattice tags on their inputs.
        """
        if arg.annotation is None:
            return _ARG_NAME_TYPE_PRIOR.get(arg.arg, "any")
        if isinstance(arg.annotation, ast.Name):
            return _ANNOTATION_TYPE_MAP.get(arg.annotation.id, "any")
        if isinstance(arg.annotation, ast.Constant):
            return _ANNOTATION_TYPE_MAP.get(str(arg.annotation.value), "any")
        if isinstance(arg.annotation, ast.Attribute):
            # e.g. typing.List → just "list"
            return _ANNOTATION_TYPE_MAP.get(arg.annotation.attr, "any")
        if isinstance(arg.annotation, ast.Subscript):
            # e.g. list[float] → "list"
            if isinstance(arg.annotation.value, ast.Name):
                return _ANNOTATION_TYPE_MAP.get(arg.annotation.value.id, "any")
        return "any"

    def _assert_supported(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop)):
                continue
            if not isinstance(child, SUPPORTED_AST):
                raise NotImplementedError(
                    f"Unsupported AST node {type(child).__name__} at {source_span(child)}"
                )

    # ------------------------------------------------------------------
    # Statement compilation
    # ------------------------------------------------------------------

    def _compile_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            value_id = self._compile_expr(stmt.value)
            for target in stmt.targets:
                self._assign_target(target, value_id)
        elif isinstance(stmt, ast.AugAssign):
            current = self._compile_expr(stmt.target)
            rhs = self._compile_expr(stmt.value)
            result = self._emit_binary(current, rhs, type(stmt.op).__name__, stmt)
            self._assign_target(stmt.target, result)
        elif isinstance(stmt, ast.Expr):
            self._compile_expr(stmt.value)
        elif isinstance(stmt, ast.Return):
            value_id = self._compile_expr(stmt.value) if stmt.value is not None else self._compile_constant(None, stmt)
            self._emit_return(value_id, stmt)
            self.state.return_values.append(value_id)
        elif isinstance(stmt, ast.If):
            self._compile_if(stmt)
        elif isinstance(stmt, ast.While):
            self._compile_while(stmt)
        elif isinstance(stmt, ast.For):
            self._compile_for(stmt)
        elif isinstance(stmt, ast.With):
            self._compile_with(stmt)
        else:
            raise NotImplementedError(f"Unsupported statement {type(stmt).__name__}")

    def _compile_if(self, stmt: ast.If) -> None:
        cond = self._compile_expr(stmt.test)
        then_block = self._new_block("if_true")
        else_block = self._new_block("if_false")
        merge_block = self._new_block("if_merge")
        self._emit_branch(cond, then_block, else_block, stmt)

        before_env = dict(self.state.name_env)
        before_versions = dict(self.state.name_versions)

        # Compile then branch
        self.state.current_block = then_block
        self.state.name_env = dict(before_env)
        self.state.name_versions = dict(before_versions)
        for body_stmt in stmt.body:
            self._compile_stmt(body_stmt)
        if not self._block_terminated(self.state.current_block):
            self._emit_jump(merge_block, stmt)
        then_end = self.state.current_block
        then_env = dict(self.state.name_env)

        # Compile else branch
        self.state.current_block = else_block
        self.state.name_env = dict(before_env)
        self.state.name_versions = dict(before_versions)
        for body_stmt in stmt.orelse:
            self._compile_stmt(body_stmt)
        if not self._block_terminated(self.state.current_block):
            self._emit_jump(merge_block, stmt)
        else_end = self.state.current_block
        else_env = dict(self.state.name_env)

        # Merge with phi nodes
        self.state.current_block = merge_block
        merged = dict(before_env)
        keys = set(before_env) | set(then_env) | set(else_env)
        for key in keys:
            t_val = then_env.get(key, before_env.get(key))
            e_val = else_env.get(key, before_env.get(key))
            if t_val != e_val and t_val is not None and e_val is not None:
                phi_type = unify_type_infos(self._value_type_info(t_val), self._value_type_info(e_val))
                phi_value = self._new_versioned_name(
                    key, stmt,
                    type_hint=type_hint_from_info(phi_type),
                    attrs={"type_info": phi_type.to_dict()},
                )
                self._emit_phi([t_val, e_val], phi_value, [then_end, else_end], stmt, var_name=key)
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
        self._emit_jump(test_block, stmt)

        self.state.current_block = test_block
        loop_phi_inputs: dict[str, tuple[str, str]] = {}  # name -> (phi_op_id, entry_value)
        self.state.name_env = dict(before_env)
        for name, incoming_value in before_env.items():
            phi_type = unify_type_infos(self._value_type_info(incoming_value))
            phi_out = self._new_versioned_name(
                name, stmt,
                type_hint=type_hint_from_info(phi_type),
                attrs={"type_info": phi_type.to_dict()},
            )
            phi_op_id = self._emit_phi(
                [incoming_value, incoming_value], phi_out,
                ["loop_entry", "loop_backedge"], stmt,
                var_name=name, loop_phi=True,
            )
            loop_phi_inputs[name] = (phi_op_id, incoming_value)
            self.state.name_env[name] = phi_out

        cond = self._compile_expr(stmt.test)
        self._emit_branch(cond, body_block, exit_block, stmt)

        self.state.current_block = body_block
        for body_stmt in stmt.body:
            self._compile_stmt(body_stmt)
        body_env = dict(self.state.name_env)
        # Update phi backedges
        for name, (phi_op_id, entry_value) in loop_phi_inputs.items():
            backedge_value = body_env.get(name, entry_value)
            phi_op = self.state.ops[phi_op_id]
            if backedge_value == phi_op.outputs[0]:
                backedge_value = entry_value
            phi_op.inputs[1] = backedge_value
            self.state.values[backedge_value].use_ops.append(phi_op_id)
        if not self._block_terminated(self.state.current_block):
            self._emit_jump(test_block, stmt, loop_backedge=True)

        self.state.current_block = exit_block
        # Build the post-loop name env: phi outputs for variables that
        # existed before the loop, PLUS any names introduced inside the
        # body (Python's `while` does not establish a scope).
        post_env: dict[str, str] = {
            name: self.state.ops[phi_op_id].outputs[0]
            for name, (phi_op_id, _) in loop_phi_inputs.items()
        }
        for name, vid in body_env.items():
            if name not in post_env:
                post_env[name] = vid
        self.state.name_env = post_env

    def _compile_for(self, stmt: ast.For) -> None:
        """
        Compile a for-loop with proper phi nodes for all variables in scope.

        Fixed: the original implementation didn't create phi nodes, so variables
        modified in the loop body weren't properly propagated to the next iteration.
        """
        before_env = dict(self.state.name_env)

        iter_value = self._compile_expr(stmt.iter)
        iter_handle = self._emit_iter_init(iter_value, stmt)

        test_block = self._new_block("for_test")
        body_block = self._new_block("for_body")
        exit_block = self._new_block("for_exit")
        self._emit_jump(test_block, stmt)

        # Create phi nodes in test block for all variables in scope
        self.state.current_block = test_block
        loop_phi_inputs: dict[str, tuple[str, str]] = {}  # name -> (phi_op_id, entry_value)
        self.state.name_env = dict(before_env)
        for name, incoming_value in before_env.items():
            phi_type = unify_type_infos(self._value_type_info(incoming_value))
            phi_out = self._new_versioned_name(
                name, stmt,
                type_hint=type_hint_from_info(phi_type),
                attrs={"type_info": phi_type.to_dict()},
            )
            phi_op_id = self._emit_phi(
                [incoming_value, incoming_value], phi_out,
                ["loop_entry", "loop_backedge"], stmt,
                var_name=name, loop_phi=True,
            )
            loop_phi_inputs[name] = (phi_op_id, incoming_value)
            self.state.name_env[name] = phi_out

        # iter_next
        next_value = self._new_value("iter_next", "object", source_span(stmt), {})
        has_next = self._new_value("iter_has_next", "bool", source_span(stmt), {})
        self._emit_iter_next(iter_handle, next_value, has_next, stmt)

        # Assign loop variable
        self._assign_target(stmt.target, next_value)

        # Branch
        self._emit_branch(has_next, body_block, exit_block, stmt)

        # Compile body
        self.state.current_block = body_block
        for body_stmt in stmt.body:
            self._compile_stmt(body_stmt)

        # Update phi backedges
        body_env = dict(self.state.name_env)
        for name, (phi_op_id, entry_value) in loop_phi_inputs.items():
            backedge_value = body_env.get(name, entry_value)
            phi_op = self.state.ops[phi_op_id]
            if backedge_value == phi_op.outputs[0]:
                backedge_value = entry_value
            phi_op.inputs[1] = backedge_value
            self.state.values[backedge_value].use_ops.append(phi_op_id)

        if not self._block_terminated(self.state.current_block):
            self._emit_jump(test_block, stmt, loop_backedge=True)

        self.state.current_block = exit_block
        # After the for loop: phi outputs for pre-loop names + any names
        # introduced inside the body (Python `for` does not establish a scope).
        post_env: dict[str, str] = {
            name: self.state.ops[phi_op_id].outputs[0]
            for name, (phi_op_id, _) in loop_phi_inputs.items()
        }
        for name, vid in body_env.items():
            if name not in post_env:
                post_env[name] = vid
        self.state.name_env = post_env

    # ------------------------------------------------------------------
    # Slot DSL: `with slot(name, inputs=(...), outputs=("a","b")): body`
    # ------------------------------------------------------------------

    def _compile_with(self, stmt: ast.With) -> None:
        if len(stmt.items) != 1:
            raise NotImplementedError(
                f"`with` must have exactly one item; got {len(stmt.items)} at {source_span(stmt)}"
            )
        item = stmt.items[0]
        if item.optional_vars is not None:
            raise NotImplementedError(
                f"`with slot(...) as X` not supported (no `as` clause); at {source_span(stmt)}"
            )
        ctx = item.context_expr
        if not (isinstance(ctx, ast.Call)
                and isinstance(ctx.func, ast.Name)
                and ctx.func.id == "slot"):
            raise NotImplementedError(
                "Only `with slot(name, inputs=..., outputs=...):` is supported "
                f"as a `with` context at {source_span(stmt)}"
            )

        # Parse positional name (single str literal)
        if (len(ctx.args) != 1
                or not isinstance(ctx.args[0], ast.Constant)
                or not isinstance(ctx.args[0].value, str)):
            raise SyntaxError(
                f"slot(...) expects a single string-literal name at {source_span(stmt)}"
            )
        pop_key: str = ctx.args[0].value

        # Parse kwargs: inputs=(name1, name2, ...), outputs=("a","b")
        kwargs = {kw.arg: kw.value for kw in ctx.keywords}
        if "inputs" not in kwargs or "outputs" not in kwargs:
            raise SyntaxError(
                f"slot('{pop_key}') requires `inputs=(...)` and `outputs=(...)` kwargs"
            )

        # inputs: tuple/list of ast.Name → resolve to current SSA value-ids
        inputs_node = kwargs["inputs"]
        if not isinstance(inputs_node, (ast.Tuple, ast.List)):
            raise SyntaxError(
                f"slot('{pop_key}') inputs must be a tuple/list literal of names"
            )
        entry_value_ids: list[str] = []
        for el in inputs_node.elts:
            if not isinstance(el, ast.Name):
                raise SyntaxError(
                    f"slot('{pop_key}') inputs entries must be plain names; got {ast.dump(el)}"
                )
            if el.id not in self.state.name_env:
                raise SyntaxError(
                    f"slot('{pop_key}') input name '{el.id}' is not bound at slot entry"
                )
            entry_value_ids.append(self.state.name_env[el.id])

        # outputs: tuple/list of string literals — declared output variable names
        outputs_node = kwargs["outputs"]
        if not isinstance(outputs_node, (ast.Tuple, ast.List)):
            raise SyntaxError(
                f"slot('{pop_key}') outputs must be a tuple/list of string literals"
            )
        output_names: list[str] = []
        for el in outputs_node.elts:
            if not (isinstance(el, ast.Constant) and isinstance(el.value, str)):
                raise SyntaxError(
                    f"slot('{pop_key}') outputs must contain string literals; got {ast.dump(el)}"
                )
            output_names.append(el.value)

        # Reject duplicate slot keys in the same function (would corrupt slot_meta).
        if pop_key in self.state.slot_meta or any(
            f["pop_key"] == pop_key for f in self.state.slot_stack
        ):
            raise SyntaxError(
                f"duplicate slot pop_key '{pop_key}' in function"
            )

        parent_key = self.state.slot_stack[-1]["pop_key"] if self.state.slot_stack else None

        frame = {
            "pop_key": pop_key,
            "entry_value_ids": tuple(entry_value_ids),
            "output_names": tuple(output_names),
            "op_ids": [],
            "parent": parent_key,
        }
        self.state.slot_stack.append(frame)
        try:
            for body_stmt in stmt.body:
                self._compile_stmt(body_stmt)
        finally:
            self.state.slot_stack.pop()

        # Resolve declared output names to current SSA value-ids
        exit_value_ids: list[str] = []
        for nm in output_names:
            if nm not in self.state.name_env:
                raise SyntaxError(
                    f"slot('{pop_key}') declared output '{nm}' is not bound at slot exit"
                )
            exit_value_ids.append(self.state.name_env[nm])

        self.state.slot_meta[pop_key] = {
            "pop_key": pop_key,
            "op_ids": tuple(frame["op_ids"]),
            "inputs": tuple(entry_value_ids),
            "outputs": tuple(exit_value_ids),
            "output_names": tuple(output_names),
            "parent": parent_key,
        }

    # ------------------------------------------------------------------
    # Expression-level control flow (S1.0)
    # ------------------------------------------------------------------

    def _compile_branching_expr(
        self,
        cond_id: str,
        then_fn,
        else_fn,
        node: ast.AST,
        result_hint: str,
    ) -> str:
        """Shared implementation for IfExp / BoolOp short-circuit.

        Emits branch-on-cond + then/else blocks + merge with phi over the
        single result value. ``then_fn`` and ``else_fn`` are zero-arg
        callables that return a value-id when invoked in their respective
        block. Returns the merged result value-id; current block is set to
        the merge block on return.
        """
        then_block = self._new_block(f"{result_hint}_then")
        else_block = self._new_block(f"{result_hint}_else")
        merge_block = self._new_block(f"{result_hint}_merge")
        self._emit_branch(cond_id, then_block, else_block, node)

        before_env = dict(self.state.name_env)
        before_versions = dict(self.state.name_versions)

        # Allocate a stable temp name materialized in both branches so the
        # source emitter sees a real assignment statement rather than just
        # phi-merging dangling expression values (those would be elided as
        # dead code in either branch and produce empty `if`/`else` bodies).
        temp_name = f"__{result_hint}_tmp_{self._gensym_counter()}"

        # then branch
        self.state.current_block = then_block
        self.state.name_env = dict(before_env)
        self.state.name_versions = dict(before_versions)
        then_val = then_fn()
        # Materialize: temp_name = then_val  (creates an `assign` op)
        self._assign_target(ast.Name(id=temp_name, ctx=ast.Store(),
                                     lineno=getattr(node, 'lineno', 0),
                                     col_offset=getattr(node, 'col_offset', 0)),
                            then_val)
        then_temp_value = self.state.name_env[temp_name]
        if not self._block_terminated(self.state.current_block):
            self._emit_jump(merge_block, node)
        then_end = self.state.current_block
        then_env = dict(self.state.name_env)

        # else branch
        self.state.current_block = else_block
        self.state.name_env = dict(before_env)
        self.state.name_versions = dict(before_versions)
        else_val = else_fn()
        self._assign_target(ast.Name(id=temp_name, ctx=ast.Store(),
                                     lineno=getattr(node, 'lineno', 0),
                                     col_offset=getattr(node, 'col_offset', 0)),
                            else_val)
        else_temp_value = self.state.name_env[temp_name]
        if not self._block_terminated(self.state.current_block):
            self._emit_jump(merge_block, node)
        else_end = self.state.current_block
        else_env = dict(self.state.name_env)

        # merge
        self.state.current_block = merge_block

        # Phi for the materialized temp variable
        result_type = unify_type_infos(
            self._value_type_info(then_temp_value),
            self._value_type_info(else_temp_value),
        )
        result_value = self._new_versioned_name(
            temp_name, node,
            type_hint=type_hint_from_info(result_type),
            attrs={"type_info": result_type.to_dict()},
        )
        self._emit_phi(
            [then_temp_value, else_temp_value], result_value,
            [then_end, else_end], node,
            var_name=temp_name,
        )

        # Phi for any other names that diverged (defensive — uncommon for
        # pure IfExp but possible if then_fn / else_fn caused side effects)
        merged_env = dict(before_env)
        merged_env[temp_name] = result_value
        keys = (set(before_env) | set(then_env) | set(else_env)) - {temp_name}
        for key in keys:
            t_val = then_env.get(key, before_env.get(key))
            e_val = else_env.get(key, before_env.get(key))
            if t_val != e_val and t_val is not None and e_val is not None:
                phi_type = unify_type_infos(
                    self._value_type_info(t_val), self._value_type_info(e_val)
                )
                phi_value = self._new_versioned_name(
                    key, node,
                    type_hint=type_hint_from_info(phi_type),
                    attrs={"type_info": phi_type.to_dict()},
                )
                self._emit_phi([t_val, e_val], phi_value, [then_end, else_end], node, var_name=key)
                merged_env[key] = phi_value
            elif t_val is not None:
                merged_env[key] = t_val
            elif e_val is not None:
                merged_env[key] = e_val
        self.state.name_env = merged_env

        return result_value

    def _gensym_counter(self) -> int:
        cnt = getattr(self.state, "_gensym_counter", 0)
        self.state._gensym_counter = cnt + 1
        return cnt

    def _compile_if_expr(self, expr: ast.IfExp) -> str:
        cond = self._compile_expr(expr.test)
        return self._compile_branching_expr(
            cond,
            then_fn=lambda: self._compile_expr(expr.body),
            else_fn=lambda: self._compile_expr(expr.orelse),
            node=expr,
            result_hint="ifexp",
        )

    def _compile_bool_op(self, expr: ast.BoolOp) -> str:
        # Short-circuit: fold left-to-right.
        # `a and b and c` -> `((a and b) and c)`
        # `a or b or c`   -> `((a or b) or c)`
        is_and = isinstance(expr.op, ast.And)
        result = self._compile_expr(expr.values[0])
        for next_expr in expr.values[1:]:
            if is_and:
                # a and b -> if a: b else: a
                result = self._compile_branching_expr(
                    result,
                    then_fn=lambda ne=next_expr: self._compile_expr(ne),
                    else_fn=lambda r=result: r,
                    node=expr,
                    result_hint="and",
                )
            else:
                # a or b -> if a: a else: b
                result = self._compile_branching_expr(
                    result,
                    then_fn=lambda r=result: r,
                    else_fn=lambda ne=next_expr: self._compile_expr(ne),
                    node=expr,
                    result_hint="or",
                )
        return result

    def _compile_slice(self, sl: ast.Slice) -> str:
        """Compile an ast.Slice into a 'build_slice' op returning a value.

        The op has up to 3 inputs (lower, upper, step); missing components
        are emitted as None constants. Source emission produces ``lo:hi:step``
        when consumed inside a get_item / set_item.
        """
        def _none_const() -> str:
            return self._compile_constant(None, sl)

        lo = self._compile_expr(sl.lower) if sl.lower is not None else _none_const()
        hi = self._compile_expr(sl.upper) if sl.upper is not None else _none_const()
        st = self._compile_expr(sl.step) if sl.step is not None else _none_const()

        out_id = self._new_value(
            name_hint="slice",
            type_hint="slice",
            source=source_span(sl),
            attrs={
                "has_lower": sl.lower is not None,
                "has_upper": sl.upper is not None,
                "has_step": sl.step is not None,
            },
        )
        op_id = self._next_op_id()
        op = Op(
            id=op_id,
            opcode="build_slice",
            inputs=[lo, hi, st],
            outputs=[out_id],
            block_id=self.state.current_block,
            source_span=source_span(sl),
            attrs={
                "has_lower": sl.lower is not None,
                "has_upper": sl.upper is not None,
                "has_step": sl.step is not None,
            },
        )
        self._register_op(op)
        return out_id

    # ------------------------------------------------------------------
    # Assignment targets
    # ------------------------------------------------------------------

    def _assign_target(self, target: ast.AST, value_id: str) -> None:
        if isinstance(target, ast.Name):
            source_value = self.state.values[value_id]
            propagated_attrs: dict[str, Any] = {}
            if "type_info" in source_value.attrs:
                propagated_attrs["type_info"] = source_value.attrs["type_info"]
            new_value = self._new_versioned_name(
                target.id, target,
                type_hint=source_value.type_hint or "object",
                attrs=propagated_attrs,
            )
            self._emit_assign(value_id, new_value, target.id, target)
            self.state.name_env[target.id] = new_value
        elif isinstance(target, ast.Attribute):
            owner = self._compile_expr(target.value)
            self._emit_set_attr(owner, value_id, target.attr, target)
        elif isinstance(target, ast.Subscript):
            owner = self._compile_expr(target.value)
            if isinstance(target.slice, ast.Slice):
                index = self._compile_slice(target.slice)
            else:
                index = self._compile_expr(target.slice)
            self._emit_set_item(owner, index, value_id, target)
        else:
            raise NotImplementedError(f"Unsupported assignment target {type(target).__name__}")

    # ------------------------------------------------------------------
    # Expression compilation
    # ------------------------------------------------------------------

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
            return self._emit_binary(lhs, rhs, type(expr.op).__name__, expr)
        if isinstance(expr, ast.UnaryOp):
            operand = self._compile_expr(expr.operand)
            return self._emit_unary(operand, type(expr.op).__name__, expr)
        if isinstance(expr, ast.Compare):
            left = self._compile_expr(expr.left)
            rights = [self._compile_expr(comp) for comp in expr.comparators]
            return self._emit_compare(left, rights, [type(op).__name__ for op in expr.ops], expr)
        if isinstance(expr, ast.Call):
            func_value = self._compile_expr(expr.func)
            args = [self._compile_expr(arg) for arg in expr.args]
            kwarg_names: list[str] = []
            kwarg_values: list[str] = []
            for kw in expr.keywords:
                if kw.arg is None:
                    continue  # skip **kwargs (not supported)
                kwarg_names.append(kw.arg)
                kwarg_values.append(self._compile_expr(kw.value))
            return self._emit_call(func_value, args, expr,
                                   kwarg_names=kwarg_names,
                                   kwarg_values=kwarg_values)
        if isinstance(expr, ast.Attribute):
            owner = self._compile_expr(expr.value)
            return self._emit_get_attr(owner, expr.attr, expr)
        if isinstance(expr, ast.Subscript):
            owner = self._compile_expr(expr.value)
            if isinstance(expr.slice, ast.Slice):
                index = self._compile_slice(expr.slice)
            else:
                index = self._compile_expr(expr.slice)
            return self._emit_get_item(owner, index, expr)
        if isinstance(expr, ast.List):
            items = [self._compile_expr(item) for item in expr.elts]
            return self._emit_build_list(items, expr)
        if isinstance(expr, ast.Tuple):
            items = [self._compile_expr(item) for item in expr.elts]
            return self._emit_build_tuple(items, expr)
        if isinstance(expr, ast.Dict):
            kv_pairs: list[str] = []
            for key, value in zip(expr.keys, expr.values):
                kv_pairs.append(self._compile_expr(key))
                kv_pairs.append(self._compile_expr(value))
            return self._emit_build_dict(kv_pairs, len(expr.keys), expr)
        if isinstance(expr, ast.IfExp):
            return self._compile_if_expr(expr)
        if isinstance(expr, ast.BoolOp):
            return self._compile_bool_op(expr)
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
        self._emit_const(value, out, node)
        return out

    def _load_global(self, name: str, node: ast.AST) -> str:
        if name in self.state.global_constants:
            return self.state.global_constants[name]
        if name in self.parsed.globals_dict:
            obj = self.parsed.globals_dict[name]
        elif hasattr(builtins, name):
            obj = getattr(builtins, name)
        else:
            raise NameError(f"Unknown name {name} at {source_span(node)}")

        # Handle module objects (e.g. `math`) — store the module itself as a constant
        if isinstance(obj, pytypes.ModuleType):
            value_id = self._new_value(
                name_hint=name,
                type_hint="module",
                source=source_span(node),
                attrs={"literal": obj, "scope": "global", "module_name": obj.__name__},
            )
            self._emit_const(obj, value_id, node, const_name=name)
            self.state.global_constants[name] = value_id
            return value_id

        value_id = self._new_value(
            name_hint=name,
            type_hint=type(obj).__name__,
            source=source_span(node),
            attrs={
                "literal": obj,
                "scope": "global",
                "qualified_name": name,
            },
        )
        self._emit_const(obj, value_id, node, const_name=name)
        self.state.global_constants[name] = value_id
        return value_id

    # ------------------------------------------------------------------
    # Op emission helpers (emit Op + track in state)
    # ------------------------------------------------------------------

    def _emit_const(self, literal: Any, out_id: str, node: ast.AST, *, const_name: str | None = None) -> str:
        op_id = self._next_op_id()
        attrs: dict[str, Any] = {"literal": literal}
        if const_name:
            attrs["name"] = const_name
        op = Op(
            id=op_id, opcode="const", inputs=[], outputs=[out_id],
            block_id=self.state.current_block, source_span=source_span(node), attrs=attrs,
        )
        self._register_op(op)
        return op_id

    def _emit_assign(self, source_id: str, out_id: str, target_name: str, node: ast.AST) -> str:
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="assign", inputs=[source_id], outputs=[out_id],
            block_id=self.state.current_block, source_span=source_span(node),
            attrs={"target": target_name},
        )
        self._register_op(op)
        return op_id

    def _emit_binary(self, lhs: str, rhs: str, operator: str, node: ast.AST) -> str:
        result_type = combine_binary_type_info(operator, self._value_type_info(lhs), self._value_type_info(rhs))
        out = self._new_value(
            "binary", type_hint_from_info(result_type), source_span(node),
            {"type_info": result_type.to_dict()},
        )
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="binary", inputs=[lhs, rhs], outputs=[out],
            block_id=self.state.current_block, source_span=source_span(node),
            attrs={"operator": operator, "type_info": result_type.to_dict()},
        )
        self._register_op(op)
        return out

    def _emit_unary(self, operand: str, operator: str, node: ast.AST) -> str:
        operand_type = self._value_type_info(operand)
        out = self._new_value(
            "unary", type_hint_from_info(operand_type), source_span(node),
            {"type_info": operand_type.to_dict()},
        )
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="unary", inputs=[operand], outputs=[out],
            block_id=self.state.current_block, source_span=source_span(node),
            attrs={"operator": operator, "type_info": operand_type.to_dict()},
        )
        self._register_op(op)
        return out

    def _emit_compare(self, left: str, rights: list[str], operators: list[str], node: ast.AST) -> str:
        out = self._new_value("compare", "bool", source_span(node), {})
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="compare", inputs=[left] + rights, outputs=[out],
            block_id=self.state.current_block, source_span=source_span(node),
            attrs={"operators": operators},
        )
        self._register_op(op)
        return out

    def _emit_call(self, func_value: str, args: list[str], node: ast.AST,
                   *, kwarg_names: list[str] | None = None,
                   kwarg_values: list[str] | None = None) -> str:
        kwarg_names = kwarg_names or []
        kwarg_values = kwarg_values or []
        all_inputs = [func_value] + args + kwarg_values
        # Lattice-aware call-result type: consult the callable registry
        # using the callee's recorded qualified_name (set by _load_global
        # / _load_resolved_global).  Falls back to TYPE_TOP when unknown,
        # never to the lying "object" tag the lifter used to emit.
        from algorithm_ir.ir.type_lattice import infer_call_return_type
        callee_val = self.state.values.get(func_value)
        callee_qname = (
            callee_val.attrs.get("qualified_name") if callee_val else None
        )
        arg_types = [
            (self.state.values[a].type_hint or "any")
            for a in args if a in self.state.values
        ]
        result_type = infer_call_return_type(callee_qname, arg_types)
        out = self._new_value("call", result_type, source_span(node), {})
        op_id = self._next_op_id()
        attrs: dict[str, Any] = {"n_args": len(args)}
        if callee_qname:
            attrs["qualified_name"] = callee_qname
        if kwarg_names:
            attrs["kwarg_names"] = kwarg_names
            attrs["n_kwargs"] = len(kwarg_names)
        op = Op(
            id=op_id, opcode="call", inputs=all_inputs, outputs=[out],
            block_id=self.state.current_block, source_span=source_span(node),
            attrs=attrs,
        )
        self._register_op(op)
        return out

    def _emit_get_attr(self, owner: str, attr_name: str, node: ast.AST) -> str:
        # If the owner is a module constant, resolve the attribute at compile time
        owner_value = self.state.values[owner]
        if owner_value.type_hint == "module" and "literal" in owner_value.attrs:
            module_obj = owner_value.attrs["literal"]
            if hasattr(module_obj, attr_name):
                resolved = getattr(module_obj, attr_name)
                return self._load_resolved_global(
                    f"{owner_value.name_hint}.{attr_name}", resolved, node
                )

        out = self._new_value("get_attr", "object", source_span(node), {})
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="get_attr", inputs=[owner], outputs=[out],
            block_id=self.state.current_block, source_span=source_span(node),
            attrs={"attr": attr_name},
        )
        self._register_op(op)
        return out

    def _load_resolved_global(self, name: str, obj: Any, node: ast.AST) -> str:
        """Load a resolved global (e.g. math.sqrt) as a const.

        Records ``qualified_name`` on the value's attrs so that
        downstream ``_emit_call`` can route the result type through the
        callable registry.
        """
        if name in self.state.global_constants:
            return self.state.global_constants[name]
        value_id = self._new_value(
            name_hint=name,
            type_hint=type(obj).__name__,
            source=source_span(node),
            attrs={
                "literal": obj,
                "scope": "global",
                "qualified_name": name,
            },
        )
        self._emit_const(obj, value_id, node, const_name=name)
        self.state.global_constants[name] = value_id
        return value_id

    def _emit_set_attr(self, owner: str, value: str, attr_name: str, node: ast.AST) -> str:
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="set_attr", inputs=[owner, value], outputs=[],
            block_id=self.state.current_block, source_span=source_span(node),
            attrs={"attr": attr_name},
        )
        self._register_op(op)
        return op_id

    def _emit_get_item(self, owner: str, index: str, node: ast.AST) -> str:
        out = self._new_value("get_item", "object", source_span(node), {})
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="get_item", inputs=[owner, index], outputs=[out],
            block_id=self.state.current_block, source_span=source_span(node), attrs={},
        )
        self._register_op(op)
        return out

    def _emit_set_item(self, owner: str, index: str, value: str, node: ast.AST) -> str:
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="set_item", inputs=[owner, index, value], outputs=[],
            block_id=self.state.current_block, source_span=source_span(node), attrs={},
        )
        self._register_op(op)
        return op_id

    def _emit_build_list(self, items: list[str], node: ast.AST) -> str:
        elem_types = [self._value_type_info(item) for item in items]
        elem_type = unify_type_infos(*elem_types) if elem_types else TypeInfo("object")
        out = self._new_value(
            "build_list", "list", source_span(node),
            {"type_info": {"kind": "list", "elem": elem_type.to_dict(), "shape": [len(items)]}},
        )
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="build_list", inputs=list(items), outputs=[out],
            block_id=self.state.current_block, source_span=source_span(node),
            attrs={"n_items": len(items), "type_info": {"kind": "list", "elem": elem_type.to_dict(), "shape": [len(items)]}},
        )
        self._register_op(op)
        return out

    def _emit_build_tuple(self, items: list[str], node: ast.AST) -> str:
        elem_types = [self._value_type_info(item) for item in items]
        elem_type = unify_type_infos(*elem_types) if elem_types else TypeInfo("object")
        out = self._new_value(
            "build_tuple", "tuple", source_span(node),
            {"type_info": {"kind": "tuple", "elem": elem_type.to_dict(), "arity": len(items), "shape": [len(items)]}},
        )
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="build_tuple", inputs=list(items), outputs=[out],
            block_id=self.state.current_block, source_span=source_span(node),
            attrs={"n_items": len(items), "type_info": {"kind": "tuple", "elem": elem_type.to_dict(), "arity": len(items), "shape": [len(items)]}},
        )
        self._register_op(op)
        return out

    def _emit_build_dict(self, kv_pairs: list[str], n_items: int, node: ast.AST) -> str:
        key_types = [self._value_type_info(kv_pairs[i]) for i in range(0, len(kv_pairs), 2)]
        val_types = [self._value_type_info(kv_pairs[i]) for i in range(1, len(kv_pairs), 2)]
        key_type = unify_type_infos(*key_types) if key_types else TypeInfo("object")
        val_type = unify_type_infos(*val_types) if val_types else TypeInfo("object")
        out = self._new_value(
            "build_dict", "dict", source_span(node),
            {"type_info": {"kind": "dict", "key": key_type.to_dict(), "value": val_type.to_dict(), "shape": [n_items]}},
        )
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="build_dict", inputs=list(kv_pairs), outputs=[out],
            block_id=self.state.current_block, source_span=source_span(node),
            attrs={"n_items": n_items, "type_info": {"kind": "dict", "key": key_type.to_dict(), "value": val_type.to_dict(), "shape": [n_items]}},
        )
        self._register_op(op)
        return out

    def _emit_phi(
        self, inputs: list[str], out_id: str, sources: list[str],
        node: ast.AST, *, var_name: str | None = None, loop_phi: bool = False,
    ) -> str:
        op_id = self._next_op_id()
        attrs: dict[str, Any] = {"sources": sources}
        if loop_phi:
            attrs["loop_phi"] = True
        if var_name:
            attrs["var_name"] = var_name
        op = Op(
            id=op_id, opcode="phi", inputs=list(inputs), outputs=[out_id],
            block_id=self.state.current_block, source_span=source_span(node), attrs=attrs,
        )
        self._register_op(op)
        return op_id

    def _emit_branch(self, cond: str, true_block: str, false_block: str, node: ast.AST) -> str:
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="branch", inputs=[cond], outputs=[],
            block_id=self.state.current_block, source_span=source_span(node),
            attrs={"true": true_block, "false": false_block},
        )
        self._register_op(op)
        self._link_blocks(self.state.current_block, true_block)
        self._link_blocks(self.state.current_block, false_block)
        return op_id

    def _emit_jump(self, target: str, node: ast.AST, *, loop_backedge: bool = False) -> str:
        op_id = self._next_op_id()
        attrs: dict[str, Any] = {"target": target}
        if loop_backedge:
            attrs["loop_backedge"] = True
        op = Op(
            id=op_id, opcode="jump", inputs=[], outputs=[],
            block_id=self.state.current_block, source_span=source_span(node), attrs=attrs,
        )
        self._register_op(op)
        self._link_blocks(self.state.current_block, target)
        return op_id

    def _emit_return(self, value: str, node: ast.AST) -> str:
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="return", inputs=[value], outputs=[],
            block_id=self.state.current_block, source_span=source_span(node), attrs={},
        )
        self._register_op(op)
        return op_id

    def _emit_iter_init(self, iterable: str, node: ast.AST) -> str:
        out = self._new_value("iter_init", "object", source_span(node), {})
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="iter_init", inputs=[iterable], outputs=[out],
            block_id=self.state.current_block, source_span=source_span(node), attrs={},
        )
        self._register_op(op)
        return out

    def _emit_iter_next(self, iterator: str, next_id: str, has_next_id: str, node: ast.AST) -> str:
        op_id = self._next_op_id()
        op = Op(
            id=op_id, opcode="iter_next", inputs=[iterator], outputs=[next_id, has_next_id],
            block_id=self.state.current_block, source_span=source_span(node), attrs={},
        )
        self._register_op(op)
        return op_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_op(self, op: Op) -> None:
        """Register an op in state: ops dict, block op_ids, def/use tracking."""
        self.state.ops[op.id] = op
        self.state.blocks[op.block_id].op_ids.append(op.id)
        # Slot tagging: only the innermost enclosing slot owns the op.
        if self.state.slot_stack:
            innermost = self.state.slot_stack[-1]
            op.attrs["slot_id"] = innermost["pop_key"]
            innermost["op_ids"].append(op.id)
        for out in op.outputs:
            self.state.values[out].def_op = op.id
        for inp in op.inputs:
            if inp in self.state.values:
                self.state.values[inp].use_ops.append(op.id)

    def _new_value(self, name_hint: str | None, type_hint: str | None,
                   source: tuple[int, int, int, int] | None, attrs: dict[str, Any]) -> str:
        value_id = f"v_{next(self.value_counter)}"
        self.state.values[value_id] = Value(
            id=value_id, name_hint=name_hint, type_hint=type_hint,
            source_span=source, attrs=dict(attrs),
        )
        return value_id

    def _new_versioned_name(self, name: str, node: ast.AST, *,
                            type_hint: str = "object", attrs: dict[str, Any] | None = None) -> str:
        version = self.state.name_versions.get(name, -1) + 1
        self.state.name_versions[name] = version
        payload = {"var_name": name, "version": version}
        if attrs:
            payload.update(attrs)
        return self._new_value(
            name_hint=f"{name}_{version}", type_hint=type_hint,
            source=source_span(node), attrs=payload,
        )

    def _value_type_info(self, value_id: str) -> TypeInfo:
        value = self.state.values[value_id]
        payload = value.attrs.get("type_info")
        if isinstance(payload, dict):
            from algorithm_ir.ir.type_info import type_info_from_dict
            return type_info_from_dict(payload)
        if value.type_hint in _TYPE_SENTINELS:
            return type_info_for_python_value(_TYPE_SENTINELS[value.type_hint])
        return TypeInfo("object")

    def _next_op_id(self) -> str:
        return f"op_{next(self.op_counter)}"

    def _new_block_id(self, prefix: str) -> str:
        return f"b_{prefix}_{next(self.block_counter)}"

    def _new_block(self, prefix: str) -> str:
        block_id = self._new_block_id(prefix)
        self.state.blocks[block_id] = Block(id=block_id)
        self.state.xdsl_blocks[block_id] = XBlock()
        return block_id

    def _link_blocks(self, src: str, dst: str) -> None:
        src_block = self.state.blocks[src]
        dst_block = self.state.blocks[dst]
        if dst not in src_block.succs:
            src_block.succs.append(dst)
        if src not in dst_block.preds:
            dst_block.preds.append(src)

    def _block_terminated(self, block_id: str) -> bool:
        for op in self.state.ops.values():
            if op.block_id == block_id and op.opcode in {"return", "branch", "jump"}:
                return True
        return False

    # ------------------------------------------------------------------
    # xDSL module building
    # ------------------------------------------------------------------

    def _build_xdsl_module(self, arg_values: list[str]) -> ModuleOp:
        """Build an xDSL ModuleOp from the compiled state using typed AlgDialect ops."""
        xdsl_block_map: dict[str, XBlock] = {}
        xdsl_op_map: dict[str, Any] = {}

        # Create fresh xDSL blocks (we need fresh ones for the module)
        block_id_order = list(self.state.blocks.keys())
        for block_id in block_id_order:
            xdsl_block_map[block_id] = XBlock()

        # Populate blocks with typed xDSL ops
        for block_id in block_id_order:
            xdsl_block = xdsl_block_map[block_id]
            block = self.state.blocks[block_id]
            for op_id in block.op_ids:
                op = self.state.ops[op_id]
                xdsl_op = self._build_xdsl_op(op, xdsl_block_map)
                if xdsl_op is not None:
                    xdsl_block.add_op(xdsl_op)
                    xdsl_op_map[op_id] = xdsl_op

        # Build func and module
        region = Region(list(xdsl_block_map.values()))
        func = FuncOp(self.parsed.tree.name, ([], []), region)

        # Store metadata in func attributes
        func.attributes["alg_id"] = StringAttr(self.function_id)
        func.attributes["alg_arg_values"] = StringAttr(",".join(arg_values))
        func.attributes["alg_return_values"] = StringAttr(",".join(self.state.return_values))
        func.attributes["alg_entry_block"] = StringAttr(self.state.entry_block)
        if self.parsed.filename:
            func.attributes["alg_filename"] = StringAttr(self.parsed.filename)
        if self.parsed.source:
            func.attributes["alg_source"] = StringAttr(self.parsed.source)

        # Store arg metadata
        for arg_id in arg_values:
            v = self.state.values[arg_id]
            meta = {
                "name_hint": v.name_hint, "type_hint": v.type_hint,
                "source_span": v.source_span, "attrs": _sanitize_for_repr(v.attrs),
            }
            func.attributes[f"alg_arg_{arg_id}"] = StringAttr(repr(meta))

        module = ModuleOp([func])
        # Store maps on the module for later use
        self._xdsl_op_map = xdsl_op_map
        # Update state's xdsl_blocks to the ones we just built
        self.state.xdsl_blocks = xdsl_block_map
        return module

    def _build_xdsl_op(self, op: Op, block_map: dict[str, XBlock]):
        """Build a single typed xDSL op from an Op dataclass."""
        import ast as py_ast

        attrs: dict[str, Any] = {}
        # Always store op_id, block_id, input_ids, output_meta, source_span
        attrs["op_id"] = StringAttr(op.id)
        attrs["block_id"] = StringAttr(op.block_id)
        attrs["input_ids"] = StringAttr(repr(op.inputs))

        # Build output_meta for value reconstruction
        output_meta = []
        for out_id in op.outputs:
            v = self.state.values[out_id]
            output_meta.append({
                "id": v.id, "name_hint": v.name_hint, "type_hint": v.type_hint,
                "source_span": v.source_span, "attrs": _sanitize_for_repr(v.attrs),
            })
        attrs["output_meta"] = StringAttr(repr(output_meta))

        if op.source_span:
            attrs["source_span"] = StringAttr(repr(op.source_span))

        # Store extra attrs that aren't captured by the op's own IRDL attrs
        extra_attrs = _filter_extra_attrs(op)
        if extra_attrs:
            attrs["alg_attrs"] = StringAttr(repr(_sanitize_for_repr(extra_attrs)))

        rt = AlgType()  # result type for all results

        if op.opcode == "const":
            literal = op.attrs.get("literal")
            attrs["value"] = StringAttr(repr(_sanitize_for_repr(literal)))
            if op.attrs.get("name"):
                attrs["alg_name"] = StringAttr(op.attrs["name"])
            # Store literal for callable reconstruction
            attrs["alg_literal"] = StringAttr(repr(_sanitize_for_repr(literal)))
            type_hint = self.state.values[op.outputs[0]].type_hint if op.outputs else None
            if type_hint:
                attrs["type_hint"] = StringAttr(type_hint)
            return AlgConst.build(result_types=[rt], attributes=attrs)

        elif op.opcode == "assign":
            if op.attrs.get("target"):
                attrs["var_name"] = StringAttr(op.attrs["target"])
            # We can't use SSA operands here because values are string-IDs, not SSA values.
            # Instead we build the op without operands and store input_ids for later.
            return AlgAssign.build(
                operands=[_DUMMY_OPERAND], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "binary":
            attrs["operator"] = StringAttr(op.attrs["operator"])
            return AlgBinary.build(
                operands=[_DUMMY_OPERAND, _DUMMY_OPERAND], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "unary":
            attrs["operator"] = StringAttr(op.attrs["operator"])
            return AlgUnary.build(
                operands=[_DUMMY_OPERAND], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "compare":
            ops_str = ",".join(op.attrs["operators"])
            attrs["operators"] = StringAttr(ops_str)
            n_args = len(op.inputs)
            return AlgCompare.build(
                operands=[[_DUMMY_OPERAND] * n_args], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "phi":
            sources = op.attrs.get("sources", [])
            attrs["sources"] = StringAttr(",".join(str(s) for s in sources))
            if op.attrs.get("var_name"):
                attrs["var_name"] = StringAttr(op.attrs["var_name"])
            n_inputs = len(op.inputs)
            return AlgPhi.build(
                operands=[[_DUMMY_OPERAND] * n_inputs], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "call":
            attrs["n_args"] = IntegerAttr(op.attrs["n_args"], i64)
            n_inputs = len(op.inputs)
            return AlgCall.build(
                operands=[[_DUMMY_OPERAND] * n_inputs], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "get_attr":
            attrs["attr_name"] = StringAttr(op.attrs["attr"])
            return AlgGetAttr.build(
                operands=[_DUMMY_OPERAND], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "set_attr":
            attrs["attr_name"] = StringAttr(op.attrs["attr"])
            return AlgSetAttr.build(
                operands=[_DUMMY_OPERAND, _DUMMY_OPERAND], attributes=attrs,
            )

        elif op.opcode == "get_item":
            return AlgGetItem.build(
                operands=[_DUMMY_OPERAND, _DUMMY_OPERAND], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "set_item":
            return AlgSetItem.build(
                operands=[_DUMMY_OPERAND, _DUMMY_OPERAND, _DUMMY_OPERAND], attributes=attrs,
            )

        elif op.opcode == "build_list":
            n = len(op.inputs)
            return AlgBuildList.build(
                operands=[[_DUMMY_OPERAND] * n], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "build_tuple":
            n = len(op.inputs)
            return AlgBuildTuple.build(
                operands=[[_DUMMY_OPERAND] * n], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "build_dict":
            n = len(op.inputs)
            return AlgBuildDict.build(
                operands=[[_DUMMY_OPERAND] * n], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "append":
            return AlgAppend.build(
                operands=[_DUMMY_OPERAND, _DUMMY_OPERAND], attributes=attrs,
            )

        elif op.opcode == "pop":
            return AlgPop.build(
                operands=[_DUMMY_OPERAND, _DUMMY_OPERAND], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "iter_init":
            return AlgIterInit.build(
                operands=[_DUMMY_OPERAND], result_types=[rt], attributes=attrs,
            )

        elif op.opcode == "iter_next":
            return AlgIterNext.build(
                operands=[_DUMMY_OPERAND], result_types=[rt, rt], attributes=attrs,
            )

        elif op.opcode == "branch":
            true_block = block_map.get(op.attrs["true"])
            false_block = block_map.get(op.attrs["false"])
            if true_block and false_block:
                return AlgBranch.build(
                    operands=[_DUMMY_OPERAND], successors=[true_block, false_block],
                    attributes=attrs,
                )

        elif op.opcode == "jump":
            target = block_map.get(op.attrs["target"])
            if op.attrs.get("loop_backedge"):
                attrs["loop_backedge"] = StringAttr("true")
            if target:
                return AlgJump.build(successors=[target], attributes=attrs)

        elif op.opcode == "return":
            return AlgReturn.build(operands=[_DUMMY_OPERAND], attributes=attrs)

        return None

    def _build_function_ir(self, arg_values: list[str], xdsl_module: ModuleOp) -> FunctionIR:
        """Build FunctionIR directly from compilation state — no xDSL roundtrip needed."""
        func = next(iter(xdsl_module.ops))

        # Build op_ids for each block
        blocks: dict[str, Block] = {}
        for block_id, block in self.state.blocks.items():
            blocks[block_id] = Block(
                id=block_id,
                op_ids=list(block.op_ids),
                preds=list(block.preds),
                succs=list(block.succs),
                attrs={},
            )

        xdsl_block_map = dict(self.state.xdsl_blocks)

        func_attrs: dict[str, Any] = {
            "filename": self.parsed.filename,
            "source": self.parsed.source,
            "xdsl_module": xdsl_module,
        }
        from algorithm_ir.ir.xdsl_bridge import render_xdsl_module
        func_attrs["xdsl_text"] = render_xdsl_module(xdsl_module)

        return FunctionIR(
            id=self.function_id,
            name=self.parsed.tree.name,
            arg_values=arg_values,
            return_values=self.state.return_values,
            values=self.state.values,
            ops=self.state.ops,
            blocks=blocks,
            entry_block=self.state.entry_block,
            attrs=func_attrs,
            xdsl_module=xdsl_module,
            xdsl_func=func,
            xdsl_op_map=self._xdsl_op_map,
            xdsl_block_map=xdsl_block_map,
            slot_meta=self._build_slot_meta_objects(),
        )

    def _build_slot_meta_objects(self) -> dict[str, "SlotMeta"]:  # noqa: F821
        from algorithm_ir.ir.model import SlotMeta as _SlotMeta
        return {
            k: _SlotMeta(
                pop_key=v["pop_key"],
                op_ids=tuple(v["op_ids"]),
                inputs=tuple(v["inputs"]),
                outputs=tuple(v["outputs"]),
                output_names=tuple(v["output_names"]),
                parent=v.get("parent"),
            )
            for k, v in self.state.slot_meta.items()
        }


def compile_function_to_ir(fn: pytypes.FunctionType) -> FunctionIR:
    parsed = parse_function(fn)
    func_ir = IRBuilder(parsed).build()
    # R3: phi backedge rewrites in _compile_while/_compile_for leave
    # stale entries in Value.use_ops (the second phi input was first
    # registered against ``incoming_value`` and then rewritten to
    # ``backedge_value``). Recompute def-use to repair.
    from algorithm_ir.ir.validator import rebuild_def_use
    return rebuild_def_use(func_ir)


def compile_source_to_ir(
    source: str,
    func_name: str | None = None,
    globals_dict: dict | None = None,
) -> FunctionIR:
    """Compile a Python source string to FunctionIR.

    Unlike ``compile_function_to_ir``, this does not call ``inspect.getsource``
    and works for dynamically generated code.
    """
    tree = ast.parse(source)
    func_node = None
    sibling_names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if func_node is None and (func_name is None or node.name == func_name):
                func_node = node
            else:
                sibling_names.append(node.name)
    if func_node is None:
        raise ValueError(
            f"No function '{func_name or '<any>'}' found in source."
        )
    # Sibling functions defined in the same module are visible to the main
    # function at runtime via module globals. Register them as placeholder
    # callables in globals_dict so the IR builder's name resolution
    # succeeds. The actual callable identity is irrelevant for IR shape.
    effective_globals = dict(globals_dict or {})
    for sname in sibling_names:
        if sname not in effective_globals:
            effective_globals[sname] = _SiblingHelperPlaceholder(sname)
    parsed = ParsedFunction(
        tree=func_node,
        source=source,
        filename=f"<dynamic_{func_node.name}>",
        globals_dict=effective_globals,
    )
    func_ir = IRBuilder(parsed).build()
    # R3: see compile_function_to_ir.
    from algorithm_ir.ir.validator import rebuild_def_use
    return rebuild_def_use(func_ir)


class _SiblingHelperPlaceholder:
    """Marker for module-level sibling helper functions referenced from
    the function being compiled. Exists solely so global name resolution
    in the IR builder succeeds; never actually invoked."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"<sibling helper {self.name!r}>"

    def __call__(self, *args, **kwargs):  # pragma: no cover - never invoked
        raise RuntimeError(
            f"sibling helper placeholder for {self.name!r} should not be called"
        )


def _filter_extra_attrs(op: Op) -> dict[str, Any]:
    """
    Extract attrs that need to be stored in alg_attrs but aren't part
    of the IRDL op definition. Exclude known IRDL-captured attrs and
    non-serializable values.
    """
    skip = {
        "literal", "name", "target", "operator", "operators", "n_args",
        "attr", "true", "false", "loop_backedge", "sources", "var_name",
        "loop_phi", "slot_id", "slot_kind",
    }
    result: dict[str, Any] = {}
    for k, v in op.attrs.items():
        if k in skip:
            continue
        if callable(v) and not isinstance(v, type):
            continue
        result[k] = v
    return result


def _sanitize_for_repr(obj: Any) -> Any:
    """
    Recursively sanitize a value so that repr()/literal_eval() roundtrip works.
    Replaces callables and modules with dicts that can be reconstructed.
    Compatible with xdsl_bridge._denormalize_payload for reconstruction.
    """
    if obj is None or isinstance(obj, (bool, int, float, str, complex)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_repr(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_repr(item) for item in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_repr(item) for item in obj]
    if isinstance(obj, pytypes.ModuleType):
        return {"__callable__": obj.__name__, "module": obj.__name__}
    if callable(obj):
        return {
            "__callable__": getattr(obj, "__name__", repr(obj)),
            "module": getattr(obj, "__module__", "builtins"),
        }
    # Try repr to check if it roundtrips
    try:
        r = repr(obj)
        import ast as _ast
        _ast.literal_eval(r)
        return obj
    except Exception:
        return repr(obj)


# Dummy SSA value for building xDSL ops when we don't have real SSA
# (we track def-use via string IDs, not xDSL SSA)
_DUMMY_BLOCK = XBlock(arg_types=[AlgType()])
_DUMMY_OPERAND = _DUMMY_BLOCK.args[0]


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
