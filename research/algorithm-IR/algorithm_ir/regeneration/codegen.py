from __future__ import annotations

import struct
from collections import defaultdict
from typing import Any

from algorithm_ir.ir import render_function_ir
from algorithm_ir.ir.model import FunctionIR, Op
from algorithm_ir.regeneration.artifact import AlgorithmArtifact


def emit_artifact_source(artifact: AlgorithmArtifact) -> str:
    return emit_python_source(artifact.ir)


# ---------------------------------------------------------------------------
# Operator mappings
# ---------------------------------------------------------------------------

_BINARY_OPS = {
    "Add": "+", "Sub": "-", "Mult": "*", "Div": "/",
    "FloorDiv": "//", "Mod": "%", "Pow": "**",
    "MatMult": "@",
    "BitAnd": "&", "BitOr": "|", "BitXor": "^",
    "LShift": "<<", "RShift": ">>",
}

_COMPARE_OPS = {
    "Lt": "<", "Gt": ">", "LtE": "<=", "GtE": ">=",
    "Eq": "==", "NotEq": "!=", "Is": "is", "IsNot": "is not",
    "In": "in", "NotIn": "not in",
}

_UNARY_OPS = {
    "USub": "-", "UAdd": "+", "Not": "not ", "Invert": "~",
}


# ---------------------------------------------------------------------------
# Expression builder
# ---------------------------------------------------------------------------

class _ExprCtx:
    """Tracks value expressions for inlining constants and simple exprs."""

    def __init__(self, func_ir: FunctionIR):
        self.func_ir = func_ir
        self.exprs: dict[str, str] = {}
        self.lines: list[str] = []
        # Pre-populate constants
        for op in func_ir.ops.values():
            if op.opcode == "const" and op.outputs:
                vid = op.outputs[0]
                lit = op.attrs.get("literal")
                name = op.attrs.get("name")
                if name:
                    self.exprs[vid] = name
                elif lit is not None:
                    self.exprs[vid] = repr(lit)
        # Pre-populate function arguments
        for vid in func_ir.arg_values:
            v = func_ir.values[vid]
            name = v.attrs.get("var_name") or v.name_hint or vid
            self.exprs[vid] = name

    def expr(self, vid: str) -> str:
        """Get a readable expression string for a value."""
        if vid in self.exprs:
            return self.exprs[vid]
        v = self.func_ir.values[vid]
        return v.attrs.get("var_name") or v.name_hint or vid

    def emit(self, indent: int, line: str) -> None:
        self.lines.append("    " * indent + line)

    def register(self, vid: str, expr_str: str) -> None:
        self.exprs[vid] = expr_str


# ---------------------------------------------------------------------------
# Op emission
# ---------------------------------------------------------------------------

def _emit_op(ctx: _ExprCtx, op: Op, indent: int) -> None:
    """Emit a single Python statement for an op."""
    func_ir = ctx.func_ir

    if op.opcode == "const":
        # Already handled in _ExprCtx.__init__
        return

    if op.opcode == "phi":
        return

    if op.opcode == "assign":
        target = op.attrs["target"]
        source = ctx.expr(op.inputs[0])
        ctx.emit(indent, f"{target} = {source}")
        if op.outputs:
            ctx.register(op.outputs[0], target)
        return

    if op.opcode == "binary":
        py_op = _BINARY_OPS.get(op.attrs.get("operator", ""), "?")
        left = ctx.expr(op.inputs[0])
        right = ctx.expr(op.inputs[1])
        expr = f"{left} {py_op} {right}"
        if op.outputs:
            out_name = _out_name(func_ir, op)
            ctx.register(op.outputs[0], expr)
        return

    if op.opcode == "unary":
        py_op = _UNARY_OPS.get(op.attrs.get("operator", ""), "?")
        operand = ctx.expr(op.inputs[0])
        expr = f"{py_op}{operand}"
        if op.outputs:
            ctx.register(op.outputs[0], expr)
        return

    if op.opcode == "compare":
        operators = op.attrs.get("operators", [])
        if len(operators) == 1 and len(op.inputs) == 2:
            py_op = _COMPARE_OPS.get(operators[0], "?")
            left = ctx.expr(op.inputs[0])
            right = ctx.expr(op.inputs[1])
            expr = f"{left} {py_op} {right}"
        else:
            parts = [ctx.expr(op.inputs[0])]
            for i, cmp_op in enumerate(operators):
                py_op = _COMPARE_OPS.get(cmp_op, "?")
                parts.append(f"{py_op} {ctx.expr(op.inputs[i + 1])}")
            expr = " ".join(parts)
        if op.outputs:
            ctx.register(op.outputs[0], expr)
        return

    if op.opcode == "call":
        n_args = op.attrs.get("n_args", 0)
        kwarg_names = op.attrs.get("kwarg_names", [])
        n_kwargs = len(kwarg_names)
        callable_vid = op.inputs[0]
        args = [ctx.expr(v) for v in op.inputs[1 : 1 + n_args]]
        kwargs = [ctx.expr(v) for v in op.inputs[1 + n_args : 1 + n_args + n_kwargs]]
        # Build argument string: positional + keyword
        arg_parts = list(args)
        for kname, kval in zip(kwarg_names, kwargs):
            arg_parts.append(f"{kname}={kval}")
        arg_str = ", ".join(arg_parts)
        # Detect method call pattern: callable defined by get_attr
        callable_val = func_ir.values.get(callable_vid)
        def_op_id = callable_val.def_op if callable_val else None
        def_op = func_ir.ops.get(def_op_id) if def_op_id else None
        if def_op and def_op.opcode == "get_attr":
            obj = ctx.expr(def_op.inputs[0])
            attr = def_op.attrs["attr"]
            expr = f"{obj}.{attr}({arg_str})"
        else:
            func_name = ctx.expr(callable_vid)
            if _needs_call_target_temp(func_name):
                call_target = f"__call_target_{op.id}"
                ctx.emit(indent, f"{call_target} = {func_name}")
                func_name = call_target
            expr = f"{func_name}({arg_str})"
        if op.outputs:
            if len(op.outputs) > 1:
                # Multi-output call: assign each output to a named variable.
                # Outputs with use_ops get an auto-name; unused ones get `_`.
                lhs_parts: list[str] = []
                any_used = False
                for i, vid in enumerate(op.outputs):
                    out_val = func_ir.values.get(vid)
                    used = bool(out_val and out_val.use_ops)
                    if used:
                        vname = (out_val.attrs.get("var_name") if out_val else None) or \
                                (out_val.name_hint if out_val else None) or f"_call_r{i}"
                        ctx.register(vid, vname)
                        lhs_parts.append(vname)
                        any_used = True
                    else:
                        lhs_parts.append("_")
                if any_used:
                    ctx.emit(indent, f"{', '.join(lhs_parts)} = {expr}")
                else:
                    ctx.emit(indent, expr)
            else:
                out_val = func_ir.values.get(op.outputs[0])
                var = out_val.attrs.get("var_name") if out_val else None
                used = bool(out_val and out_val.use_ops)
                if var:
                    ctx.emit(indent, f"{var} = {expr}")
                    ctx.register(op.outputs[0], var)
                elif used:
                    ctx.register(op.outputs[0], expr)
                else:
                    ctx.emit(indent, expr)
        else:
            ctx.emit(indent, expr)
        return

    if op.opcode == "get_attr":
        obj = ctx.expr(op.inputs[0])
        attr = op.attrs["attr"]
        expr = f"{obj}.{attr}"
        if op.outputs:
            ctx.register(op.outputs[0], expr)
        return

    if op.opcode == "set_attr":
        obj = ctx.expr(op.inputs[0])
        attr = op.attrs["attr"]
        val = ctx.expr(op.inputs[1])
        ctx.emit(indent, f"{obj}.{attr} = {val}")
        return

    if op.opcode == "get_item":
        obj = ctx.expr(op.inputs[0])
        key = ctx.expr(op.inputs[1])
        if _needs_call_target_temp(obj):
            target = f"__getitem_target_{op.id}"
            ctx.emit(indent, f"{target} = {obj}")
            obj = target
        expr = f"{obj}[{key}]"
        if op.outputs:
            out_val = func_ir.values.get(op.outputs[0])
            var = out_val.attrs.get("var_name") if out_val else None
            if var:
                ctx.emit(indent, f"{var} = {expr}")
                ctx.register(op.outputs[0], var)
            else:
                ctx.register(op.outputs[0], expr)
        return

    if op.opcode == "set_item":
        obj = ctx.expr(op.inputs[0])
        key = ctx.expr(op.inputs[1])
        val = ctx.expr(op.inputs[2])
        if _needs_call_target_temp(obj):
            target = f"__setitem_target_{op.id}"
            ctx.emit(indent, f"{target} = {obj}")
            obj = target
        ctx.emit(indent, f"{obj}[{key}] = {val}")
        return

    if op.opcode == "build_list":
        items = [ctx.expr(v) for v in op.inputs]
        expr = f"[{', '.join(items)}]"
        if op.outputs:
            ctx.register(op.outputs[0], expr)
        return

    if op.opcode == "build_tuple":
        items = [ctx.expr(v) for v in op.inputs]
        if len(items) == 1:
            expr = f"({items[0]},)"
        else:
            expr = f"({', '.join(items)})"
        if op.outputs:
            ctx.register(op.outputs[0], expr)
        return

    if op.opcode == "build_dict":
        n_items = op.attrs.get("n_items", 0)
        pairs = []
        for i in range(n_items):
            key = ctx.expr(op.inputs[2 * i])
            val = ctx.expr(op.inputs[2 * i + 1])
            pairs.append(f"{key}: {val}")
        expr = "{" + ", ".join(pairs) + "}"
        if op.outputs:
            ctx.register(op.outputs[0], expr)
        return

    if op.opcode == "append":
        obj = ctx.expr(op.inputs[0])
        val = ctx.expr(op.inputs[1])
        ctx.emit(indent, f"{obj}.append({val})")
        return

    if op.opcode == "pop":
        obj = ctx.expr(op.inputs[0])
        if len(op.inputs) > 1:
            idx = ctx.expr(op.inputs[1])
            expr = f"{obj}.pop({idx})"
        else:
            expr = f"{obj}.pop()"
        if op.outputs:
            out_val = func_ir.values.get(op.outputs[0])
            var = out_val.attrs.get("var_name") if out_val else None
            if var:
                ctx.emit(indent, f"{var} = {expr}")
                ctx.register(op.outputs[0], var)
            else:
                ctx.register(op.outputs[0], expr)
        return

    if op.opcode == "iter_init":
        iterable = ctx.expr(op.inputs[0])
        expr = f"iter({iterable})"
        if op.outputs:
            ctx.register(op.outputs[0], expr)
        return

    if op.opcode == "iter_next":
        iterator = ctx.expr(op.inputs[0])
        expr = f"next({iterator})"
        if op.outputs:
            ctx.register(op.outputs[0], expr)
        return

    if op.opcode == "slot":
        # Render slot as a function call placeholder with arguments
        args = [ctx.expr(v) for v in op.inputs]
        placeholder = f"__slot_{op.id}__"
        expr = f"{placeholder}({', '.join(args)})"
        if op.outputs:
            out_val = func_ir.values.get(op.outputs[0])
            var = out_val.attrs.get("var_name") if out_val else None
            if var:
                ctx.emit(indent, f"{var} = {expr}")
                ctx.register(op.outputs[0], var)
            else:
                ctx.register(op.outputs[0], expr)
        else:
            ctx.emit(indent, expr)
        return

    # Fallback: emit as comment
    ctx.emit(indent, f"# {op.opcode} {op.id}")


def _out_name(func_ir: FunctionIR, op: Op) -> str:
    """Get a readable output variable name for an op."""
    if op.outputs:
        v = func_ir.values.get(op.outputs[0])
        if v:
            return v.attrs.get("var_name") or v.name_hint or op.outputs[0]
    return op.id


def _needs_call_target_temp(expr: str) -> bool:
    """Avoid parser warnings for odd call targets like numeric literals."""
    if not expr:
        return True
    stripped = expr.strip()
    if stripped.isidentifier():
        return False
    if stripped.startswith("_") and stripped.replace("_", "").isalnum():
        return False
    # Common simple attribute / indexing forms are safe as direct call targets.
    if stripped[0].isalpha() or stripped[0] == "_":
        return False
    return True


# ---------------------------------------------------------------------------
# Control flow reconstruction
# ---------------------------------------------------------------------------

def _find_merge_block(func_ir: FunctionIR, true_id: str, false_id: str) -> str | None:
    """Find the continuation block after an if/else structure."""
    # Collect all blocks reachable from true branch
    true_reachable: set[str] = set()
    stack = [true_id]
    while stack:
        bid = stack.pop()
        if bid in true_reachable:
            continue
        true_reachable.add(bid)
        stack.extend(func_ir.blocks[bid].succs)

    # BFS from false branch to find first common block
    false_visited: set[str] = set()
    queue = list(func_ir.blocks[false_id].succs)
    while queue:
        bid = queue.pop(0)
        if bid in false_visited:
            continue
        false_visited.add(bid)
        if bid in true_reachable and bid != true_id and bid != false_id:
            return bid
        queue.extend(func_ir.blocks[bid].succs)

    # No common merge — one branch returns. The continuation is
    # wherever the other branch eventually leads.
    false_block = func_ir.blocks[false_id]
    if false_block.succs:
        return false_block.succs[0]
    true_block = func_ir.blocks[true_id]
    if true_block.succs:
        return true_block.succs[0]
    return None


def _block_has_content(func_ir: FunctionIR, block_id: str) -> bool:
    """Check if a block has meaningful ops (not just phi/jump)."""
    block = func_ir.blocks[block_id]
    for op_id in block.op_ids:
        op = func_ir.ops[op_id]
        if op.opcode not in ("phi", "jump"):
            return True
    return False


def _emit_block(
    ctx: _ExprCtx,
    block_id: str,
    indent: int,
    stop_at: frozenset[str],
) -> None:
    """Recursively emit Python code for a block and its successors."""
    if block_id in stop_at:
        return
    func_ir = ctx.func_ir
    if block_id not in func_ir.blocks:
        return
    block = func_ir.blocks[block_id]

    for op_id in block.op_ids:
        op = func_ir.ops[op_id]

        if op.opcode in ("phi",):
            # Register phi output with var_name for readability
            var = op.attrs.get("var_name")
            if var and op.outputs:
                ctx.register(op.outputs[0], var)
            continue

        if op.opcode == "return":
            if op.inputs:
                ctx.emit(indent, f"return {ctx.expr(op.inputs[0])}")
            else:
                ctx.emit(indent, "return")
            return

        if op.opcode == "jump":
            target = op.attrs["target"]
            if target in stop_at:
                return
            _emit_block(ctx, target, indent, stop_at)
            return

        if op.opcode == "branch":
            _emit_branch(ctx, op, block_id, indent, stop_at)
            return

        _emit_op(ctx, op, indent)


def _emit_branch(
    ctx: _ExprCtx,
    op: Op,
    current_block: str,
    indent: int,
    stop_at: frozenset[str],
) -> None:
    """Handle a branch op: emit while loop or if/else."""
    func_ir = ctx.func_ir
    cond = ctx.expr(op.inputs[0])
    true_target = op.attrs["true"]
    false_target = op.attrs["false"]

    # While loop pattern: true branch goes to while_body
    if true_target.startswith("b_while_body") or true_target.startswith("b_for_body"):
        ctx.emit(indent, f"while {cond}:")
        # Process body; stop at current block (back-edge to while_test)
        _emit_block(ctx, true_target, indent + 1, stop_at | frozenset([current_block]))
        # Continue after the while with the exit block
        _emit_block(ctx, false_target, indent, stop_at)
    else:
        # If/else pattern
        merge = _find_merge_block(func_ir, true_target, false_target)
        inner_stop = stop_at | (frozenset([merge]) if merge else frozenset())

        ctx.emit(indent, f"if {cond}:")
        _emit_block(ctx, true_target, indent + 1, inner_stop)

        if _block_has_content(func_ir, false_target):
            ctx.emit(indent, "else:")
            _emit_block(ctx, false_target, indent + 1, inner_stop)

        if merge:
            _emit_block(ctx, merge, indent, stop_at)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def emit_python_source(func_ir: FunctionIR) -> str:
    """Reconstruct readable Python source code from a FunctionIR."""
    ctx = _ExprCtx(func_ir)

    # Function header
    params: list[str] = []
    for vid in func_ir.arg_values:
        v = func_ir.values[vid]
        name = v.attrs.get("var_name") or v.name_hint or vid
        hint = v.type_hint
        if hint and hint != "object":
            params.append(f"{name}: {hint}")
        else:
            params.append(name)
    ctx.emit(0, f"def {func_ir.name}({', '.join(params)}):")

    # Walk from entry block
    _emit_block(ctx, func_ir.entry_block, 1, frozenset())

    # Ensure function body is not empty
    if len(ctx.lines) == 1:
        ctx.emit(1, "pass")

    return "\n".join(ctx.lines)


# ---------------------------------------------------------------------------
# C++ opcode emission
# ---------------------------------------------------------------------------

# Opcode constants — must match C++ ir_eval.h
class CppOp:
    CONST_F64     = 0
    LOAD_ARG      = 1
    ADD           = 2
    SUB           = 3
    MUL           = 4
    DIV           = 5
    SQRT          = 6
    ABS           = 7
    NEG           = 8
    EXP           = 9
    LOG           = 10
    TANH          = 11
    MIN           = 12
    MAX           = 13
    LT            = 14
    GT            = 15
    LE            = 16
    GE            = 17
    EQ            = 18
    IF_START      = 19
    ELSE          = 20
    ENDIF         = 21
    WHILE_START   = 22
    WHILE_END     = 23
    RETURN        = 24
    SAFE_DIV      = 25
    SAFE_LOG      = 26
    SAFE_SQRT     = 27
    NE            = 28
    NOT           = 29
    DUP           = 30
    POP           = 31
    NOP           = 32


_BINARY_TO_CPP = {
    "Add": CppOp.ADD,
    "Sub": CppOp.SUB,
    "Mult": CppOp.MUL,
    "Div": CppOp.SAFE_DIV,
    "FloorDiv": CppOp.SAFE_DIV,
    "Mod": CppOp.NOP,
    "Pow": CppOp.NOP,
}

_COMPARE_TO_CPP = {
    "Lt": CppOp.LT,
    "Gt": CppOp.GT,
    "LtE": CppOp.LE,
    "GtE": CppOp.GE,
    "Eq": CppOp.EQ,
    "NotEq": CppOp.NE,
}

_UNARY_TO_CPP = {
    "USub": CppOp.NEG,
    "Not": CppOp.NOT,
}


def _encode_f64(value: float) -> tuple[int, int]:
    """Encode a float64 as two int32 words (little-endian)."""
    raw = struct.pack("<d", value)
    lo, hi = struct.unpack("<ii", raw)
    return lo, hi


def emit_cpp_ops(func_ir: FunctionIR) -> list[int]:
    """Serialize a FunctionIR to a flat C++ opcode array.

    Uses a recursive tree-walk from return ops: each value is emitted by
    recursively emitting its defining operation. This ensures correct
    stack ordering for the stack-based C++ evaluator (ir_eval.h).
    """
    ops: list[int] = []
    arg_index = {vid: i for i, vid in enumerate(func_ir.arg_values)}
    visited_blocks: set[str] = set()

    def _emit_block(bid: str) -> None:
        if bid in visited_blocks:
            return
        visited_blocks.add(bid)
        block = func_ir.blocks[bid]
        for op_id in block.op_ids:
            op = func_ir.ops.get(op_id)
            if not op:
                continue
            if op.opcode == "return":
                if op.inputs:
                    _push_value(func_ir, op.inputs[0], arg_index, ops)
                else:
                    lo, hi = _encode_f64(0.0)
                    ops.extend([CppOp.CONST_F64, lo, hi])
                ops.append(CppOp.RETURN)
            elif op.opcode == "branch":
                if (op.inputs
                        and op.attrs.get("true")
                        and op.attrs.get("false")):
                    _push_value(func_ir, op.inputs[0], arg_index, ops)
                    ops.append(CppOp.IF_START)
                    _emit_block(op.attrs["true"])
                    ops.append(CppOp.ELSE)
                    _emit_block(op.attrs["false"])
                    ops.append(CppOp.ENDIF)
            elif op.opcode == "jump":
                target = op.attrs.get("target")
                if target:
                    _emit_block(target)
            # All other ops are pulled in recursively by _push_value

    _emit_block(func_ir.entry_block)

    if not ops or ops[-1] != CppOp.RETURN:
        lo, hi = _encode_f64(0.0)
        ops.extend([CppOp.CONST_F64, lo, hi, CppOp.RETURN])

    return ops


def _push_value(
    func_ir: FunctionIR,
    vid: str,
    arg_index: dict[str, int],
    ops: list[int],
) -> None:
    """Recursively push a value onto the C++ stack.

    For function arguments: emits LOAD_ARG.
    For constants: emits CONST_F64.
    For computed values: recursively emits the defining operation
    (which in turn pushes *its* inputs first).
    """
    # Function argument
    if vid in arg_index:
        ops.extend([CppOp.LOAD_ARG, arg_index[vid]])
        return

    # Look up the defining operation
    v = func_ir.values.get(vid)
    if not (v and v.def_op):
        lo, hi = _encode_f64(0.0)
        ops.extend([CppOp.CONST_F64, lo, hi])
        return

    def_op = func_ir.ops.get(v.def_op)
    if def_op is None:
        lo, hi = _encode_f64(0.0)
        ops.extend([CppOp.CONST_F64, lo, hi])
        return

    # --- Dispatch on defining op type ---

    if def_op.opcode == "const":
        lit = def_op.attrs.get("literal")
        if lit is not None:
            val = float(lit) if not isinstance(lit, bool) else (1.0 if lit else 0.0)
            lo, hi = _encode_f64(val)
            ops.extend([CppOp.CONST_F64, lo, hi])
        return

    if def_op.opcode == "binary":
        operator = def_op.attrs.get("operator", "Add")
        _push_value(func_ir, def_op.inputs[0], arg_index, ops)
        _push_value(func_ir, def_op.inputs[1], arg_index, ops)
        ops.append(_BINARY_TO_CPP.get(operator, CppOp.ADD))
        return

    if def_op.opcode == "compare":
        operators = def_op.attrs.get("operators", ["Lt"])
        operator = operators[0] if operators else "Lt"
        _push_value(func_ir, def_op.inputs[0], arg_index, ops)
        _push_value(func_ir, def_op.inputs[1], arg_index, ops)
        ops.append(_COMPARE_TO_CPP.get(operator, CppOp.LT))
        return

    if def_op.opcode == "unary":
        operator = def_op.attrs.get("operator", "USub")
        _push_value(func_ir, def_op.inputs[0], arg_index, ops)
        ops.append(_UNARY_TO_CPP.get(operator, CppOp.NEG))
        return

    if def_op.opcode == "call":
        name = def_op.attrs.get("name", "")
        if name in ("abs", "fabs"):
            _push_value(func_ir, def_op.inputs[0], arg_index, ops)
            ops.append(CppOp.ABS)
        elif name in ("sqrt", "math.sqrt"):
            _push_value(func_ir, def_op.inputs[0], arg_index, ops)
            ops.append(CppOp.SAFE_SQRT)
        elif name in ("log", "math.log"):
            _push_value(func_ir, def_op.inputs[0], arg_index, ops)
            ops.append(CppOp.SAFE_LOG)
        elif name in ("exp", "math.exp"):
            _push_value(func_ir, def_op.inputs[0], arg_index, ops)
            ops.append(CppOp.EXP)
        elif name in ("tanh", "math.tanh"):
            _push_value(func_ir, def_op.inputs[0], arg_index, ops)
            ops.append(CppOp.TANH)
        elif name in ("min",):
            _push_value(func_ir, def_op.inputs[0], arg_index, ops)
            _push_value(func_ir, def_op.inputs[1], arg_index, ops)
            ops.append(CppOp.MIN)
        elif name in ("max",):
            _push_value(func_ir, def_op.inputs[0], arg_index, ops)
            _push_value(func_ir, def_op.inputs[1], arg_index, ops)
            ops.append(CppOp.MAX)
        elif name in ("_safe_div",):
            _push_value(func_ir, def_op.inputs[0], arg_index, ops)
            _push_value(func_ir, def_op.inputs[1], arg_index, ops)
            ops.append(CppOp.SAFE_DIV)
        elif name in ("_safe_log",):
            _push_value(func_ir, def_op.inputs[0], arg_index, ops)
            ops.append(CppOp.SAFE_LOG)
        elif name in ("_safe_sqrt",):
            _push_value(func_ir, def_op.inputs[0], arg_index, ops)
            ops.append(CppOp.SAFE_SQRT)
        else:
            lo, hi = _encode_f64(0.0)
            ops.extend([CppOp.CONST_F64, lo, hi])
        return

    # Unknown defining op → push 0.0
    lo, hi = _encode_f64(0.0)
    ops.extend([CppOp.CONST_F64, lo, hi])
