#!/usr/bin/env python
"""
Algorithm-IR 交互式教程
========================
展示 Python 源码 → IR 编译 → 代码回生成 的完整流程。
每一步都用 ASCII 图可视化 IR 内部结构。

运行方式:
    cd research/algorithm-IR
    python ir_tutorial_demo.py
"""
from __future__ import annotations

import io
import math
import os
import pathlib
import struct
import sys
import textwrap

# Ensure UTF-8 output on Windows terminals (avoids GBK encoding errors)
if sys.platform == "win32" and not os.environ.get("PYTHONIOENCODING"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm_ir.frontend import compile_source_to_ir
from algorithm_ir.ir import render_function_ir, validate_function_ir
from algorithm_ir.ir.model import FunctionIR, Op
from algorithm_ir.regeneration.codegen import emit_python_source, emit_cpp_ops, CppOp

# ---------------------------------------------------------------------------
# Helper namespace for safe math functions
# ---------------------------------------------------------------------------
_HELPERS = {
    "__builtins__": __builtins__,
    "_safe_div": lambda a, b: a / b if abs(b) > 1e-30 else 0.0,
    "_safe_log": lambda a: math.log(max(a, 1e-30)),
    "_safe_sqrt": lambda a: math.sqrt(max(a, 0.0)),
    "abs": abs,
    "math": math,
}

# ---------------------------------------------------------------------------
# Printing utilities
# ---------------------------------------------------------------------------
_WIDTH = 90

def banner(title: str) -> None:
    print()
    print("=" * _WIDTH)
    print(f"  {title}")
    print("=" * _WIDTH)

def section(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, _WIDTH - len(title) - 5))

def box(lines: list[str], title: str = "") -> str:
    """Draw an ASCII box around lines."""
    max_w = max((len(l) for l in lines), default=0)
    if title:
        max_w = max(max_w, len(title) + 2)
    border = "+" + "-" * (max_w + 2) + "+"
    result = [border]
    if title:
        result.append(f"| {title:<{max_w}} |")
        result.append("+" + "-" * (max_w + 2) + "+")
    for l in lines:
        result.append(f"| {l:<{max_w}} |")
    result.append(border)
    return "\n".join(result)

def indent_block(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + l for l in text.splitlines())

# ---------------------------------------------------------------------------
# ASCII IR Visualization
# ---------------------------------------------------------------------------

def ascii_value(func_ir: FunctionIR, vid: str) -> str:
    """Compact string for a Value."""
    v = func_ir.values[vid]
    name = v.name_hint or ""
    tpe = v.type_hint or "?"
    if name:
        return f"{vid}({name}:{tpe})"
    return f"{vid}(:{tpe})"


def ascii_op(func_ir: FunctionIR, op: Op) -> str:
    """One-line op summary."""
    inputs_str = ", ".join(ascii_value(func_ir, i) for i in op.inputs)
    outputs_str = ", ".join(ascii_value(func_ir, o) for o in op.outputs)

    extra = ""
    if op.opcode == "binary":
        extra = f" [{op.attrs.get('operator', '?')}]"
    elif op.opcode == "unary":
        extra = f" [{op.attrs.get('operator', '?')}]"
    elif op.opcode == "compare":
        extra = f" [{op.attrs.get('operators', ['?'])}]"
    elif op.opcode == "const":
        lit = op.attrs.get("literal")
        name = op.attrs.get("name", "")
        if name:
            extra = f" name={name}"
        elif lit is not None:
            extra = f" = {lit!r}"
    elif op.opcode == "call":
        extra = f" fn={op.attrs.get('name', '?')}"
    elif op.opcode == "branch":
        extra = f" true→{op.attrs.get('true','?')} false→{op.attrs.get('false','?')}"
    elif op.opcode == "jump":
        extra = f" →{op.attrs.get('target','?')}"
    elif op.opcode == "phi":
        extra = f" var={op.attrs.get('var_name','?')}"
    elif op.opcode == "assign":
        extra = f" var={op.attrs.get('target','?')}"

    return f"{op.id}: {op.opcode}{extra}  ({inputs_str}) → ({outputs_str})"


def ascii_block_box(func_ir: FunctionIR, block_id: str) -> str:
    """Render a block as an ASCII box."""
    block = func_ir.blocks[block_id]
    lines = []
    for op_id in block.op_ids:
        op = func_ir.ops[op_id]
        lines.append(ascii_op(func_ir, op))
    title = f"Block: {block_id}"
    if block.preds:
        title += f"  ← from {block.preds}"
    return box(lines if lines else ["(empty)"], title)


def ascii_cfg(func_ir: FunctionIR) -> str:
    """Draw the control flow graph as connected ASCII block boxes."""
    parts = []
    block_order = _topo_order(func_ir)
    for bid in block_order:
        b = func_ir.blocks[bid]
        parts.append(ascii_block_box(func_ir, bid))
        if b.succs:
            arrow = "        │"
            parts.append(arrow)
            targets = ", ".join(b.succs)
            parts.append(f"        ▼  ({targets})")
            parts.append("")
    return "\n".join(parts)


def _topo_order(func_ir: FunctionIR) -> list[str]:
    """Topological order of blocks starting from entry."""
    order = []
    visited = set()
    def dfs(bid):
        if bid in visited:
            return
        visited.add(bid)
        order.append(bid)
        for s in func_ir.blocks[bid].succs:
            dfs(s)
    dfs(func_ir.entry_block)
    # Add any unreachable blocks at the end
    for bid in func_ir.blocks:
        if bid not in visited:
            order.append(bid)
    return order


# ---------------------------------------------------------------------------
# Def-Use Chain Visualization
# ---------------------------------------------------------------------------

def ascii_def_use(func_ir: FunctionIR) -> str:
    """Draw definition-use chains as an ASCII diagram."""
    lines = []
    lines.append("定义-使用链 (Definition-Use Chains):")
    lines.append("")

    for vid in func_ir.arg_values:
        v = func_ir.values[vid]
        name = v.name_hint or vid
        users = []
        for uid in v.use_ops:
            op = func_ir.ops.get(uid)
            if op:
                users.append(f"{uid}({op.opcode})")
        lines.append(f"  [参数] {vid}({name}) ──uses──→ {', '.join(users) if users else '(未使用)'}")

    lines.append("")

    for vid, v in func_ir.values.items():
        if vid in func_ir.arg_values:
            continue
        name = v.name_hint or ""
        def_str = v.def_op or "(无)"
        def_op = func_ir.ops.get(v.def_op) if v.def_op else None
        def_desc = f"{v.def_op}({def_op.opcode})" if def_op else "(无)"
        users = []
        for uid in v.use_ops:
            op = func_ir.ops.get(uid)
            if op:
                users.append(f"{uid}({op.opcode})")
        label = f"({name})" if name else ""
        lines.append(f"  {def_desc} ──def──→ {vid}{label} ──uses──→ {', '.join(users) if users else '(未使用)'}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Expression Tree Visualization
# ---------------------------------------------------------------------------

def ascii_expr_tree(func_ir: FunctionIR, vid: str, depth: int = 0) -> list[str]:
    """Build an ASCII expression tree from a value, walking backwards through defs."""
    prefix = "    " * depth
    connector = "├── " if depth > 0 else ""

    v = func_ir.values.get(vid)
    if not v:
        return [f"{prefix}{connector}??? ({vid})"]

    # Argument node
    if vid in func_ir.arg_values:
        name = v.name_hint or vid
        return [f"{prefix}{connector}📥 ARG: {name}"]

    # Get defining op
    if not v.def_op:
        return [f"{prefix}{connector}??? (no def_op for {vid})"]

    op = func_ir.ops.get(v.def_op)
    if not op:
        return [f"{prefix}{connector}??? (missing op {v.def_op})"]

    # Leaf: constant
    if op.opcode == "const":
        lit = op.attrs.get("literal")
        name = op.attrs.get("name", "")
        if name:
            return [f"{prefix}{connector}📌 CONST: {name}"]
        return [f"{prefix}{connector}📌 CONST: {lit!r}"]

    # Internal: operation
    label = op.opcode
    if op.opcode == "binary":
        sym = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/"}.get(op.attrs.get("operator", ""), "?")
        label = f"binary [{sym}]"
    elif op.opcode == "unary":
        label = f"unary [{op.attrs.get('operator', '?')}]"
    elif op.opcode == "call":
        label = f"call [{op.attrs.get('name', '?')}]"
    elif op.opcode == "compare":
        ops_list = op.attrs.get("operators", [])
        label = f"compare [{ops_list}]"

    lines = [f"{prefix}{connector}🔧 {label}  →  {vid}"]
    for i, inp_vid in enumerate(op.inputs):
        child_prefix = "    " * (depth + 1)
        is_last = (i == len(op.inputs) - 1)
        child_conn = "└── " if is_last else "├── "
        child_lines = ascii_expr_tree(func_ir, inp_vid, depth + 1)
        # Replace the connector for proper tree shape
        if child_lines:
            first = child_lines[0]
            # Replace generic connector
            first = child_prefix + child_conn + first.lstrip().lstrip("├── ").lstrip("└── ")
            child_lines[0] = first
        lines.extend(child_lines)

    return lines


def print_expr_tree(func_ir: FunctionIR) -> None:
    """Print the expression tree(s) for all return values."""
    print("表达式树 (Expression Tree) — 从返回值回溯:")
    print()
    for op in func_ir.ops.values():
        if op.opcode == "return" and op.inputs:
            ret_vid = op.inputs[0]
            tree_lines = _build_tree(func_ir, ret_vid, "", True)
            for line in tree_lines:
                print(f"    {line}")
            print()


def _build_tree(func_ir: FunctionIR, vid: str, prefix: str, is_last: bool) -> list[str]:
    """Recursively build an expression tree with proper box-drawing characters."""
    v = func_ir.values.get(vid)
    if not v:
        return [f"{prefix}{'└── ' if is_last else '├── '}??? ({vid})"]

    connector = "└── " if is_last else "├── "
    extension = "    " if is_last else "│   "

    # Argument
    if vid in func_ir.arg_values:
        name = v.name_hint or vid
        return [f"{prefix}{connector}📥 ARG({name})"]

    # No defining op
    if not v.def_op:
        return [f"{prefix}{connector}??? {vid}"]

    op = func_ir.ops.get(v.def_op)
    if not op:
        return [f"{prefix}{connector}??? {vid}"]

    # Constant
    if op.opcode == "const":
        lit = op.attrs.get("literal")
        name = op.attrs.get("name", "")
        if name:
            return [f"{prefix}{connector}📌 CONST({name})"]
        return [f"{prefix}{connector}📌 {lit!r}"]

    # Operation node
    label = _op_label(op)
    lines = [f"{prefix}{connector}🔧 {label}"]

    child_prefix = prefix + extension
    for i, inp_vid in enumerate(op.inputs):
        child_is_last = (i == len(op.inputs) - 1)
        lines.extend(_build_tree(func_ir, inp_vid, child_prefix, child_is_last))

    return lines


def _op_label(op: Op) -> str:
    """Human-readable label for an op."""
    if op.opcode == "binary":
        sym = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/",
               "FloorDiv": "//", "Mod": "%", "Pow": "**"}.get(
            op.attrs.get("operator", ""), "?")
        return f"{sym}  (binary)"
    if op.opcode == "unary":
        sym = {"USub": "-", "Not": "not"}.get(op.attrs.get("operator", ""), "?")
        return f"{sym}  (unary)"
    if op.opcode == "call":
        return f"{op.attrs.get('name', '?')}()  (call)"
    if op.opcode == "compare":
        ops = op.attrs.get("operators", [])
        sym = {"Lt": "<", "Gt": ">", "LtE": "<=", "GtE": ">=",
               "Eq": "==", "NotEq": "!="}.get(ops[0] if ops else "", "?")
        return f"{sym}  (compare)"
    if op.opcode == "assign":
        return f"=  (assign to {op.attrs.get('target', '?')})"
    if op.opcode == "phi":
        return f"φ  (merge '{op.attrs.get('var_name', '?')}')"
    return op.opcode


# ---------------------------------------------------------------------------
# C++ Opcode Visualization
# ---------------------------------------------------------------------------

_OPCODE_NAMES = {v: k for k, v in vars(CppOp).items()
                 if isinstance(v, int) and not k.startswith("_")}

def ascii_cpp_ops(cpp_ops: list[int]) -> str:
    """Render C++ opcodes as an annotated stack trace."""
    lines = []
    lines.append(f"C++ 操作码序列 ({len(cpp_ops)} 个整数)")
    lines.append("")

    i = 0
    stack_desc = []  # Symbolic stack tracking
    while i < len(cpp_ops):
        op = cpp_ops[i]
        name = _OPCODE_NAMES.get(op, f"UNKNOWN({op})")

        if op == CppOp.CONST_F64 and i + 2 < len(cpp_ops):
            lo, hi = cpp_ops[i + 1], cpp_ops[i + 2]
            # Handle potential signed/unsigned int32 encoding
            try:
                raw = struct.pack("<II", lo & 0xFFFFFFFF, hi & 0xFFFFFFFF)
                val = struct.unpack("<d", raw)[0]
            except struct.error:
                val = 0.0
            stack_desc.append(f"{val:g}")
            stack_str = " ".join(f"[{s}]" for s in stack_desc)
            lines.append(f"  [{i:3d}] CONST_F64  {val:g}")
            lines.append(f"         栈: {stack_str}")
            i += 3
        elif op == CppOp.LOAD_ARG and i + 1 < len(cpp_ops):
            idx = cpp_ops[i + 1]
            stack_desc.append(f"arg{idx}")
            stack_str = " ".join(f"[{s}]" for s in stack_desc)
            lines.append(f"  [{i:3d}] LOAD_ARG   index={idx}")
            lines.append(f"         栈: {stack_str}")
            i += 2
        elif op == CppOp.RETURN:
            result = stack_desc[-1] if stack_desc else "?"
            lines.append(f"  [{i:3d}] RETURN     → 返回 {result}")
            i += 1
        elif op in (CppOp.ADD, CppOp.SUB, CppOp.MUL, CppOp.SAFE_DIV):
            sym = {CppOp.ADD: "+", CppOp.SUB: "-", CppOp.MUL: "*", CppOp.SAFE_DIV: "÷"}.get(op, "?")
            if len(stack_desc) >= 2:
                b = stack_desc.pop()
                a = stack_desc.pop()
                result = f"({a}{sym}{b})"
                stack_desc.append(result)
            stack_str = " ".join(f"[{s}]" for s in stack_desc)
            lines.append(f"  [{i:3d}] {name:10s} {sym}")
            lines.append(f"         栈: {stack_str}")
            i += 1
        elif op in (CppOp.ABS, CppOp.NEG, CppOp.SQRT, CppOp.SAFE_LOG, CppOp.SAFE_SQRT,
                     CppOp.EXP, CppOp.LOG, CppOp.TANH, CppOp.NOT):
            fn_name = {CppOp.ABS: "abs", CppOp.NEG: "-", CppOp.SQRT: "sqrt",
                       CppOp.SAFE_LOG: "safe_log", CppOp.SAFE_SQRT: "safe_sqrt",
                       CppOp.EXP: "exp", CppOp.LOG: "log", CppOp.TANH: "tanh",
                       CppOp.NOT: "not"}.get(op, "?")
            if stack_desc:
                a = stack_desc.pop()
                stack_desc.append(f"{fn_name}({a})")
            stack_str = " ".join(f"[{s}]" for s in stack_desc)
            lines.append(f"  [{i:3d}] {name:10s}")
            lines.append(f"         栈: {stack_str}")
            i += 1
        elif op in (CppOp.LT, CppOp.GT, CppOp.LE, CppOp.GE, CppOp.EQ, CppOp.NE):
            sym = {CppOp.LT: "<", CppOp.GT: ">", CppOp.LE: "<=",
                   CppOp.GE: ">=", CppOp.EQ: "==", CppOp.NE: "!="}.get(op, "?")
            if len(stack_desc) >= 2:
                b = stack_desc.pop()
                a = stack_desc.pop()
                stack_desc.append(f"({a}{sym}{b})")
            stack_str = " ".join(f"[{s}]" for s in stack_desc)
            lines.append(f"  [{i:3d}] {name:10s} {sym}")
            lines.append(f"         栈: {stack_str}")
            i += 1
        elif op in (CppOp.MIN, CppOp.MAX):
            fn = "min" if op == CppOp.MIN else "max"
            if len(stack_desc) >= 2:
                b = stack_desc.pop()
                a = stack_desc.pop()
                stack_desc.append(f"{fn}({a},{b})")
            stack_str = " ".join(f"[{s}]" for s in stack_desc)
            lines.append(f"  [{i:3d}] {name:10s}")
            lines.append(f"         栈: {stack_str}")
            i += 1
        elif op == CppOp.IF_START:
            lines.append(f"  [{i:3d}] IF_START   (弹出条件值，如果为假跳到ELSE)")
            if stack_desc:
                stack_desc.pop()
            i += 1
        elif op == CppOp.ELSE:
            lines.append(f"  [{i:3d}] ELSE       (跳到ENDIF)")
            i += 1
        elif op == CppOp.ENDIF:
            lines.append(f"  [{i:3d}] ENDIF")
            i += 1
        else:
            lines.append(f"  [{i:3d}] {name}")
            i += 1

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo: Skeleton → Random Program → Compile → Visualize flow
# ---------------------------------------------------------------------------

def demo_skeleton_flow():
    """Show the complete flow from skeleton definition to compiled code."""
    from evolution.skeleton_registry import ProgramSpec, SkeletonSpec, SkeletonRegistry
    from evolution.random_program import random_ir_program
    import numpy as np

    banner("骨架 (Skeleton) → 随机程序 → IR → 代码生成 全流程演示")

    # =====================================================================
    # Step 1: Skeleton Definition
    # =====================================================================
    section("步骤 1: 定义骨架 (Skeleton)")
    print("""
骨架定义了「需要进化什么程序」——每个程序的名字、参数、类型。
这是进化框架与具体应用之间的桥梁。

以 MIMO BP 检测器为例，我们需要 4 个小程序：
""")

    spec = SkeletonSpec(
        skeleton_id="mimo_bp_detector",
        program_specs=[
            ProgramSpec(
                name="f_down",
                param_names=["parent_m_down", "local_dist"],
                param_types=["float", "float"],
                return_type="float",
                constraints={"min_depth": 1, "max_depth": 6},
            ),
            ProgramSpec(
                name="f_up",
                param_names=["sum_child_ld", "sum_child_m_up", "n_children"],
                param_types=["float", "float", "float"],
                return_type="float",
                constraints={"min_depth": 1, "max_depth": 6},
            ),
            ProgramSpec(
                name="f_belief",
                param_names=["cum_dist", "m_down", "m_up"],
                param_types=["float", "float", "float"],
                return_type="float",
                constraints={"min_depth": 1, "max_depth": 6},
            ),
            ProgramSpec(
                name="h_halt",
                param_names=["old_root_m_up", "new_root_m_up"],
                param_types=["float", "float"],
                return_type="float",
                constraints={"min_depth": 1, "max_depth": 4},
            ),
        ],
    )

    registry = SkeletonRegistry()
    registry.register(spec)

    print("注册的骨架:")
    print(box([
        f"骨架 ID: {spec.skeleton_id}",
        f"模式:    {spec.mode}",
        f"角色数:  {len(spec.program_specs)}",
    ], "SkeletonSpec"))

    print()
    for ps in spec.program_specs:
        params = ", ".join(f"{n}: {t}" for n, t in zip(ps.param_names, ps.param_types))
        print(f"  {ps.name}({params}) → {ps.return_type}")
        if ps.constraints:
            print(f"    约束: {ps.constraints}")

    # =====================================================================
    # Step 2: Random Program Generation
    # =====================================================================
    section("步骤 2: 按规格生成随机程序")
    print("""
random_ir_program() 按照 ProgramSpec 的签名，生成随机的 Python 函数源码，
然后编译为 IR。每次调用会产生不同的程序。
""")

    rng = np.random.default_rng(42)

    for ps in spec.program_specs:
        print(f"\n{'─' * 60}")
        print(f"角色: {ps.name}")
        print(f"{'─' * 60}")

        ir = random_ir_program(ps, rng, max_depth=3)
        source = emit_python_source(ir)
        print(f"\n  生成的 Python 源码:")
        print(indent_block(source, "    "))

        # =====================================================================
        # Step 3: Show IR Structure
        # =====================================================================
        print(f"\n  FunctionIR 概览:")
        print(f"    名称:     {ir.name}")
        print(f"    参数值:   {ir.arg_values}")
        print(f"    操作数:   {len(ir.ops)}")
        print(f"    值数:     {len(ir.values)}")
        print(f"    块数:     {len(ir.blocks)}")

        errors = validate_function_ir(ir)
        print(f"    验证:     {'✅ 通过' if not errors else '❌ ' + str(errors)}")

        # =====================================================================
        # Step 4: Detailed IR Dump
        # =====================================================================
        print(f"\n  IR 文本渲染 (render_function_ir):")
        ir_text = render_function_ir(ir)
        print(indent_block(ir_text, "    "))

        # =====================================================================
        # Step 5: ASCII CFG
        # =====================================================================
        print(f"\n  控制流图 (ASCII):")
        cfg = ascii_cfg(ir)
        print(indent_block(cfg, "    "))

        # =====================================================================
        # Step 6: Def-Use Chains
        # =====================================================================
        print(f"\n  " + ascii_def_use(ir).replace("\n", "\n  "))

        # =====================================================================
        # Step 7: Expression Tree
        # =====================================================================
        print(f"\n  表达式树:")
        for op in ir.ops.values():
            if op.opcode == "return" and op.inputs:
                tree_lines = _build_tree(ir, op.inputs[0], "", True)
                for line in tree_lines:
                    print(f"    {line}")

        # =====================================================================
        # Step 8: C++ Opcodes
        # =====================================================================
        cpp = emit_cpp_ops(ir)
        print(f"\n  {ascii_cpp_ops(cpp).replace(chr(10), chr(10) + '  ')}")


def demo_simple_compilation():
    """Show step-by-step compilation of a simple function."""

    banner("演示 A: 简单函数的完整编译流程")

    source = textwrap.dedent("""\
    def score(cum_dist, m_down, m_up):
        return cum_dist + m_down + m_up
    """)

    # =====================================================================
    # Layer 0: Source Code
    # =====================================================================
    section("第 0 层: Python 源代码")
    print("""
这是人类编写的 Python 函数。编译器的任务是把它翻译成结构化的内部表示。
""")
    print(box(source.strip().splitlines(), "Python Source"))

    # =====================================================================
    # Layer 1: AST (Abstract Syntax Tree)
    # =====================================================================
    section("第 1 层: 抽象语法树 (AST)")
    print("""
Python 解析器首先把源码变成抽象语法树 (AST)。
AST 是源码的树形结构表示——每个节点代表一个语法元素。
compile_source_to_ir() 内部首先调用 ast.parse() 得到 AST。
""")
    import ast
    tree = ast.parse(source)
    print(box([
        "FunctionDef: score",
        "  args: [cum_dist, m_down, m_up]",
        "  body:",
        "    Return:",
        "      BinOp:",
        "        left: BinOp:",
        "          left:  Name(cum_dist)",
        "          op:    Add",
        "          right: Name(m_down)",
        "        op:    Add",
        "        right: Name(m_up)",
    ], "AST (简化表示)"))

    # =====================================================================
    # Layer 2: FunctionIR (our IR)
    # =====================================================================
    section("第 2 层: FunctionIR (中间表示)")
    print("""
IRBuilder 遍历 AST，为每个节点生成 IR 操作 (Op)。
关键转换：
  - 每个函数参数 → 一个 Value (arg_value)
  - 每个表达式运算 → 一个 Op + 输出 Value
  - SSA形式：每个值只被定义一次
""")

    func_ir = compile_source_to_ir(source, "score")

    print("  创建的 Value（值）:")
    print("  " + "─" * 70)
    for vid, v in func_ir.values.items():
        role = "参数" if vid in func_ir.arg_values else "计算结果"
        name = v.name_hint or ""
        def_by = v.def_op or "(函数入口)"
        used_by = v.use_ops if v.use_ops else ["(无)"]
        print(f"    {vid:6s} | 名称: {name:12s} | 角色: {role:6s} | 定义者: {def_by:6s} | 使用者: {used_by}")
    print()

    print("  创建的 Op（操作）:")
    print("  " + "─" * 70)
    for op_id, op in func_ir.ops.items():
        print(f"    {ascii_op(func_ir, op)}")
    print()

    print("  控制流图:")
    print(indent_block(ascii_cfg(func_ir), "    "))

    print()
    print("  表达式树 (从 return 回溯):")
    for op in func_ir.ops.values():
        if op.opcode == "return" and op.inputs:
            tree_lines = _build_tree(func_ir, op.inputs[0], "", True)
            for line in tree_lines:
                print(f"    {line}")

    print()
    print("  定义-使用链:")
    print("  " + ascii_def_use(func_ir).replace("\n", "\n  "))

    # =====================================================================
    # Layer 3: xDSL Native Representation
    # =====================================================================
    section("第 3 层: xDSL 原生表示")
    print("""
FunctionIR 内部由 xDSL 框架支撑。xDSL 是一个 Python 的 MLIR 实现，
提供了类型安全的操作定义(IRDL)和经过验证的 IR 操作。

我们定义了 AlgDialect，包含 21 种操作类型（AlgConst, AlgBinary 等）。
xDSL module 是 IR 的"真正来源" (source of truth)，
FunctionIR 的 dict 视图只是为了方便访问。
""")
    xdsl_text = func_ir.attrs.get("xdsl_text", "(无 xDSL 文本)")
    xdsl_lines = xdsl_text.splitlines()
    # Show first 20 lines
    display = xdsl_lines[:20]
    if len(xdsl_lines) > 20:
        display.append(f"... (共 {len(xdsl_lines)} 行)")
    print(box(display, "xDSL IR (部分)"))

    # =====================================================================
    # Layer 4a: Python Regeneration
    # =====================================================================
    section("第 4a 层: IR → Python 回生成")
    print("""
emit_python_source() 将 IR 图重新转换为可读的 Python 代码。
它会重建表达式（内联简单计算）、恢复控制流，生成人类可读的源码。
""")
    regen = emit_python_source(func_ir)
    print(box(regen.strip().splitlines(), "Regenerated Python"))

    print()
    print("对比原始 vs 重生成:")
    print(f"  原始:   def score(cum_dist, m_down, m_up): return cum_dist + m_down + m_up")
    print(f"  重生成: {regen.strip().splitlines()[-1].strip()}")
    print("  语义相同 ✅")

    # =====================================================================
    # Layer 4b: C++ Opcode Generation
    # =====================================================================
    section("第 4b 层: IR → C++ 操作码")
    print("""
emit_cpp_ops() 使用递归树遍历，从 return 操作开始，
向后追溯每个值的定义操作，以正确的顺序生成栈操作码。

C++ 求值器 (ir_eval.h) 是一个栈式计算器：
  - LOAD_ARG: 把函数参数推入栈
  - CONST_F64: 把常量推入栈
  - ADD/SUB/MUL...: 弹出栈顶两个值，计算结果推入栈
  - RETURN: 返回栈顶值
""")

    cpp = emit_cpp_ops(func_ir)
    print(box([f"整数数组: {cpp}"], "Raw C++ Ops"))
    print()
    print(ascii_cpp_ops(cpp))


def demo_branch_compilation():
    """Show compilation of a function with conditional branching."""

    banner("演示 B: 带分支的函数编译")

    source = textwrap.dedent("""\
    def f_down(parent_m_down, local_dist):
        if parent_m_down > 0.5:
            result = parent_m_down + local_dist
        else:
            result = local_dist
        return result
    """)

    section("源代码")
    print(box(source.strip().splitlines(), "Python Source"))

    section("FunctionIR")
    func_ir = compile_source_to_ir(source, "f_down")

    print(f"  块数: {len(func_ir.blocks)}  (有分支所以不止一个块)")
    print()

    print("  控制流图:")
    print(indent_block(ascii_cfg(func_ir), "    "))

    section("分支如何工作")
    print("""
当编译器遇到 if/else 时：
  1. 在当前块末尾生成 branch 操作（条件跳转）
  2. 创建 true 分支块和 false 分支块
  3. 在合并点创建 phi 节点（φ节点）

φ节点 是 SSA 形式的核心概念：当一个变量在两个分支中可能有不同的值时，
合并点用 φ 节点来选择正确的值。

    if (cond):            ┌─ true 分支:  result_1 = ... ──┐
        result = ...      │                                │
    else:                 │                                ├─→ φ(result) = 合并
        result = ...      └─ false 分支: result_2 = ... ──┘
""")

    print("  IR 详情:")
    ir_text = render_function_ir(func_ir)
    print(indent_block(ir_text, "    "))

    section("C++ 操作码 (带 IF/ELSE/ENDIF)")
    cpp = emit_cpp_ops(func_ir)
    print(ascii_cpp_ops(cpp))

    section("回生成的 Python")
    regen = emit_python_source(func_ir)
    print(box(regen.strip().splitlines(), "Regenerated Python"))


def demo_safe_functions():
    """Show how safe math functions compile to IR and C++."""

    banner("演示 C: 安全数学函数")

    print("""
进化过程中可能产生除以零、对负数取对数等操作。
我们提供了安全版本的数学函数：

  _safe_div(a, b)  → 如果 |b| < 1e-30，返回 0
  _safe_log(a)     → log(max(a, 1e-30))
  _safe_sqrt(a)    → sqrt(max(a, 0))
""")

    source = textwrap.dedent("""\
    def f_up(sum_child_ld, sum_child_m_up, n_children):
        return _safe_log(sum_child_ld) + _safe_div(sum_child_m_up, n_children)
    """)

    section("源代码")
    print(box(source.strip().splitlines(), "Python Source"))

    func_ir = compile_source_to_ir(source, "f_up", globals_dict=_HELPERS)

    section("IR 操作列表")
    for op_id, op in func_ir.ops.items():
        print(f"  {ascii_op(func_ir, op)}")

    section("表达式树")
    for op in func_ir.ops.values():
        if op.opcode == "return" and op.inputs:
            tree_lines = _build_tree(func_ir, op.inputs[0], "", True)
            for line in tree_lines:
                print(f"  {line}")

    section("C++ 操作码 (注意 SAFE_LOG 和 SAFE_DIV)")
    cpp = emit_cpp_ops(func_ir)
    print(ascii_cpp_ops(cpp))

    section("回生成的 Python")
    print(emit_python_source(func_ir))


def demo_mutation_flow():
    """Show how IR-level mutation works step by step."""

    banner("演示 D: IR 级变异操作")

    from evolution.operators import mutate_ir
    import numpy as np

    rng = np.random.default_rng(123)

    source = textwrap.dedent("""\
    def f_belief(cum_dist, m_down, m_up):
        return cum_dist + m_down * 2.5
    """)

    original = compile_source_to_ir(source, "f_belief")

    section("原始 IR")
    print(f"  源码: {emit_python_source(original).strip()}")
    print()
    for op_id, op in original.ops.items():
        print(f"    {ascii_op(original, op)}")

    # --- Point Mutation ---
    section("点变异 (point mutation)")
    print("""
点变异直接修改 IR 图中某个操作的属性。
例如把一个 binary 操作的 operator 从 "Add" 改为 "Sub"。
""")
    mutated = mutate_ir(original, rng, mutation_type="point")
    print(f"  变异后: {emit_python_source(mutated).strip()}")
    print()
    for op_id, op in mutated.ops.items():
        if op.opcode == "binary":
            orig_op = original.ops.get(op_id)
            if orig_op and orig_op.attrs.get("operator") != op.attrs.get("operator"):
                print(f"    ⚡ {op_id}: operator 从 '{orig_op.attrs.get('operator')}'"
                      f" 变为 '{op.attrs.get('operator')}'")

    # --- Constant Perturbation ---
    section("常量扰动 (constant_perturb)")
    print("""
找到 IR 中的一个常量操作 (const)，对其数值加高斯噪声。
""")
    perturbed = mutate_ir(original, rng, mutation_type="constant_perturb")
    print(f"  扰动后: {emit_python_source(perturbed).strip()}")
    print()
    for op_id, op in perturbed.ops.items():
        if op.opcode == "const":
            orig_op = original.ops.get(op_id)
            if orig_op:
                old_val = orig_op.attrs.get("literal")
                new_val = op.attrs.get("literal")
                if old_val != new_val:
                    print(f"    ⚡ {op_id}: 常量从 {old_val} 变为 {new_val}")

    print()
    print("关键优势: 变异直接操作 IR 图结构，比修改文本字符串更安全、更精确。")


def demo_full_pipeline():
    """Show the complete end-to-end pipeline."""

    banner("演示 E: 完整端到端流水线")
    print("""
    ┌──────────────┐   compile_source_   ┌──────────┐   emit_python_  ┌──────────┐
    │ Python 源码  │ ─────to_ir()──────→ │ Function │ ──source()────→ │ Python   │
    │ def f(a,b):  │                     │    IR    │                 │ 源码     │
    │   return a+b │                     │ (SSA图)  │   emit_cpp_    ┌──────────┐
    └──────────────┘                     └──────────┘ ──ops()──────→ │ C++ Ops  │
                                              │                      │ [1,0,...] │
                                              │                      └──────────┘
                                         mutate_ir()
                                         crossover_ir()
                                              │
                                              ▼
                                        ┌──────────┐
                                        │ 变异后的 │
                                        │    IR    │
                                        └──────────┘
    """)

    source = "def f(a, b): return a * b + a"

    section("1. 编译")
    ir = compile_source_to_ir(source, "f")
    print(f"  {source}")
    print(f"  → FunctionIR: {len(ir.ops)} ops, {len(ir.values)} values, {len(ir.blocks)} blocks")

    section("2. IR 内部结构")
    print(indent_block(ascii_cfg(ir), "  "))

    section("3. 表达式树")
    for op in ir.ops.values():
        if op.opcode == "return" and op.inputs:
            tree_lines = _build_tree(ir, op.inputs[0], "", True)
            for line in tree_lines:
                print(f"  {line}")

    section("4. 回生成 Python")
    print(f"  {emit_python_source(ir).strip()}")

    section("5. 生成 C++ 操作码")
    cpp = emit_cpp_ops(ir)
    print(f"  操作码: {cpp}")
    print()
    print(ascii_cpp_ops(cpp))

    section("6. 变异")
    from evolution.operators import mutate_ir
    import numpy as np
    rng = np.random.default_rng(7)
    mutated = mutate_ir(ir, rng, mutation_type="point")
    print(f"  原始:   {emit_python_source(ir).strip()}")
    print(f"  变异后: {emit_python_source(mutated).strip()}")

    mutated_cpp = emit_cpp_ops(mutated)
    print()
    print("  变异后 C++ 操作码:")
    print(indent_block(ascii_cpp_ops(mutated_cpp), "  "))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔" + "═" * (_WIDTH - 2) + "╗")
    print("║" + " Algorithm-IR 交互式教程 ".center(_WIDTH - 2) + "║")
    print("║" + " Python → IR → Python/C++ 完整流程 ".center(_WIDTH - 2) + "║")
    print("╚" + "═" * (_WIDTH - 2) + "╝")

    # Demo A: Simple function compilation
    demo_simple_compilation()

    # Demo B: Branching
    demo_branch_compilation()

    # Demo C: Safe math functions
    demo_safe_functions()

    # Demo D: IR mutation
    demo_mutation_flow()

    # Demo E: Full pipeline
    demo_full_pipeline()

    # Demo F: Skeleton flow
    demo_skeleton_flow()

    print()
    banner("教程完成!")
    print("""
你已经看到了 Algorithm-IR 的所有核心层：

  第 0 层: Python 源代码          — 人类可读
  第 1 层: 抽象语法树 (AST)       — Python 解析器生成
  第 2 层: FunctionIR (SSA 图)    — 我们的核心 IR，可以变异/交叉
  第 3 层: xDSL 原生表示          — 底层类型安全的操作定义
  第 4a层: 回生成 Python          — 从 IR 重建可执行 Python
  第 4b层: C++ 操作码             — 从 IR 生成高性能 C++ 求值指令

进化引擎在第 2 层 (FunctionIR) 上操作：
  - random_ir_program(): 在第 0 层生成随机源码，编译到第 2 层
  - mutate_ir(): 直接修改第 2 层的 Op 属性
  - crossover_ir(): 在第 2 层交换结构
  - to_callable(): 通过第 4a 层执行
  - to_cpp_ops(): 通过第 4b 层执行（高性能）

更多信息请参阅 README.md。
""")
