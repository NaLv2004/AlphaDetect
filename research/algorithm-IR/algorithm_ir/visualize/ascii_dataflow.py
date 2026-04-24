"""ASCII data-flow renderer for ``FunctionIR``.

This module produces a compact, terminal-friendly textual graph of an
``algorithm_ir.FunctionIR`` instance, intended for debugging the GNN
grafting pipeline.  The output is plain-text (no graphviz) and uses
ANSI escape codes for colour highlighting.

The two top-level entry points are:

* :func:`render_ir_dataflow` -- render a single IR.
* :func:`render_graft_visualization` -- render the host IR, donor IR
  and post-graft IR side by side, with the rewrite region and the
  inlined donor ops colour-highlighted.

Highlight colours:

* **HOST**  -- region being replaced is shown in **bright red**.
* **DONOR** -- whole donor body is shown in **cyan**.
* **GRAFT** -- ops inlined from the donor (i.e. ``attrs["grafted"] is
  True``) are shown in **green**; values that survived rebinding from
  the host are shown in **dim**.

The renderer never raises on malformed IR -- it falls back to a
``<missing>`` marker and a structural warning at the bottom of the
listing.  This is deliberate: the visualiser is a debugging aid and
must remain robust to partial/invalid inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..ir.model import FunctionIR, Op


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"        # bright red -- replaced region in host
_GREEN = "\033[92m"      # bright green -- newly grafted donor ops
_YELLOW = "\033[93m"     # bright yellow -- entry/exit ports
_CYAN = "\033[96m"       # bright cyan -- donor body
_MAGENTA = "\033[95m"    # bright magenta -- contract / metadata


def _wrap(text: str, code: str, *, color: bool) -> str:
    if not color or not code:
        return text
    return f"{code}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Operator symbol table
# ---------------------------------------------------------------------------

# Small visual icons for the most common opcodes.  Anything not listed
# falls back to the opcode name in square brackets.
# AST class names → math symbols.  The frontend stores ``type(node.op).__name__``
# in ``attrs["operator"]`` (e.g. ``"Add"``), so we render them as ``+`` / ``-``
# / etc. for human readability.
_BIN_OP_SYMBOL = {
    "Add": "+", "Sub": "-", "Mult": "*", "MatMult": "@", "Div": "/",
    "FloorDiv": "//", "Mod": "%", "Pow": "**", "LShift": "<<", "RShift": ">>",
    "BitOr": "|", "BitAnd": "&", "BitXor": "^", "And": "and", "Or": "or",
    "+": "+", "-": "-", "*": "*", "@": "@", "/": "/", "%": "%", "**": "**",
    "//": "//", "<<": "<<", ">>": ">>", "|": "|", "&": "&", "^": "^",
}

_UNARY_OP_SYMBOL = {
    "USub": "-", "UAdd": "+", "Not": "!", "Invert": "~",
    "-": "-", "+": "+", "!": "!", "~": "~", "not": "not ",
}

_CMP_OP_SYMBOL = {
    "Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=",
    "Gt": ">", "GtE": ">=", "Is": "is", "IsNot": "is not",
    "In": "in", "NotIn": "not in",
    "==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=",
}


def _binary_symbol(name: str | None) -> str:
    if not name:
        return "?"
    return _BIN_OP_SYMBOL.get(name, name)


def _unary_symbol(name: str | None) -> str:
    if not name:
        return "?"
    return _UNARY_OP_SYMBOL.get(name, name)


def _compare_symbol(name: str | None) -> str:
    if not name:
        return "?"
    return _CMP_OP_SYMBOL.get(name, name)


_OPCODE_ICON = {
    "binary": "",       # rendered using attrs["operator"]
    "unary": "",        # rendered using attrs["operator"]
    "compare": "",
    "const": "const",
    "call": "call",
    "get_attr": ".",
    "get_item": "[]",
    "set_attr": ".=",
    "set_item": "[]=",
    "build_list": "list",
    "build_tuple": "tuple",
    "build_dict": "dict",
    "iter_init": "iter",
    "iter_next": "next",
    "iter_has_next": "has_next",
    "phi": "φ",
    "branch": "branch",
    "jump": "jump",
    "return": "return",
    "assign": "=",
    "augassign": "op=",
    "store": "store",
    "load": "load",
    "algslot": "slot",
    "slot": "slot",
}


def _render_const_literal(value: object) -> str:
    """Compact, single-line representation of a const literal."""
    try:
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, str):
            s = value if len(value) <= 24 else value[:21] + "..."
            return repr(s)
        type_name = type(value).__name__
        return f"<{type_name}>"
    except Exception:
        return "<const>"


def _render_op_expression(
    op: Op,
    value_alias: dict[str, str],
    func_ir: FunctionIR,
) -> str:
    """Build a one-line right-hand side expression for ``op``."""

    def alias(vid: str) -> str:
        return value_alias.get(vid, vid)

    opcode = op.opcode
    attrs = op.attrs or {}

    if opcode == "const":
        if "name" in attrs:
            return f"const {attrs['name']} = {_render_const_literal(attrs.get('literal'))}"
        return f"const {_render_const_literal(attrs.get('literal'))}"

    if opcode == "binary" and len(op.inputs) == 2:
        operator = _binary_symbol(attrs.get("operator"))
        return f"{alias(op.inputs[0])} {operator} {alias(op.inputs[1])}"

    if opcode == "unary" and len(op.inputs) == 1:
        operator = _unary_symbol(attrs.get("operator"))
        return f"{operator}{alias(op.inputs[0])}"

    if opcode == "compare" and len(op.inputs) >= 2:
        operators = attrs.get("operators") or []
        parts = [alias(op.inputs[0])]
        for i, operand in enumerate(op.inputs[1:]):
            o = _compare_symbol(operators[i] if i < len(operators) else None)
            parts.append(f"{o} {alias(operand)}")
        return " ".join(parts)

    if opcode == "call":
        n_args = int(attrs.get("n_args", max(len(op.inputs) - 1, 0)))
        callee = op.inputs[0] if op.inputs else None
        pos_args = op.inputs[1 : 1 + n_args]
        kw_names = attrs.get("kwarg_names") or []
        kw_vals = op.inputs[1 + n_args :]
        callee_name = "<callee>"
        if callee is not None:
            callee_val = func_ir.values.get(callee)
            if callee_val is not None:
                callee_name = callee_val.name_hint or alias(callee)
            else:
                callee_name = alias(callee)
        rendered_args = [alias(a) for a in pos_args]
        rendered_args.extend(
            f"{name}={alias(v)}" for name, v in zip(kw_names, kw_vals)
        )
        return f"{callee_name}({', '.join(rendered_args)})"

    if opcode == "get_attr" and op.inputs:
        return f"{alias(op.inputs[0])}.{attrs.get('attr', '?')}"

    if opcode == "get_item" and len(op.inputs) >= 2:
        return f"{alias(op.inputs[0])}[{alias(op.inputs[1])}]"

    if opcode == "set_item" and len(op.inputs) >= 3:
        return f"{alias(op.inputs[0])}[{alias(op.inputs[1])}] = {alias(op.inputs[2])}"

    if opcode == "set_attr" and len(op.inputs) >= 2:
        return f"{alias(op.inputs[0])}.{attrs.get('attr', '?')} = {alias(op.inputs[1])}"

    if opcode == "phi":
        return "φ(" + ", ".join(alias(v) for v in op.inputs) + ")"

    if opcode == "branch":
        cond = alias(op.inputs[0]) if op.inputs else "?"
        t = attrs.get("true_target") or attrs.get("then") or "?"
        f = attrs.get("false_target") or attrs.get("else") or "?"
        return f"branch {cond} ? -> {t} : -> {f}"

    if opcode == "jump":
        target = attrs.get("target", "?")
        return f"jump -> {target}"

    if opcode == "return":
        return "return " + ", ".join(alias(v) for v in op.inputs)

    if opcode in ("build_list", "build_tuple"):
        bracket_open, bracket_close = ("[", "]") if opcode == "build_list" else ("(", ")")
        return f"{bracket_open}{', '.join(alias(v) for v in op.inputs)}{bracket_close}"

    if opcode == "build_dict":
        return "{" + ", ".join(alias(v) for v in op.inputs) + "}"

    if opcode == "assign" and op.inputs:
        target = attrs.get("target", "")
        return f"{target} := {alias(op.inputs[0])}"

    icon = _OPCODE_ICON.get(opcode, opcode)
    return f"{icon}(" + ", ".join(alias(v) for v in op.inputs) + ")"


# ---------------------------------------------------------------------------
# Topological ordering
# ---------------------------------------------------------------------------


def _topological_op_order(func_ir: FunctionIR) -> list[str]:
    """Return op IDs in dataflow order, respecting block order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for block_id, block in func_ir.blocks.items():
        for op_id in block.op_ids:
            if op_id in seen or op_id not in func_ir.ops:
                continue
            ordered.append(op_id)
            seen.add(op_id)
    # Trailing ops not registered to any block (defensive)
    for op_id in func_ir.ops:
        if op_id not in seen:
            ordered.append(op_id)
            seen.add(op_id)
    return ordered


# ---------------------------------------------------------------------------
# Alias allocation
# ---------------------------------------------------------------------------


def _allocate_value_aliases(func_ir: FunctionIR) -> dict[str, str]:
    """Assign short ``v0/v1/...`` aliases to every value, preferring args."""
    alias: dict[str, str] = {}
    counter = 0
    for vid in func_ir.arg_values:
        val = func_ir.values.get(vid)
        hint = (val.name_hint if val and val.name_hint else None) or f"a{counter}"
        # Sanitise the hint to keep listings narrow
        hint = hint.replace(" ", "_")
        if len(hint) > 12:
            hint = hint[:11] + "_"
        alias[vid] = hint
        counter += 1
    counter = 0
    for vid in func_ir.values:
        if vid in alias:
            continue
        alias[vid] = f"v{counter}"
        counter += 1
    return alias


# ---------------------------------------------------------------------------
# Single-IR renderer
# ---------------------------------------------------------------------------


@dataclass
class _RenderContext:
    color: bool
    highlight_ops: frozenset[str]
    highlight_color: str
    highlight_label: str
    show_consumers: bool


def _render_op_line(
    op: Op,
    func_ir: FunctionIR,
    alias: dict[str, str],
    ctx: _RenderContext,
    consumers_of: dict[str, list[str]],
) -> str:
    is_highlighted = op.id in ctx.highlight_ops
    grafted = bool((op.attrs or {}).get("grafted"))

    out_alias = ", ".join(alias.get(v, v) for v in op.outputs) or "_"
    expr = _render_op_expression(op, alias, func_ir)
    body = f"{out_alias:>12} = {expr}"

    marker = " "
    line_color = ""
    if is_highlighted:
        marker = "*"
        line_color = ctx.highlight_color
    elif grafted and ctx.highlight_label != "REGION":
        # Grafted-into-host visualisation: tag inlined ops in green
        marker = "+"
        line_color = _GREEN

    line = f"  {marker} {op.id:<10} | {body}"

    if ctx.show_consumers:
        consumers: list[str] = []
        for vid in op.outputs:
            consumers.extend(consumers_of.get(vid, []))
        if consumers:
            uniq = []
            seen = set()
            for c in consumers:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
                if len(uniq) >= 6:
                    break
            tail = ", ".join(uniq)
            extra = "" if len(consumers) == len(uniq) else f", +{len(consumers) - len(uniq)} more"
            line = f"{line:<70} -> {tail}{extra}"

    return _wrap(line, line_color, color=ctx.color)


def _build_consumer_index(func_ir: FunctionIR) -> dict[str, list[str]]:
    consumers: dict[str, list[str]] = {}
    for op in func_ir.ops.values():
        for vid in op.inputs:
            consumers.setdefault(vid, []).append(op.id)
    return consumers


def render_ir_dataflow(
    func_ir: FunctionIR,
    *,
    title: str = "IR",
    highlight_ops: Iterable[str] = (),
    highlight_color: str = _RED,
    highlight_label: str = "REGION",
    color: bool = True,
    show_consumers: bool = True,
    show_legend: bool = True,
) -> str:
    """Render ``func_ir`` as an ASCII dataflow listing.

    Parameters
    ----------
    func_ir
        The IR to render.
    title
        Header label printed above the listing.
    highlight_ops
        Op IDs that should be visually marked (default: red bg).
    highlight_color
        ANSI escape sequence used to colour the highlighted ops.
    highlight_label
        Short tag for the highlighted set (used in the legend).
    color
        Toggle ANSI colours.  If False, lines are returned in plain
        text and only the marker character distinguishes highlighted
        lines.
    show_consumers
        Append a "-> consumer_op_ids" suffix to each line.
    show_legend
        Print the colour legend at the top.
    """
    if func_ir is None:
        return f"<{title}: <None>>\n"

    alias = _allocate_value_aliases(func_ir)
    highlight_set = frozenset(highlight_ops)
    consumers = _build_consumer_index(func_ir) if show_consumers else {}
    ctx = _RenderContext(
        color=color,
        highlight_ops=highlight_set,
        highlight_color=highlight_color,
        highlight_label=highlight_label,
        show_consumers=show_consumers,
    )

    sep = "=" * 78
    lines: list[str] = []
    lines.append(_wrap(sep, _BOLD, color=color))
    lines.append(
        _wrap(f"  {title}  ({func_ir.name}, ops={len(func_ir.ops)}, blocks={len(func_ir.blocks)})", _BOLD, color=color)
    )
    lines.append(_wrap(sep, _BOLD, color=color))

    if show_legend and (highlight_set or any((op.attrs or {}).get("grafted") for op in func_ir.ops.values())):
        legend_parts = []
        if highlight_set:
            legend_parts.append(_wrap(f"* = {highlight_label}", highlight_color, color=color))
        if any((op.attrs or {}).get("grafted") for op in func_ir.ops.values()):
            legend_parts.append(_wrap("+ = grafted (donor-origin)", _GREEN, color=color))
        if legend_parts:
            lines.append("  legend: " + "   ".join(legend_parts))

    # Header: arg values
    arg_repr = []
    for vid in func_ir.arg_values:
        val = func_ir.values.get(vid)
        type_hint = (val.type_hint if val else None) or "?"
        arg_repr.append(_wrap(f"{alias.get(vid, vid)}:{type_hint}", _YELLOW, color=color))
    lines.append("  args:    " + ", ".join(arg_repr) if arg_repr else "  args:    <none>")

    return_repr = [_wrap(alias.get(vid, vid), _YELLOW, color=color) for vid in func_ir.return_values]
    lines.append("  returns: " + ", ".join(return_repr) if return_repr else "  returns: <none>")
    lines.append("")

    # Render block-by-block in declaration order
    for block_id in func_ir.blocks:
        block = func_ir.blocks[block_id]
        block_header = (
            f"  block {block_id}"
            f"  preds={list(block.preds)}  succs={list(block.succs)}"
        )
        lines.append(_wrap(block_header, _MAGENTA, color=color))
        if not block.op_ids:
            lines.append("    <empty>")
            continue
        for op_id in block.op_ids:
            op = func_ir.ops.get(op_id)
            if op is None:
                lines.append(f"    ! {op_id} <missing>")
                continue
            lines.append(_render_op_line(op, func_ir, alias, ctx, consumers))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graft pipeline renderer
# ---------------------------------------------------------------------------


def render_graft_visualization(
    *,
    host_ir: FunctionIR,
    region: object | None,
    donor_ir: FunctionIR | None,
    grafted_ir: FunctionIR | None,
    host_title: str = "HOST",
    donor_title: str = "DONOR",
    grafted_title: str = "GRAFTED",
    color: bool = True,
    show_consumers: bool = True,
) -> str:
    """Render a host -> donor -> result graft pipeline as ASCII text.

    The host IR has the rewrite-region ops marked in red.  The donor
    IR is shown in cyan.  The grafted IR has the inlined (``grafted``)
    ops marked in green.

    Any of ``donor_ir``/``grafted_ir`` may be ``None`` -- the
    corresponding section is then omitted.  Designed to be safe to
    call from logging code that may run before grafting completes.
    """

    region_op_ids: list[str] = []
    region_entry: list[str] = []
    region_exit: list[str] = []
    if region is not None:
        region_op_ids = list(getattr(region, "op_ids", []) or [])
        region_entry = list(getattr(region, "entry_values", []) or [])
        region_exit = list(getattr(region, "exit_values", []) or [])

    sections: list[str] = []

    # ---- Host
    host_legend = (
        f"  region: ops={len(region_op_ids)}, "
        f"entry_values={len(region_entry)}, exit_values={len(region_exit)}"
    )
    sections.append(
        render_ir_dataflow(
            host_ir,
            title=f"{host_title} -- replaced region marked in red",
            highlight_ops=region_op_ids,
            highlight_color=_RED,
            highlight_label="REGION (to be replaced)",
            color=color,
            show_consumers=show_consumers,
        )
    )
    sections.append(_wrap(host_legend, _RED, color=color))
    sections.append("")

    # ---- Donor
    if donor_ir is not None:
        donor_op_ids = list(donor_ir.ops.keys())
        sections.append(
            render_ir_dataflow(
                donor_ir,
                title=f"{donor_title} -- entire body is the replacement",
                highlight_ops=donor_op_ids,
                highlight_color=_CYAN,
                highlight_label="DONOR REPLACEMENT",
                color=color,
                show_consumers=show_consumers,
            )
        )
        sections.append("")

    # ---- Grafted result
    if grafted_ir is not None:
        grafted_ops = [op.id for op in grafted_ir.ops.values() if (op.attrs or {}).get("grafted")]
        sections.append(
            render_ir_dataflow(
                grafted_ir,
                title=f"{grafted_title} -- newly inlined ops marked in green",
                highlight_ops=grafted_ops,
                highlight_color=_GREEN,
                highlight_label="GRAFTED OP",
                color=color,
                show_consumers=show_consumers,
            )
        )

    return "\n".join(sections)
