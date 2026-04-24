"""Fully-Inlined IR (FII) view of an AlgorithmGenome with per-op provenance.

Per code_review.md §2:

After AST inlining each ``_slot_*`` helper body, this module brackets the
inlined statements with sentinel marker calls:

    __fii_provmark_begin(<call_site_id>)
    ... inlined body ...
    __fii_provmark_end(<call_site_id>)

These markers compile down to ``call`` ops whose first input resolves to
a ``const`` op with ``attrs["name"]`` == ``"__fii_provmark_begin"`` /
``"__fii_provmark_end"``.  After IR construction we walk every block in
declaration order, maintain a stack of active slot frames (a frame is
keyed by ``call_site_id``), and write a ``_provenance`` dict to each
op's ``attrs`` describing:

    - ``from_slot_id``    : slot helper name (e.g. ``_slot_jacobi_step``)
                            or ``None`` for structural (host) ops
    - ``slot_pop_key``    : same as ``from_slot_id`` (alias used by the
                            dispatcher)
    - ``variant_idx``     : the slot population's ``best_idx`` at time of
                            inlining
    - ``orig_op_id``      : ``None`` (re-compiled IR loses original op
                            identity; placeholder reserved per spec)
    - ``is_slot_boundary``: ``True`` for the marker ops themselves
    - ``boundary_kind``   : ``"enter"`` / ``"exit"`` / ``None``
    - ``call_site_id``    : unique per inlined call site (so back-mapping
                            can identify exactly which call was inlined)

A side ``provenance_map: dict[op_id, dict]`` is also stored on
``ir.attrs["_provenance_map"]`` for callers that need bulk lookup, and
side-table ``ir.attrs["_provenance_call_sites"]`` records the per-call
metadata (slot helper name, variant_idx) used by the graft dispatcher
when back-mapping Case I grafts.
"""
from __future__ import annotations

import ast
import logging
from copy import deepcopy
from typing import Any

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.frontend.ir_builder import compile_source_to_ir

from evolution.materialize import materialize
from evolution.ir_pool import _template_globals

logger = logging.getLogger(__name__)

__all__ = [
    "build_fii_ir",
    "build_fii_ir_with_provenance",
    "inline_all_helpers_source",
    "clear_fii_cache",
    "get_op_provenance",
    "MARKER_BEGIN",
    "MARKER_END",
]


MARKER_BEGIN = "__fii_provmark_begin"
MARKER_END = "__fii_provmark_end"


def _provmark_begin(*_args, **_kwargs):
    return None


def _provmark_end(*_args, **_kwargs):
    return None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_FII_CACHE: dict[tuple, tuple[FunctionIR, dict[str, dict[str, Any]], dict[int, dict[str, Any]]]] = {}
_FII_CACHE_MAX = 512


def _cache_key(genome) -> tuple:
    bests = tuple(
        (k, p.best_idx) for k, p in sorted(genome.slot_populations.items())
    )
    return (genome.algo_id, id(genome.structural_ir), bests)


def _evict_if_needed() -> None:
    if len(_FII_CACHE) >= _FII_CACHE_MAX:
        for k in list(_FII_CACHE.keys())[: _FII_CACHE_MAX // 4]:
            del _FII_CACHE[k]


def build_fii_ir_with_provenance(
    genome,
):
    """Build the fully-inlined IR with a per-op provenance map.

    Returns ``(ir, provenance_map, call_site_table)`` on success or
    ``(None, None, None)`` if inlining or recompilation fails.
    """
    key = _cache_key(genome)
    cached = _FII_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        source = materialize(genome)
        main_name = genome.metadata.get("detector_name") or genome.algo_id
        raw_id = genome.algo_id or "xx"
        prefix = "".join(c for c in raw_id if c.isalnum())[:6].lower() or "gx"

        slot_variant_idx: dict[str, int] = {}
        for slot_id, pop in genome.slot_populations.items():
            slot_variant_idx[slot_id] = pop.best_idx

        inlined_source, call_site_table = inline_all_helpers_source(
            source,
            main_name,
            prefix=prefix,
            slot_variant_idx=slot_variant_idx,
            emit_markers=True,
            return_call_site_table=True,
        )
        g = _template_globals()
        try:
            from evolution.pool_ops_l2 import TreeNode
            g["TreeNode"] = TreeNode
        except Exception:
            pass
        g[MARKER_BEGIN] = _provmark_begin
        g[MARKER_END] = _provmark_end

        ir = compile_source_to_ir(inlined_source, main_name, g)

        provenance_map = _annotate_provenance(ir, call_site_table)
    except Exception as exc:
        logger.debug("FII build failed for %s: %r", getattr(genome, "algo_id", "?"), exc)
        return None, None, None

    try:
        ir.attrs["_provenance_map"] = provenance_map  # type: ignore[attr-defined]
        ir.attrs["_provenance_call_sites"] = call_site_table  # type: ignore[attr-defined]
    except Exception:
        pass

    _evict_if_needed()
    _FII_CACHE[key] = (ir, provenance_map, call_site_table)
    return ir, provenance_map, call_site_table


def build_fii_ir(genome) -> FunctionIR | None:
    """Backward-compatible wrapper returning only the IR."""
    ir, _pm, _cs = build_fii_ir_with_provenance(genome)
    return ir


def clear_fii_cache() -> None:
    _FII_CACHE.clear()


def get_op_provenance(ir: FunctionIR, op_id: str) -> dict[str, Any]:
    """Return the provenance dict for an op (empty dict if none)."""
    op = ir.ops.get(op_id)
    if op is None:
        return {}
    prov = op.attrs.get("_provenance")
    if isinstance(prov, dict):
        return prov
    return {}


# ---------------------------------------------------------------------------
# Provenance walker (post-IR-construction)
# ---------------------------------------------------------------------------


def _const_name_for_call(op, ir: FunctionIR) -> str | None:
    if op.opcode != "call" or not op.inputs:
        return None
    func_value_id = op.inputs[0]
    val = ir.values.get(func_value_id)
    def_op = None
    if val is not None and val.def_op is not None:
        def_op = ir.ops.get(val.def_op)
    if def_op is None:
        for cand in ir.ops.values():
            if func_value_id in cand.outputs:
                def_op = cand
                break
    if def_op is None or def_op.opcode != "const":
        return None
    return def_op.attrs.get("name") or def_op.attrs.get("module_name")


def _parse_marker_call_site_id(op, ir: FunctionIR) -> int | None:
    if op.opcode != "call" or len(op.inputs) < 2:
        return None
    arg_value_id = op.inputs[1]
    val = ir.values.get(arg_value_id)
    def_op = None
    if val is not None and val.def_op is not None:
        def_op = ir.ops.get(val.def_op)
    if def_op is None:
        for cand in ir.ops.values():
            if arg_value_id in cand.outputs:
                def_op = cand
                break
    if def_op is None or def_op.opcode != "const":
        return None
    lit = def_op.attrs.get("literal")
    if isinstance(lit, int):
        return lit
    return None


def _annotate_provenance(
    ir: FunctionIR,
    call_site_table: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    provenance_map: dict[str, dict[str, Any]] = {}

    for block in ir.blocks.values():
        frame_stack: list[int] = []
        for op_id in block.op_ids:
            op = ir.ops.get(op_id)
            if op is None:
                continue

            marker_name = None
            marker_site_id = None
            if op.opcode == "call":
                cname = _const_name_for_call(op, ir)
                if cname == MARKER_BEGIN or cname == MARKER_END:
                    marker_name = cname
                    marker_site_id = _parse_marker_call_site_id(op, ir)

            if marker_name == MARKER_BEGIN and marker_site_id is not None:
                meta = call_site_table.get(marker_site_id, {})
                prov = {
                    "from_slot_id": meta.get("slot_id"),
                    "slot_pop_key": meta.get("slot_id"),
                    "variant_idx": meta.get("variant_idx", 0),
                    "orig_op_id": None,
                    "is_slot_boundary": True,
                    "boundary_kind": "enter",
                    "call_site_id": marker_site_id,
                }
                op.attrs["_provenance"] = prov
                provenance_map[op_id] = prov
                frame_stack.append(marker_site_id)
                continue

            if marker_name == MARKER_END and marker_site_id is not None:
                meta = call_site_table.get(marker_site_id, {})
                prov = {
                    "from_slot_id": meta.get("slot_id"),
                    "slot_pop_key": meta.get("slot_id"),
                    "variant_idx": meta.get("variant_idx", 0),
                    "orig_op_id": None,
                    "is_slot_boundary": True,
                    "boundary_kind": "exit",
                    "call_site_id": marker_site_id,
                }
                op.attrs["_provenance"] = prov
                provenance_map[op_id] = prov
                if frame_stack and frame_stack[-1] == marker_site_id:
                    frame_stack.pop()
                else:
                    logger.debug(
                        "FII provenance: unbalanced END marker site=%d op=%s",
                        marker_site_id, op_id,
                    )
                continue

            if frame_stack:
                site_id = frame_stack[-1]
                meta = call_site_table.get(site_id, {})
                prov = {
                    "from_slot_id": meta.get("slot_id"),
                    "slot_pop_key": meta.get("slot_id"),
                    "variant_idx": meta.get("variant_idx", 0),
                    "orig_op_id": None,
                    "is_slot_boundary": False,
                    "boundary_kind": None,
                    "call_site_id": site_id,
                }
            else:
                prov = {
                    "from_slot_id": None,
                    "slot_pop_key": None,
                    "variant_idx": 0,
                    "orig_op_id": None,
                    "is_slot_boundary": False,
                    "boundary_kind": None,
                    "call_site_id": None,
                }
            op.attrs["_provenance"] = prov
            provenance_map[op_id] = prov

    return provenance_map


# ---------------------------------------------------------------------------
# AST-level source inlining
# ---------------------------------------------------------------------------

class _FreshName:
    def __init__(self, prefix: str = "") -> None:
        self.counter = 0
        self.prefix = prefix

    def __call__(self, base: str) -> str:
        self.counter += 1
        if self.prefix:
            return f"__fii_{self.prefix}_{base}_{self.counter}"
        return f"__fii_{base}_{self.counter}"


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.mapping:
            return ast.copy_location(
                ast.Name(id=self.mapping[node.id], ctx=node.ctx), node,
            )
        return node


class _ReturnToAssign(ast.NodeTransformer):
    def __init__(self, lhs_name: str) -> None:
        self.lhs_name = lhs_name

    def visit_Return(self, node: ast.Return) -> Any:
        if node.value is None:
            return node
        new = ast.Assign(
            targets=[ast.Name(id=self.lhs_name, ctx=ast.Store())],
            value=node.value,
        )
        return ast.copy_location(new, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        # Do NOT recurse into nested defs
        return node


def _collect_local_names(func_def: ast.FunctionDef) -> set[str]:
    """Return the set of names locally bound inside the helper body
    (parameters + assignment targets + for-loop targets)."""
    names: set[str] = set()
    for arg in func_def.args.args:
        names.add(arg.arg)
    for node in ast.walk(func_def):
        if isinstance(node, ast.FunctionDef) and node is not func_def:
            continue
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, ast.Tuple):
                    for elt in t.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def _make_marker_stmt(name: str, call_site_id: int) -> ast.Expr:
    """Produce an AST stmt: ``__fii_provmark_*(<call_site_id>)`` (no LHS)."""
    call = ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()),
        args=[ast.Constant(value=call_site_id)],
        keywords=[],
    )
    return ast.Expr(value=call)


def _inline_one_call(
    helper_def: ast.FunctionDef,
    call: ast.Call,
    lhs_name: str,
    fresh: _FreshName,
    *,
    call_site_id: int = 0,
    emit_markers: bool = False,
    helpers: dict[str, ast.FunctionDef] | None = None,
    call_site_table: dict[int, dict[str, Any]] | None = None,
    site_counter: list[int] | None = None,
    slot_variant_idx: dict[str, int] | None = None,
) -> list[ast.stmt]:
    """Produce a list of statements implementing ``lhs_name = helper(args)``.

    When ``emit_markers`` is True the body is bracketed with sentinel
    ``__fii_provmark_begin / _end`` calls carrying ``call_site_id``.
    Recursively inlines nested ``_slot_*`` calls inside the helper body.
    """
    if helpers is None:
        helpers = {}
    if call_site_table is None:
        call_site_table = {}
    if site_counter is None:
        site_counter = [0]
    if slot_variant_idx is None:
        slot_variant_idx = {}

    local_names = _collect_local_names(helper_def)
    mapping: dict[str, str] = {name: fresh(name) for name in local_names}

    stmts: list[ast.stmt] = []

    if emit_markers:
        stmts.append(_make_marker_stmt(MARKER_BEGIN, call_site_id))

    # 1. Parameter bindings: param_fresh = arg_expr
    for param, arg_expr in zip(helper_def.args.args, call.args):
        stmts.append(
            ast.Assign(
                targets=[ast.Name(id=mapping[param.arg], ctx=ast.Store())],
                value=deepcopy(arg_expr),
            )
        )

    # 2. Helper body: alpha-rename locals + rewrite Return to lhs assignment
    renamer = _Renamer(mapping)
    ret_rewriter = _ReturnToAssign(lhs_name)
    body_stmts: list[ast.stmt] = []
    for body_stmt in helper_def.body:
        new_stmt = deepcopy(body_stmt)
        new_stmt = renamer.visit(new_stmt)
        new_stmt = ret_rewriter.visit(new_stmt)
        ast.fix_missing_locations(new_stmt)
        body_stmts.append(new_stmt)

    # Recursively inline nested ``_slot_*`` calls inside the body.
    body_stmts = _inline_stmt_list(
        body_stmts,
        helpers,
        fresh,
        emit_markers=emit_markers,
        call_site_table=call_site_table,
        site_counter=site_counter,
        slot_variant_idx=slot_variant_idx,
    )
    stmts.extend(body_stmts)

    if emit_markers:
        stmts.append(_make_marker_stmt(MARKER_END, call_site_id))

    return stmts


def _inline_stmt_list(
    stmts: list[ast.stmt],
    helpers: dict[str, ast.FunctionDef],
    fresh: _FreshName,
    *,
    emit_markers: bool = False,
    call_site_table: dict[int, dict[str, Any]] | None = None,
    site_counter: list[int] | None = None,
    slot_variant_idx: dict[str, int] | None = None,
) -> list[ast.stmt]:
    if call_site_table is None:
        call_site_table = {}
    if site_counter is None:
        site_counter = [0]
    if slot_variant_idx is None:
        slot_variant_idx = {}

    out: list[ast.stmt] = []
    for stmt in stmts:
        # Simple-call form: x = helper(args)
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id in helpers
        ):
            helper_name = stmt.value.func.id
            helper = helpers[helper_name]
            site_counter[0] += 1
            call_site_id = site_counter[0]
            v_idx = slot_variant_idx.get(helper_name, 0)
            call_site_table[call_site_id] = {
                "slot_id": helper_name,
                "variant_idx": v_idx,
            }
            out.extend(_inline_one_call(
                helper, stmt.value, stmt.targets[0].id, fresh,
                call_site_id=call_site_id,
                emit_markers=emit_markers,
                helpers=helpers,
                call_site_table=call_site_table,
                site_counter=site_counter,
                slot_variant_idx=slot_variant_idx,
            ))
            continue

        # Recurse into containers
        if isinstance(stmt, ast.While):
            stmt.body = _inline_stmt_list(
                stmt.body, helpers, fresh,
                emit_markers=emit_markers,
                call_site_table=call_site_table,
                site_counter=site_counter,
                slot_variant_idx=slot_variant_idx,
            )
            stmt.orelse = _inline_stmt_list(
                stmt.orelse, helpers, fresh,
                emit_markers=emit_markers,
                call_site_table=call_site_table,
                site_counter=site_counter,
                slot_variant_idx=slot_variant_idx,
            )
        elif isinstance(stmt, ast.If):
            stmt.body = _inline_stmt_list(
                stmt.body, helpers, fresh,
                emit_markers=emit_markers,
                call_site_table=call_site_table,
                site_counter=site_counter,
                slot_variant_idx=slot_variant_idx,
            )
            stmt.orelse = _inline_stmt_list(
                stmt.orelse, helpers, fresh,
                emit_markers=emit_markers,
                call_site_table=call_site_table,
                site_counter=site_counter,
                slot_variant_idx=slot_variant_idx,
            )
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            stmt.body = _inline_stmt_list(
                stmt.body, helpers, fresh,
                emit_markers=emit_markers,
                call_site_table=call_site_table,
                site_counter=site_counter,
                slot_variant_idx=slot_variant_idx,
            )
            stmt.orelse = _inline_stmt_list(
                stmt.orelse, helpers, fresh,
                emit_markers=emit_markers,
                call_site_table=call_site_table,
                site_counter=site_counter,
                slot_variant_idx=slot_variant_idx,
            )

        out.append(stmt)
    return out


def inline_all_helpers_source(
    source: str,
    main_func_name: str,
    prefix: str = "",
    *,
    slot_variant_idx: dict[str, int] | None = None,
    emit_markers: bool = False,
    return_call_site_table: bool = False,
):
    """Inline every ``_slot_*`` helper called as ``x = _slot_*(args)``
    inside ``main_func_name``.

    Parameters
    ----------
    slot_variant_idx : optional ``{helper_name: variant_idx}`` recording
        which variant index of each slot population produced the helper
        body being inlined (used solely for provenance annotation).
    emit_markers : when True, bracket each inlined body with sentinel
        ``__fii_provmark_begin / _end`` calls.
    return_call_site_table : when True, returns ``(source, table)`` where
        ``table[call_site_id] = {"slot_id": ..., "variant_idx": ...}``.
    """
    if slot_variant_idx is None:
        slot_variant_idx = {}

    tree = ast.parse(source)
    helpers: dict[str, ast.FunctionDef] = {}
    main_func: ast.FunctionDef | None = None
    other_stmts: list[ast.stmt] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            if stmt.name == main_func_name:
                main_func = stmt
            elif stmt.name.startswith("_slot_"):
                helpers[stmt.name] = stmt
            else:
                other_stmts.append(stmt)
        else:
            other_stmts.append(stmt)

    if main_func is None:
        raise ValueError(
            f"main function {main_func_name!r} not found in source"
        )

    fresh = _FreshName(prefix=prefix)
    call_site_table: dict[int, dict[str, Any]] = {}
    site_counter = [0]
    main_func.body = _inline_stmt_list(
        main_func.body, helpers, fresh,
        emit_markers=emit_markers,
        call_site_table=call_site_table,
        site_counter=site_counter,
        slot_variant_idx=slot_variant_idx,
    )

    new_tree = ast.Module(body=other_stmts + [main_func], type_ignores=[])
    ast.fix_missing_locations(new_tree)
    new_source = ast.unparse(new_tree)

    if return_call_site_table:
        return new_source, call_site_table
    return new_source
