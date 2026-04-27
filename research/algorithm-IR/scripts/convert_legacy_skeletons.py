"""Convert legacy skeleton_library specs (parameter-based slot dispatch) to
annotation-only `with slot(...)` form, producing ``evolution/extended_pool.py``.

Reads JSON dumps produced from a sandboxed exec of the deleted
`evolution/skeleton_library.py` (kept under D:/Temp/skel_specs.json and
D:/Temp/skel_defaults.json) and emits a single Python module that defines:

    EXTENDED_DETECTOR_SPECS : list[_DetectorSpec]
    EXTENDED_DEFAULT_BODIES : dict[str, str]   # raw default function sources

The conversion rule for each statement of the form

        x_new = slot_iterate(G, rhs, x)

(or tuple-unpacking forms) is:

        with slot("<algo>.<short>",
                  inputs=(G, rhs, x),
                  outputs=("x_new",)):
            <renamed inline body of the default function bound to slot_iterate>

Body locals are prefixed with ``_<algo>_<short>__`` to avoid colliding with the
caller's locals; body params are alpha-renamed to the call-site argument names;
each ``return EXPR`` becomes either ``<output_name> = EXPR`` (single output) or
``<o1>, <o2>, ... = EXPR`` (tuple-unpacked).
"""
from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path

SPECS_JSON = Path("D:/Temp/skel_specs.json")
DEFAULTS_JSON = Path("D:/Temp/skel_defaults.json")
OUT_MODULE = Path(__file__).resolve().parents[1] / "evolution" / "extended_pool.py"


# Base default bodies that the legacy skeleton_library.py resolved at runtime
# from a separate registry (now folded inline here so the converter is
# self-contained).
BASE_DEFAULTS: dict[str, str] = {
    "hard_decision": (
        "def hard_decision(x_soft, constellation):\n"
        "    Nt = x_soft.shape[0]\n"
        "    K = constellation.shape[0]\n"
        "    out = np.zeros(Nt, dtype=complex)\n"
        "    i = 0\n"
        "    while i < Nt:\n"
        "        best_idx = 0\n"
        "        best_dist = float('inf')\n"
        "        k = 0\n"
        "        while k < K:\n"
        "            d = abs(x_soft[i] - constellation[k])\n"
        "            if d < best_dist:\n"
        "                best_dist = d\n"
        "                best_idx = k\n"
        "            k = k + 1\n"
        "        out[i] = constellation[best_idx]\n"
        "        i = i + 1\n"
        "    return out\n"
    ),
    "regularizer": (
        "def regularizer(G, sigma2):\n"
        "    Nt = G.shape[0]\n"
        "    return G + sigma2 * np.eye(Nt)\n"
    ),
}


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _slot_short_name(slot_arg_name: str) -> str:
    """``slot_iterate`` -> ``iterate`` (drop the ``slot_`` prefix)."""
    assert slot_arg_name.startswith("slot_"), slot_arg_name
    return slot_arg_name[len("slot_"):]


class _RenameLoadsAndStores(ast.NodeTransformer):
    """Rename Name nodes in both Load and Store contexts according to a map."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node


def _collect_local_names(fn: ast.FunctionDef) -> set[str]:
    """All names assigned-to or used as loop targets inside the function body
    (excluding the parameters themselves)."""
    locs: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                _collect_assign_targets(tgt, locs)
        elif isinstance(node, ast.AugAssign):
            _collect_assign_targets(node.target, locs)
        elif isinstance(node, ast.For):
            _collect_assign_targets(node.target, locs)
    locs.difference_update(a.arg for a in fn.args.args)
    return locs


def _collect_assign_targets(node: ast.AST, into: set[str]) -> None:
    if isinstance(node, ast.Name):
        into.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for el in node.elts:
            _collect_assign_targets(el, into)
    elif isinstance(node, (ast.Subscript, ast.Attribute, ast.Starred)):
        # subscript/attribute assignments don't bind a new name
        pass


def _expr_to_src(node: ast.AST) -> str:
    """Compact source rendering for an AST node."""
    return ast.unparse(node)


def _stmts_to_src(stmts: list[ast.stmt], indent: str = "    ") -> str:
    """Render statements with the given leading indent on each line."""
    out_lines: list[str] = []
    for s in stmts:
        rendered = ast.unparse(s)
        for line in rendered.splitlines():
            out_lines.append(indent + line)
    return "\n".join(out_lines)


# ───────────────────────────────────────────────────────────────────────────
# Default-body inliner
# ───────────────────────────────────────────────────────────────────────────

class _ReturnToAssignTransformer(ast.NodeTransformer):
    """Rewrite ``return EXPR`` as assignment(s) into the slot's declared
    output names. The slot has 1+ outputs; ``return`` may be a single value
    or a tuple. We assume a *single* return at the end of the body (true for
    every default in the legacy library)."""

    def __init__(self, output_names: list[str]) -> None:
        self.output_names = output_names

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if node.value is None:
            raise ValueError("default body has bare 'return' — cannot map to outputs")
        if len(self.output_names) == 1:
            tgt = ast.Name(id=self.output_names[0], ctx=ast.Store())
            return ast.copy_location(ast.Assign(targets=[tgt], value=node.value), node)
        # tuple unpacking
        tgts = ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Store()) for n in self.output_names],
                         ctx=ast.Store())
        return ast.copy_location(ast.Assign(targets=[tgts], value=node.value), node)


def _inline_default_body(
    default_src: str,
    call_args_src: list[str],
    output_names: list[str],
    rename_prefix: str,
) -> list[ast.stmt]:
    """Parse ``default_src`` (a complete `def F(...): ...` source), rewrite it
    so that its body produces statements that assign into ``output_names``.

    - Body parameters are renamed to the source rendering of the call args.
    - Body local names are prefixed with ``rename_prefix`` to avoid collisions.
    - Final ``return EXPR`` becomes an assignment into ``output_names``.
    """
    tree = ast.parse(textwrap.dedent(default_src))
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef), f"default not a single FunctionDef: {default_src[:120]}"

    param_names = [a.arg for a in fn.args.args]
    if len(param_names) != len(call_args_src):
        # Some defaults take more / fewer args than the call site supplies
        # (e.g. body has H but call only passes G). We still rename one-for-one
        # over the prefix and leave the rest of the body's parameter names
        # un-rewritten (they will be looked up in the enclosing template scope).
        n = min(len(param_names), len(call_args_src))
    else:
        n = len(param_names)

    # Build rename map: param i -> call arg src; locals -> prefixed
    locals_set = _collect_local_names(fn)
    name_map: dict[str, str] = {}
    # Param renames: only rename when the call-site arg is itself a plain Name;
    # otherwise we leave the param name and emit a leading assign that binds
    # the param to the expression.
    leading_binds: list[ast.stmt] = []
    for i in range(n):
        param = param_names[i]
        arg_src = call_args_src[i]
        try:
            arg_expr = ast.parse(arg_src, mode="eval").body
        except SyntaxError:
            arg_expr = ast.Name(id=arg_src, ctx=ast.Load())
        if isinstance(arg_expr, ast.Name):
            name_map[param] = arg_expr.id
        else:
            # Bind expression to a scoped local so the rest of body can use param
            scoped = f"{rename_prefix}{param}"
            name_map[param] = scoped
            leading_binds.append(
                ast.Assign(
                    targets=[ast.Name(id=scoped, ctx=ast.Store())],
                    value=arg_expr,
                )
            )
    # Local renames
    for loc in locals_set:
        if loc in name_map:
            continue
        name_map[loc] = f"{rename_prefix}{loc}"

    body = [s for s in fn.body]
    # Rename load/store sites
    transformer = _RenameLoadsAndStores(name_map)
    body = [transformer.visit(s) for s in body]
    # Now rewrite return → assign
    rt = _ReturnToAssignTransformer(output_names)
    body = [rt.visit(s) for s in body]
    # Need to ast.fix_missing_locations on rewrites
    for s in body:
        ast.fix_missing_locations(s)
    return leading_binds + body


# ───────────────────────────────────────────────────────────────────────────
# Spec converter
# ───────────────────────────────────────────────────────────────────────────

class _SpecConverter(ast.NodeTransformer):
    def __init__(self, spec: dict, defaults: dict[str, str]) -> None:
        self.spec = spec
        self.defaults = defaults
        self.algo_id: str = spec["algo_id"]
        self.slot_arg_names: list[str] = list(spec["slot_arg_names"])
        self.slot_default_keys: dict[str, str] = dict(spec["slot_default_keys"])
        self.encountered_slots: list[tuple[str, list[str], list[str]]] = []
        # Track which slot shorts have been used (to disambiguate repeats)
        self._slot_use_counts: dict[str, int] = {}
        self._tmp_counter = 0

    def _new_tmp(self, hint: str = "tmp") -> str:
        self._tmp_counter += 1
        return f"_{self.algo_id}__{hint}_{self._tmp_counter}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Strip slot_xxx parameters from the signature
        node.args.args = [a for a in node.args.args
                          if a.arg not in self.slot_arg_names]
        node.body = self._rewrite_body(node.body)
        return node

    def _rewrite_body(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for s in stmts:
            new_stmts = self._rewrite_stmt(s)
            out.extend(new_stmts)
        return out

    def _rewrite_stmt(self, stmt: ast.stmt) -> list[ast.stmt]:
        # Recurse into compound statements first so nested slot calls are
        # rewritten too.
        if isinstance(stmt, (ast.For, ast.While)):
            stmt.body = self._rewrite_body(stmt.body)
            stmt.orelse = self._rewrite_body(stmt.orelse) if stmt.orelse else []
            return [stmt]
        if isinstance(stmt, ast.If):
            stmt.body = self._rewrite_body(stmt.body)
            stmt.orelse = self._rewrite_body(stmt.orelse) if stmt.orelse else []
            return [stmt]
        if isinstance(stmt, ast.With):
            stmt.body = self._rewrite_body(stmt.body)
            return [stmt]

        # Direct: `target = slot_X(...)` -> with-block
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id in self.slot_arg_names:
                pre, with_block = self._lower_slot_call(stmt.targets[0], call)
                return pre + [with_block]

        # Otherwise: scan stmt for embedded slot_X calls and spill them to
        # temporary assignments preceding this stmt.
        pre, new_stmt = self._spill_embedded_slot_calls(stmt)
        return pre + [new_stmt]

    def _spill_embedded_slot_calls(self, stmt: ast.stmt) -> tuple[list[ast.stmt], ast.stmt]:
        """Find any ``slot_X(...)`` call inside ``stmt`` (other than at the
        top-level RHS of an Assign — that case is handled directly) and replace
        it with a fresh temp name. Emit the lowered slot blocks before the
        original stmt.
        """
        pre: list[ast.stmt] = []

        class _Spill(ast.NodeTransformer):
            def __init__(inner, outer):
                inner.outer = outer

            def visit_Call(inner, node: ast.Call):
                # Recurse into args first
                inner.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id in inner.outer.slot_arg_names:
                    tmp = inner.outer._new_tmp(node.func.id[5:])  # drop slot_
                    tmp_target = ast.Name(id=tmp, ctx=ast.Store())
                    pre_stmts, with_block = inner.outer._lower_slot_call(tmp_target, node)
                    pre.extend(pre_stmts)
                    pre.append(with_block)
                    return ast.Name(id=tmp, ctx=ast.Load())
                return node

        new_stmt = _Spill(self).visit(stmt)
        ast.fix_missing_locations(new_stmt)
        return pre, new_stmt

    def _lower_slot_call(
        self, target: ast.AST, call: ast.Call
    ) -> tuple[list[ast.stmt], ast.With]:
        """Build (pre_stmts, with_block) for ``target = slot_X(args)``.

        ``target`` is the LHS AST node (Name or Tuple of Names) where the slot
        result(s) should be bound.
        """
        slot_arg = call.func.id  # e.g. "slot_iterate"
        short = _slot_short_name(slot_arg)
        idx = self._slot_use_counts.get(short, 0)
        self._slot_use_counts[short] = idx + 1
        unique_short = short if idx == 0 else f"{short}_{idx + 1}"
        pop_key = f"{self.algo_id}.{unique_short}"

        if call.keywords:
            raise NotImplementedError(
                f"{self.algo_id}: slot call '{slot_arg}' uses keyword args — not supported"
            )

        # Spill non-Name args to temp variables BEFORE the with-block, so the
        # slot's `inputs=(...)` tuple contains only plain Names.
        pre_stmts: list[ast.stmt] = []
        input_names: list[str] = []
        for a in call.args:
            if isinstance(a, ast.Name):
                input_names.append(a.id)
            else:
                tmp = self._new_tmp(f"in")
                pre_stmts.append(ast.Assign(
                    targets=[ast.Name(id=tmp, ctx=ast.Store())],
                    value=a,
                ))
                input_names.append(tmp)

        # Outputs: LHS targets
        if isinstance(target, ast.Name):
            output_names = [target.id]
        elif isinstance(target, ast.Tuple):
            output_names = []
            for el in target.elts:
                if not isinstance(el, ast.Name):
                    raise NotImplementedError(
                        f"{self.algo_id}: slot call tuple-unpack with non-Name target"
                    )
                output_names.append(el.id)
        else:
            raise NotImplementedError(
                f"{self.algo_id}: slot call assigned to non-Name/tuple target"
            )

        # Look up default body
        default_key = self.slot_default_keys.get(slot_arg)
        if not default_key:
            raise KeyError(
                f"{self.algo_id}: no default key for slot arg '{slot_arg}'"
            )
        default_src = self.defaults.get(default_key)
        if not default_src:
            raise KeyError(
                f"{self.algo_id}: default body '{default_key}' missing"
            )

        rename_prefix = f"_{self.algo_id}_{unique_short}__"
        body_stmts = _inline_default_body(
            default_src,
            input_names,    # all Names now
            output_names,
            rename_prefix,
        )

        self.encountered_slots.append((pop_key, list(input_names), list(output_names)))

        inputs_tuple = ast.Tuple(
            elts=[ast.Name(id=n, ctx=ast.Load()) for n in input_names],
            ctx=ast.Load(),
        )
        outputs_tuple = ast.Tuple(
            elts=[ast.Constant(value=n) for n in output_names],
            ctx=ast.Load(),
        )
        with_call = ast.Call(
            func=ast.Name(id="slot", ctx=ast.Load()),
            args=[ast.Constant(value=pop_key)],
            keywords=[
                ast.keyword(arg="inputs", value=inputs_tuple),
                ast.keyword(arg="outputs", value=outputs_tuple),
            ],
        )
        with_item = ast.withitem(context_expr=with_call, optional_vars=None)
        with_node = ast.With(items=[with_item], body=body_stmts)
        for s in pre_stmts:
            ast.fix_missing_locations(s)
        ast.fix_missing_locations(with_node)
        return pre_stmts, with_node


def _is_ident(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s))


# ───────────────────────────────────────────────────────────────────────────
# Top-level driver
# ───────────────────────────────────────────────────────────────────────────

def convert_one(spec: dict, defaults: dict[str, str]) -> tuple[str, list[tuple[str, list[str], list[str]]]]:
    """Convert a single spec; return (new_source, [(pop_key, inputs, outputs), ...])."""
    tree = ast.parse(textwrap.dedent(spec["source"]))
    converter = _SpecConverter(spec, defaults)
    new_tree = converter.visit(tree)
    ast.fix_missing_locations(new_tree)
    new_src = ast.unparse(new_tree)
    return new_src, converter.encountered_slots


def main() -> None:
    specs = json.loads(SPECS_JSON.read_text(encoding="utf-8"))
    defaults = json.loads(DEFAULTS_JSON.read_text(encoding="utf-8"))
    # Merge in base defaults so that 'hard_decision', 'regularizer', etc.
    # resolve. Legacy library resolved these at runtime from a separate map.
    for k, v in BASE_DEFAULTS.items():
        defaults.setdefault(k, v)

    converted: list[dict] = []
    failures: list[tuple[str, str]] = []
    for spec in specs:
        try:
            new_src, slots = convert_one(spec, defaults)
            converted.append({
                "algo_id": spec["algo_id"],
                "func_name": spec["func_name"],
                "source": new_src,
                "tags": list(spec["tags"]),
                "level": spec["level"],
                "slots": slots,
            })
        except Exception as e:
            failures.append((spec["algo_id"], str(e)))

    print(f"Converted {len(converted)} / {len(specs)} specs")
    if failures:
        print(f"Failures ({len(failures)}):")
        for aid, err in failures[:10]:
            print(f"  {aid}: {err}")

    # Emit the new module
    parts = [
        '"""Auto-generated extended detector pool — annotation-only `with slot(...)` form.\n\n'
        'This module is generated by ``scripts/convert_legacy_skeletons.py`` from the\n'
        'legacy ``skeleton_library.py`` (deleted in commit 189b59a). Do **not** edit by\n'
        'hand; re-run the converter instead.\n\n'
        'Each entry is a (source, func_name, tags, level) tuple consumed by\n'
        '``evolution.ir_pool.build_ir_pool``.\n"""\n',
        "from __future__ import annotations\n",
        "",
        "EXTENDED_DETECTOR_TEMPLATES: list[dict] = [\n",
    ]
    for c in converted:
        parts.append("    {\n")
        parts.append(f'        "algo_id": {c["algo_id"]!r},\n')
        parts.append(f'        "func_name": {c["func_name"]!r},\n')
        parts.append(f'        "level": {c["level"]!r},\n')
        parts.append(f'        "tags": {sorted(c["tags"])!r},\n')
        slots_repr = [(p, list(i), list(o)) for p, i, o in c["slots"]]
        parts.append(f'        "slots": {slots_repr!r},\n')
        parts.append('        "source": (\n')
        for line in c["source"].splitlines():
            esc = line.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'            "{esc}\\n"\n')
        parts.append('        ),\n')
        parts.append("    },\n")
    parts.append("]\n")

    OUT_MODULE.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {OUT_MODULE} ({OUT_MODULE.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
