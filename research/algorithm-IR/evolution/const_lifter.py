"""Const-lifter — AST pass that promotes hardcoded numeric literals and
loop bounds to 0-arg const-slot calls (Step S1 / code_review.md §3.13.4
and §3.13.5).

The transform operates on **Python source strings** (the same form the
skeleton library stores its templates in) and returns a rewritten
source plus a manifest of the lifted constants so the caller can
register matching `ProgramSpec`s and seed `SlotPopulation` defaults.

Scope (kept deliberately narrow per S1 spec):

* Lift numeric `Constant` literals inside function bodies to calls
  `_slot_<func>__c<idx>()`.  Values in ``EXEMPT_VALUES`` (canonical
  noise floors, identity multipliers, 0/1 indices) are skipped to avoid
  cluttering pools with trivial slots.
* Lift integer literals appearing as the right-hand side of comparison
  in `while i < N` and `while i <= N` loops to a const-slot
  `_slot_<func>__loop<idx>()`.
* Skip literals that already appear inside `_slot_*` call arguments
  (those are explicit slot wiring, not domain constants).
* Idempotent: a second pass on lifted source yields no further lifts.

The lifter does NOT touch:
* String, bytes, bool, None constants.
* Literals used as keyword-argument defaults of ``def`` (these are
  template signatures, not body constants).
* Literals inside type-annotation subscripts.

Public API:

    lift_source(src, *, exempt=None, lift_loops=True) -> LiftResult
    LiftResult.new_source : rewritten Python source
    LiftResult.lifted     : list[LiftedConstant]
    LiftResult.diagnostic : str (one-line summary)
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Constants too trivial / domain-agnostic to be worth lifting.
EXEMPT_VALUES: frozenset = frozenset({
    0, 1, -1,
    0.0, 1.0, -1.0,
    0.5,            # damping default — but kept as exempt by request:
                    # damping is lifted via dedicated _PS_DAMPING spec
                    # in the skeleton library, not by the generic lifter.
    1e-30, 1e-12, 1e-9, 1e-6,  # standard numerical floors
    2, 3, 4,                    # very small loop indices
})

#: Maximum number of literals to lift per function body (defensive cap).
MAX_LIFTS_PER_FUNC: int = 16


@dataclass
class LiftedConstant:
    """Description of one lifted literal."""
    slot_name: str
    original_value: int | float | complex
    type_hint: str            # "int" / "float" / "complex"
    func_name: str            # owning function in the source
    role: str                 # "literal" | "loop_bound"
    line: int


@dataclass
class LiftResult:
    new_source: str
    lifted: list[LiftedConstant] = field(default_factory=list)
    diagnostic: str = ""


# ---------------------------------------------------------------------------
# Internal AST visitor
# ---------------------------------------------------------------------------

class _LiftRewriter(ast.NodeTransformer):
    def __init__(
        self,
        *,
        exempt: frozenset,
        lift_loops: bool,
    ) -> None:
        self.exempt = exempt
        self.lift_loops = lift_loops
        self.lifted: list[LiftedConstant] = []
        self._func_stack: list[str] = []
        self._lift_counts: dict[str, int] = {}

    # ----- helpers -----
    def _next_idx(self, fname: str) -> int:
        n = self._lift_counts.get(fname, 0)
        self._lift_counts[fname] = n + 1
        return n

    def _is_already_slot_call(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("_slot_")
        )

    def _classify(self, value) -> str | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, complex):
            return "complex"
        return None

    def _lift(self, value, role: str, lineno: int) -> ast.Call | None:
        if not self._func_stack:
            return None
        fname = self._func_stack[-1]
        if self._lift_counts.get(fname, 0) >= MAX_LIFTS_PER_FUNC:
            return None
        if value in self.exempt:
            return None
        type_hint = self._classify(value)
        if type_hint is None:
            return None
        idx = self._next_idx(fname)
        prefix = "loop" if role == "loop_bound" else "c"
        slot_name = f"_slot_{fname}__{prefix}{idx}"
        self.lifted.append(LiftedConstant(
            slot_name=slot_name,
            original_value=value,
            type_hint=type_hint,
            func_name=fname,
            role=role,
            line=lineno,
        ))
        return ast.Call(
            func=ast.Name(id=slot_name, ctx=ast.Load()),
            args=[],
            keywords=[],
        )

    # ----- visitors -----
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._func_stack.append(node.name)
        # Skip rewriting argument defaults / annotations entirely:
        # only walk the body.
        node.body = [self.visit(b) for b in node.body]
        self._func_stack.pop()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        # Don't touch the annotation; only rewrite the value.
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Don't lift indices like `x[0]` or `x[1]`; numpy/list indexing
        # constants are structural, not numerical hyperparameters.
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # If this is already a slot call, don't recurse into its args.
        if self._is_already_slot_call(node):
            return node
        node.func = self.visit(node.func)
        node.args = [self.visit(a) for a in node.args]
        node.keywords = [
            ast.keyword(arg=k.arg, value=self.visit(k.value))
            for k in node.keywords
        ]
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        # Special-case `while i < N` / `while i <= N` for loop-bound lifting.
        if (
            self.lift_loops
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], (ast.Lt, ast.LtE))
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
        ):
            const = node.test.comparators[0]
            replacement = self._lift(const.value, "loop_bound", const.lineno)
            if replacement is not None:
                node.test.comparators[0] = replacement
        # Recurse into body / orelse normally.
        node.body = [self.visit(b) for b in node.body]
        node.orelse = [self.visit(b) for b in node.orelse]
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        replacement = self._lift(node.value, "literal", node.lineno)
        return replacement if replacement is not None else node


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lift_source(
    src: str,
    *,
    exempt: Iterable | None = None,
    lift_loops: bool = True,
) -> LiftResult:
    """Lift hardcoded numeric literals in ``src`` to const-slot calls.

    Parameters
    ----------
    src : str
        Python source code containing one or more ``def`` blocks.
    exempt : iterable of values, optional
        Values to leave untouched.  Defaults to :data:`EXEMPT_VALUES`.
    lift_loops : bool
        If True, also lift integer bounds in ``while i < N`` loops.
    """
    exempt_set = frozenset(exempt) if exempt is not None else EXEMPT_VALUES
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return LiftResult(
            new_source=src,
            lifted=[],
            diagnostic=f"parse_error: {exc.msg}",
        )
    rewriter = _LiftRewriter(exempt=exempt_set, lift_loops=lift_loops)
    new_tree = rewriter.visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        new_src = ast.unparse(new_tree)
    except Exception as exc:
        logger.warning("ast.unparse failed: %r; returning original", exc)
        return LiftResult(
            new_source=src,
            lifted=[],
            diagnostic=f"unparse_error: {exc!r}",
        )
    diag = (
        f"lifted {len(rewriter.lifted)} literals "
        f"({sum(1 for l in rewriter.lifted if l.role == 'loop_bound')} loop bounds)"
    )
    return LiftResult(
        new_source=new_src,
        lifted=rewriter.lifted,
        diagnostic=diag,
    )


def make_const_slot_default_source(lifted: LiftedConstant) -> str:
    """Emit a 0-arg slot definition that returns the original constant.

    Suitable for direct registration as a default ``slot_<name>``
    implementation in :mod:`evolution.skeleton_library`.
    """
    val = lifted.original_value
    if lifted.type_hint == "complex":
        repr_val = f"complex({val.real!r}, {val.imag!r})"
    else:
        repr_val = repr(val)
    return (
        f"def {lifted.slot_name}():\n"
        f"    \"\"\"Auto-lifted const ({lifted.type_hint}) "
        f"from {lifted.func_name} L{lifted.line}.\"\"\"\n"
        f"    return {repr_val}\n"
    )


def is_idempotent(src: str, *, exempt: Iterable | None = None) -> bool:
    """Return True iff a second lift pass produces no further changes."""
    first = lift_source(src, exempt=exempt)
    second = lift_source(first.new_source, exempt=exempt)
    return len(second.lifted) == 0
