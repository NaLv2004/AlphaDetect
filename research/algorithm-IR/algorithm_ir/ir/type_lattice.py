"""Type lattice for the Algorithm IR.

Single source of truth for value types across the IR, lifter, grafter,
and GP synthesiser.  Owns:

* The atomic vocabulary (``vec_f``, ``mat_cx``, ``cx``, ``prob_table`` …)
* The subtype lattice (``is_subtype``, ``unify``, ancestors)
* Composite type parsing (``tuple<...>``, ``list<T>``, ``dict<T>``)
* A numpy / scipy / builtin function registry mapping callables to their
  return type (``infer_call_return_type``)
* Lattice-aware binary / unary op rules respecting numpy broadcasting
  shape and dtype (``combine_binary_type``, ``combine_unary_type``)
* Concrete fallback values per type (``default_value``)
* Runtime introspection (``infer_value_type`` over Python values)

The lattice is intentionally **not a full Hindley–Milner system** — its
sole purpose is to prune the search space of GP operators and to enable
type-checked host/donor binding during grafting.

Architectural note: this file lives under ``algorithm_ir/ir/`` because
types are a property of the IR, not of the evolutionary engine.  A
backward-compat shim at ``evolution.types_lattice`` re-exports the
same names.

Type vocabulary
---------------

    PRIMITIVE  ::= "int" | "float" | "bool" | "cx"
    TENSOR     ::= "vec_f" | "vec_cx" | "vec_i" | "mat_f" | "mat_cx"
                   | "tensor3_f" | "tensor3_cx"
    COMPOSITE  ::= "tuple<T1,...,Tn>" | "list<T>" | "dict<T>"
    OBJECT     ::= "node" | "candidate_list" | "open_set"
                   | "mat_decomp" | "prob_table"
    UNIVERSAL  ::= "any"      — top type (only used as fallback)
                  "void"      — for statements with no value

Composite types use a small string-encoded grammar so they remain
serialisable as ``Value.attrs`` strings without dataclass nesting.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

__all__ = [
    "PRIMITIVE_TYPES",
    "TENSOR_TYPES",
    "OBJECT_TYPES",
    "ALL_ATOMIC_TYPES",
    "TYPE_TOP",
    "TYPE_VOID",
    "is_subtype",
    "unify",
    "available_ops_for_type",
    "default_value",
    "infer_value_type",
    "parse_composite",
    "is_tuple_type",
    "is_list_type",
    "is_dict_type",
    "tuple_components",
    "list_element_type",
    "dict_value_type",
    # Extended completeness API
    "combine_binary_type",
    "combine_unary_type",
    "infer_call_return_type",
    "register_callable_return",
    "callable_return_type",
    "is_numeric",
    "is_array_like",
    "is_real",
    "is_complex",
    "promote_dtype",
    "promote_rank",
]

# ---------------------------------------------------------------------------
# Atomic vocabulary
# ---------------------------------------------------------------------------

PRIMITIVE_TYPES: tuple[str, ...] = ("int", "float", "bool", "cx")
TENSOR_TYPES: tuple[str, ...] = (
    "vec_f", "vec_cx", "vec_i",
    "mat_f", "mat_cx",
    "tensor3_f", "tensor3_cx",
)
OBJECT_TYPES: tuple[str, ...] = (
    "node",          # discrete-search candidate (cost + symbols)
    "candidate_list", "open_set",
    "mat_decomp",    # generic decomposition tuple holder
    "prob_table",    # discrete distribution per symbol
)
ALL_ATOMIC_TYPES: tuple[str, ...] = (
    PRIMITIVE_TYPES + TENSOR_TYPES + OBJECT_TYPES + ("any", "void", "object")
)

TYPE_TOP = "any"
TYPE_VOID = "void"


# ---------------------------------------------------------------------------
# Subtype lattice
# ---------------------------------------------------------------------------

# Direct super-type relation.  Transitive closure handled by walk.
# Read as: ``key`` is a subtype of every entry in the value list.
_SUPER: dict[str, tuple[str, ...]] = {
    "bool":      ("int", "float", "any"),
    "int":       ("float", "cx", "any"),
    "float":     ("cx", "any"),
    "cx":        ("any",),
    "vec_i":     ("vec_f", "any"),
    "vec_f":     ("vec_cx", "any"),
    "vec_cx":    ("any",),
    "mat_f":     ("mat_cx", "any"),
    "mat_cx":    ("any",),
    "tensor3_f": ("tensor3_cx", "any"),
    "tensor3_cx":("any",),
    "node":      ("object", "any"),
    "candidate_list": ("object", "any"),
    "open_set":  ("object", "any"),
    "mat_decomp": ("object", "any"),
    "prob_table": ("vec_f", "any"),  # a probability table IS a real vector
    "object":    ("any",),
    "void":      (),
    "any":       (),
}


def _ancestors(t: str) -> set[str]:
    """Return ``t`` and all transitive supertypes (atomic only)."""
    seen: set[str] = set()
    stack = [t]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for parent in _SUPER.get(cur, ()):
            if parent not in seen:
                stack.append(parent)
    return seen


# ---------------------------------------------------------------------------
# Composite parsing
# ---------------------------------------------------------------------------

_COMPOSITE_RE = re.compile(r"^(tuple|list|dict)<(.+)>$")


def parse_composite(t: str) -> tuple[str, list[str]] | None:
    """Return ``("tuple"|"list"|"dict", [components])`` or ``None``."""
    m = _COMPOSITE_RE.match(t.strip())
    if not m:
        return None
    head = m.group(1)
    body = m.group(2).strip()
    parts = _split_top_level(body, sep=",")
    return head, parts


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out


def is_tuple_type(t: str) -> bool:
    p = parse_composite(t)
    return p is not None and p[0] == "tuple"


def is_list_type(t: str) -> bool:
    p = parse_composite(t)
    return p is not None and p[0] == "list"


def is_dict_type(t: str) -> bool:
    p = parse_composite(t)
    return p is not None and p[0] == "dict"


def tuple_components(t: str) -> list[str]:
    p = parse_composite(t)
    if p is None or p[0] != "tuple":
        return []
    return p[1]


def list_element_type(t: str) -> str:
    p = parse_composite(t)
    if p is None or p[0] != "list" or not p[1]:
        return TYPE_TOP
    return p[1][0]


def dict_value_type(t: str) -> str:
    p = parse_composite(t)
    if p is None or p[0] != "dict" or not p[1]:
        return TYPE_TOP
    return p[1][0]


# ---------------------------------------------------------------------------
# is_subtype
# ---------------------------------------------------------------------------

def is_subtype(a: str, b: str) -> bool:
    """Return True iff every value of type ``a`` is also of type ``b``.

    Atomic <: any; composite covariant in components.
    Unknown atomic types are conservatively treated as ``any``.
    """
    if a == b:
        return True
    if b == TYPE_TOP:
        return True
    if a == TYPE_VOID or b == TYPE_VOID:
        return False

    pa = parse_composite(a)
    pb = parse_composite(b)
    if pa is not None and pb is not None:
        if pa[0] != pb[0]:
            return False
        if pa[0] == "tuple":
            if len(pa[1]) != len(pb[1]):
                return False
            return all(is_subtype(x, y) for x, y in zip(pa[1], pb[1]))
        # list / dict — covariant in element type
        return is_subtype(pa[1][0], pb[1][0])

    # Atomic / atomic (or atomic / composite -> False)
    if pa is not None or pb is not None:
        return False

    # Both atomic
    return b in _ancestors(a)


# ---------------------------------------------------------------------------
# unify (least common supertype)
# ---------------------------------------------------------------------------

def unify(a: str, b: str) -> str:
    """Return the least common supertype of ``a`` and ``b``.

    For composites, recurse component-wise; if shapes mismatch, fall back
    to ``any``.  For atomics, walk both ancestor sets and pick a common
    ancestor with shortest path from ``a``.
    """
    if a == b:
        return a
    if a == TYPE_VOID or b == TYPE_VOID:
        return TYPE_VOID
    if a == TYPE_TOP or b == TYPE_TOP:
        return TYPE_TOP

    pa = parse_composite(a)
    pb = parse_composite(b)
    if pa is not None and pb is not None and pa[0] == pb[0]:
        if pa[0] == "tuple":
            if len(pa[1]) != len(pb[1]):
                return TYPE_TOP
            comps = [unify(x, y) for x, y in zip(pa[1], pb[1])]
            return f"tuple<{','.join(comps)}>"
        if pa[0] == "list":
            return f"list<{unify(pa[1][0], pb[1][0])}>"
        if pa[0] == "dict":
            return f"dict<{unify(pa[1][0], pb[1][0])}>"
    if pa is not None or pb is not None:
        return TYPE_TOP

    # Both atomic — find shallowest shared ancestor
    anc_a = _ancestors(a)
    anc_b = _ancestors(b)
    common = anc_a & anc_b
    if not common:
        return TYPE_TOP
    if a in common:
        return a
    if b in common:
        return b
    # Pick the first non-"any" ancestor walking from a
    queue = [a]
    seen: set[str] = set()
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        if cur in common and cur != TYPE_TOP:
            return cur
        for parent in _SUPER.get(cur, ()):
            queue.append(parent)
    return TYPE_TOP


# ---------------------------------------------------------------------------
# available_ops_for_type — coarse opcode catalogue
# ---------------------------------------------------------------------------

# These opcode strings are the IR opcodes used by `algorithm_ir.ir.model`.
# Each entry maps a producible type to the opcodes that may produce it.
# A union of opcode names across all subtypes is returned by the public
# helper below.

_PRODUCERS: dict[str, tuple[str, ...]] = {
    "bool":   ("const", "binary", "compare", "unary", "call"),
    "int":    ("const", "binary", "unary", "call"),
    "float":  ("const", "binary", "unary", "call", "get_item", "get_attr"),
    "cx":     ("const", "binary", "unary", "call", "get_item"),
    "vec_f":  ("const", "call", "binary", "get_item", "get_attr"),
    "vec_cx": ("const", "call", "binary", "get_item", "get_attr"),
    "vec_i":  ("const", "call", "binary", "get_item"),
    "mat_f":  ("const", "call", "binary", "get_item", "get_attr"),
    "mat_cx": ("const", "call", "binary", "get_item", "get_attr"),
    "tensor3_f":  ("const", "call"),
    "tensor3_cx": ("const", "call"),
    "node":            ("call", "const"),
    "candidate_list":  ("call", "const"),
    "open_set":        ("call", "const"),
    "mat_decomp":      ("call",),
    "prob_table":      ("call", "binary"),
    "object":          ("call", "const", "get_attr", "get_item"),
    "any":             ("const", "call", "binary", "unary", "compare",
                        "get_item", "get_attr", "phi"),
    "void":            ("return", "store"),
}


def available_ops_for_type(t: str) -> tuple[str, ...]:
    """Return the IR opcodes that may produce a value of type ``t``.

    Walks the supertype chain so a vec_i request returns ops from vec_f
    / vec_cx / any too.  Composite types reduce to their head category.
    """
    pc = parse_composite(t)
    if pc is not None:
        if pc[0] == "tuple":
            return ("call", "const")
        if pc[0] == "list":
            return ("call", "const", "get_item")
        if pc[0] == "dict":
            return ("call", "const", "get_item", "get_attr")

    seen: set[str] = set()
    queue = [t]
    out: list[str] = []
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        for op in _PRODUCERS.get(cur, ()):
            if op not in out:
                out.append(op)
        for parent in _SUPER.get(cur, ()):
            queue.append(parent)
    if not out:
        out = list(_PRODUCERS["any"])
    return tuple(out)


# ---------------------------------------------------------------------------
# default_value
# ---------------------------------------------------------------------------

_DEFAULT_VEC_LEN = 4
_DEFAULT_MAT_DIM = 4


def default_value(t: str) -> Any:
    """Return a concrete fallback value of type ``t``.

    Used as the safety net by GP synthesis when a sub-DAG cannot be
    generated.  Never returns ``None`` for non-void types.
    """
    if t == TYPE_VOID:
        return None
    pc = parse_composite(t)
    if pc is not None:
        if pc[0] == "tuple":
            return tuple(default_value(c) for c in pc[1])
        if pc[0] == "list":
            return [default_value(pc[1][0])]
        if pc[0] == "dict":
            return {"_": default_value(pc[1][0])}
    if t == "bool":
        return False
    if t == "int":
        return 0
    if t == "float":
        return 0.0
    if t == "cx":
        return 0.0 + 0.0j
    if t == "vec_i":
        return np.zeros(_DEFAULT_VEC_LEN, dtype=np.int64)
    if t == "vec_f" or t == "prob_table":
        v = np.full(_DEFAULT_VEC_LEN, 1.0 / _DEFAULT_VEC_LEN, dtype=np.float64) \
            if t == "prob_table" else np.zeros(_DEFAULT_VEC_LEN, dtype=np.float64)
        return v
    if t == "vec_cx":
        return np.zeros(_DEFAULT_VEC_LEN, dtype=np.complex128)
    if t == "mat_f":
        return np.eye(_DEFAULT_MAT_DIM, dtype=np.float64)
    if t == "mat_cx":
        return np.eye(_DEFAULT_MAT_DIM, dtype=np.complex128)
    if t == "tensor3_f":
        return np.zeros((2, _DEFAULT_MAT_DIM, _DEFAULT_MAT_DIM), dtype=np.float64)
    if t == "tensor3_cx":
        return np.zeros((2, _DEFAULT_MAT_DIM, _DEFAULT_MAT_DIM), dtype=np.complex128)
    if t == "node":
        return {"cost": 0.0, "symbols": []}
    if t == "candidate_list":
        return []
    if t == "open_set":
        return set()
    if t == "mat_decomp":
        I = np.eye(_DEFAULT_MAT_DIM, dtype=np.complex128)
        return {"Q": I.copy(), "R": I.copy()}
    if t == "object":
        return None
    if t == TYPE_TOP:
        return 0.0
    return None


# ---------------------------------------------------------------------------
# infer_value_type — single-pass type inference for a runtime value
# ---------------------------------------------------------------------------

def infer_value_type(value: Any) -> str:
    """Best-effort inference of a type label from a runtime Python value."""
    if value is None:
        return TYPE_VOID
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    if isinstance(value, (complex, np.complexfloating)):
        return "cx"
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            if np.issubdtype(value.dtype, np.complexfloating):
                return "vec_cx"
            if np.issubdtype(value.dtype, np.integer):
                return "vec_i"
            return "vec_f"
        if value.ndim == 2:
            if np.issubdtype(value.dtype, np.complexfloating):
                return "mat_cx"
            return "mat_f"
        if value.ndim == 3:
            if np.issubdtype(value.dtype, np.complexfloating):
                return "tensor3_cx"
            return "tensor3_f"
        return TYPE_TOP
    if isinstance(value, tuple):
        comps = ",".join(infer_value_type(v) for v in value)
        return f"tuple<{comps}>"
    if isinstance(value, list):
        if not value:
            return f"list<{TYPE_TOP}>"
        # Use unification across elements (capped to 16 for cost)
        t = infer_value_type(value[0])
        for el in value[1:16]:
            t = unify(t, infer_value_type(el))
        return f"list<{t}>"
    if isinstance(value, set):
        return "open_set"
    if isinstance(value, dict):
        if "Q" in value and "R" in value:
            return "mat_decomp"
        if not value:
            return f"dict<{TYPE_TOP}>"
        first = next(iter(value.values()))
        return f"dict<{infer_value_type(first)}>"
    return "object"



# ===========================================================================
# Extended completeness: dtype/rank predicates, callable-return registry,
# and lattice-aware binary/unary combiners.
# ===========================================================================
#
# The atomic vocabulary above gives us *what* a value is.  The pieces in
# this section give us the operational rules needed by the IR lifter and
# the grafter:
#
#   * is_numeric / is_real / is_complex / is_array_like  �?predicates used
#     by the lifter to decide whether to apply numeric promotion.
#   * promote_dtype / promote_rank �?algebraic combiners over (dtype, rank)
#     that mirror numpy broadcasting.
#   * combine_binary_type / combine_unary_type �?given AST operator names
#     and operand lattice types, return the result lattice type.  This is
#     what the lifter calls in place of ``TypeInfo("object")``.
#   * register_callable_return / callable_return_type / infer_call_return_type
#     �?a registry mapping numpy / scipy / builtin callables to their
#     return type, so ``np.linalg.inv(H)`` lifts to ``mat_cx`` instead of
#     ``object``.
#
# Together these make the lifter produce lattice-conformant tags on
# every SSA value, which in turn makes ``is_subtype`` checks meaningful
# inside the grafter's host/donor binding.

# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

_REAL_ATOMIC = frozenset({"bool", "int", "float"})
_COMPLEX_ATOMIC = frozenset({"cx"})
_VEC_TYPES = frozenset({"vec_i", "vec_f", "vec_cx"})
_MAT_TYPES = frozenset({"mat_f", "mat_cx"})
_TENSOR3_TYPES = frozenset({"tensor3_f", "tensor3_cx"})
_ARRAY_TYPES = _VEC_TYPES | _MAT_TYPES | _TENSOR3_TYPES | {"prob_table"}


def is_numeric(t: str) -> bool:
    """True iff ``t`` denotes a number-like value (scalar or tensor)."""
    return t in _REAL_ATOMIC or t in _COMPLEX_ATOMIC or t in _ARRAY_TYPES


def is_array_like(t: str) -> bool:
    """True iff ``t`` is a numpy-array-like tensor type."""
    return t in _ARRAY_TYPES


def is_real(t: str) -> bool:
    """True iff ``t`` only carries real numbers (no complex component)."""
    return t in {"bool", "int", "float", "vec_i", "vec_f", "mat_f",
                 "tensor3_f", "prob_table"}


def is_complex(t: str) -> bool:
    """True iff ``t`` may carry complex numbers."""
    return t in {"cx", "vec_cx", "mat_cx", "tensor3_cx"}


# ---------------------------------------------------------------------------
# Dtype / rank algebra (mirrors numpy broadcasting, lattice-encoded)
# ---------------------------------------------------------------------------

# Maps each lattice type to (rank, dtype-class).
#   rank: 0 = scalar, 1 = vector, 2 = matrix, 3 = 3D-tensor, -1 = unknown.
#   dtype-class: "i" (integer), "f" (real float), "c" (complex), "?" (unknown).
_TYPE_DECOMP: dict[str, tuple[int, str]] = {
    "bool": (0, "i"), "int": (0, "i"), "float": (0, "f"), "cx": (0, "c"),
    "vec_i": (1, "i"), "vec_f": (1, "f"), "vec_cx": (1, "c"),
    "prob_table": (1, "f"),
    "mat_f": (2, "f"), "mat_cx": (2, "c"),
    "tensor3_f": (3, "f"), "tensor3_cx": (3, "c"),
}

# Reverse table: given (rank, dtype-class), produce the canonical lattice tag.
_TYPE_COMPOSE: dict[tuple[int, str], str] = {
    (0, "i"): "int", (0, "f"): "float", (0, "c"): "cx",
    (1, "i"): "vec_i", (1, "f"): "vec_f", (1, "c"): "vec_cx",
    (2, "f"): "mat_f", (2, "c"): "mat_cx",
    (3, "f"): "tensor3_f", (3, "c"): "tensor3_cx",
}


def promote_dtype(a: str, b: str) -> str:
    """Numpy-style dtype promotion: i �?f �?c, ``?`` is bottom."""
    order = {"i": 0, "f": 1, "c": 2, "?": -1}
    if a not in order or b not in order:
        return "?"
    return a if order[a] >= order[b] else b


def promote_rank(a: int, b: int) -> int:
    """Numpy broadcasting on rank: take the maximum (scalar broadcasts up)."""
    if a < 0 or b < 0:
        return -1
    return max(a, b)


def _decomp(t: str) -> tuple[int, str]:
    if t in _TYPE_DECOMP:
        return _TYPE_DECOMP[t]
    return (-1, "?")


def _compose(rank: int, dtype: str) -> str:
    if rank < 0 or dtype == "?":
        return TYPE_TOP
    out = _TYPE_COMPOSE.get((rank, dtype))
    return out if out is not None else TYPE_TOP


# ---------------------------------------------------------------------------
# combine_binary_type / combine_unary_type
# ---------------------------------------------------------------------------

# AST BinOp names that broadcast like numpy arithmetic.
_ARITH_OPS = frozenset({
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow",
    "+", "-", "*", "/", "//", "%", "**",
})

# Bitwise / logical �?bool/int closure.
_BITWISE_OPS = frozenset({
    "BitAnd", "BitOr", "BitXor", "LShift", "RShift",
    "&", "|", "^", "<<", ">>",
})

# Comparisons always produce bool (or vec_i for elementwise array compare).
_COMPARE_OPS = frozenset({
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE",
    "==", "!=", "<", "<=", ">", ">=",
})

# Matrix multiplication has its own rank algebra.
_MATMUL_OPS = frozenset({"MatMult", "@"})


def combine_binary_type(operator: str, lhs: str | None, rhs: str | None) -> str:
    """Lattice-aware result type of ``lhs OP rhs``.

    Falls back to ``unify(lhs, rhs)`` for ops with no special rule and to
    ``TYPE_TOP`` when either operand is unknown.  Handles numpy
    broadcasting in rank and dtype promotion in (int �?float �?complex).
    """
    if lhs is None or rhs is None:
        return TYPE_TOP
    if lhs == TYPE_VOID or rhs == TYPE_VOID:
        return TYPE_VOID
    if operator in _COMPARE_OPS:
        # Scalar compare �?bool; array compare �?vec_i (elementwise mask).
        la, _ = _decomp(lhs)
        ra, _ = _decomp(rhs)
        rank = promote_rank(la, ra)
        if rank <= 0:
            return "bool"
        return _compose(rank, "i") if rank > 0 else "bool"
    if operator in _BITWISE_OPS:
        # Bitwise on numerics �?preserve rank, force int dtype.
        la, _ = _decomp(lhs)
        ra, _ = _decomp(rhs)
        rank = promote_rank(la, ra)
        if rank < 0:
            return TYPE_TOP
        return _compose(rank, "i")
    if operator in _MATMUL_OPS:
        return _matmul_result(lhs, rhs)
    if operator in _ARITH_OPS:
        la, da = _decomp(lhs)
        ra, db = _decomp(rhs)
        rank = promote_rank(la, ra)
        dtype = promote_dtype(da, db)
        # Division of integers in Python yields float �?promote.
        if operator in {"Div", "/"} and dtype == "i":
            dtype = "f"
        if rank < 0 or dtype == "?":
            return unify(lhs, rhs)
        return _compose(rank, dtype)
    # Anything else: fall back to the lattice's least common supertype.
    return unify(lhs, rhs)


def _matmul_result(lhs: str, rhs: str) -> str:
    """Result type of ``lhs @ rhs`` per numpy's matmul rank algebra:

      mat @ mat �?mat,  mat @ vec �?vec,  vec @ mat �?vec,
      vec @ vec �?scalar (dot-product).
    """
    la, da = _decomp(lhs)
    ra, db = _decomp(rhs)
    if la < 0 or ra < 0 or da == "?" or db == "?":
        return TYPE_TOP
    dtype = promote_dtype(da, db)
    if la == 2 and ra == 2:
        return _compose(2, dtype)
    if la == 2 and ra == 1:
        return _compose(1, dtype)
    if la == 1 and ra == 2:
        return _compose(1, dtype)
    if la == 1 and ra == 1:
        return _compose(0, dtype)
    return TYPE_TOP


def combine_unary_type(operator: str, operand: str | None) -> str:
    """Lattice-aware result type of unary ops (``-x``, ``+x``, ``~x``, ``not x``)."""
    if operand is None or operand == TYPE_VOID:
        return TYPE_TOP
    if operator in {"Not", "not"}:
        # Logical not always coerces to bool / mask.
        rank, _ = _decomp(operand)
        if rank <= 0:
            return "bool"
        return _compose(rank, "i")
    if operator in {"Invert", "~"}:
        rank, _ = _decomp(operand)
        if rank < 0:
            return TYPE_TOP
        return _compose(rank, "i")
    if operator in {"USub", "UAdd", "-", "+"}:
        return operand
    if operator in {"abs", "Abs"}:
        # |complex| �?real, |real| �?real, |int| �?int.
        rank, dtype = _decomp(operand)
        if dtype == "c":
            dtype = "f"
        if rank < 0 or dtype == "?":
            return TYPE_TOP
        return _compose(rank, dtype)
    return operand


# ---------------------------------------------------------------------------
# Callable return-type registry
# ---------------------------------------------------------------------------
#
# A callable can either be registered with a fixed return type, or with a
# rule callable ``(arg_types: list[str]) -> str``.  Fixed entries are
# stored as the type string; rule entries are stored as the callable.

_CALLABLE_RETURNS: dict[str, "str | callable"] = {}


def register_callable_return(qualified_name: str, ret: "str | callable") -> None:
    """Register a return-type rule for ``qualified_name`` (e.g. ``np.linalg.inv``)."""
    _CALLABLE_RETURNS[qualified_name] = ret


def callable_return_type(qualified_name: str | None) -> "str | callable | None":
    """Look up the registered return type / rule for ``qualified_name``."""
    if qualified_name is None:
        return None
    return _CALLABLE_RETURNS.get(qualified_name)


def infer_call_return_type(
    qualified_name: str | None,
    arg_types: list[str] | None = None,
) -> str:
    """Return the inferred lattice type of a call.

    Falls back to ``TYPE_TOP`` (``"any"``) when the callee is unknown so
    downstream consumers know the type is unconstrained rather than
    misleading-them with ``"object"``.
    """
    rule = callable_return_type(qualified_name)
    if rule is None:
        return TYPE_TOP
    if isinstance(rule, str):
        return rule
    try:
        return rule(arg_types or [])
    except Exception:
        return TYPE_TOP


# ---------------------------------------------------------------------------
# Built-in numpy / scipy / Python registry
# ---------------------------------------------------------------------------
#
# Conventions:
#   * Functions returning a *new* matrix �?``mat_cx`` if any arg is complex,
#     else ``mat_f``.  Encoded as a lambda over arg_types.
#   * Functions guaranteed to return a vector �?similar real/complex
#     dispatch.
#   * Reductions returning a scalar �?real/complex scalar.

def _matlike(arg_types: list[str]) -> str:
    return "mat_cx" if any(is_complex(t) for t in arg_types) else "mat_f"


def _veclike(arg_types: list[str]) -> str:
    return "vec_cx" if any(is_complex(t) for t in arg_types) else "vec_f"


def _scalarlike(arg_types: list[str]) -> str:
    return "cx" if any(is_complex(t) for t in arg_types) else "float"


def _solve_like(arg_types: list[str]) -> str:
    # np.linalg.solve(A, b): output rank == rank(b).
    if len(arg_types) >= 2:
        rb, db = _decomp(arg_types[1])
        ra, da = _decomp(arg_types[0])
        if rb < 0:
            return _veclike(arg_types)
        dtype = promote_dtype(da, db)
        return _compose(rb, dtype)
    return _veclike(arg_types)


def _transpose_like(arg_types: list[str]) -> str:
    if not arg_types:
        return TYPE_TOP
    return arg_types[0]  # transpose preserves rank/dtype.


def _diag_like(arg_types: list[str]) -> str:
    # np.diag of vector �?matrix; of matrix �?vector.
    if not arg_types:
        return TYPE_TOP
    rank, dtype = _decomp(arg_types[0])
    if rank == 1:
        return _compose(2, dtype)
    if rank == 2:
        return _compose(1, dtype)
    return TYPE_TOP


_BUILTIN_REGISTRY: dict[str, "str | callable"] = {
    # Constructors / shape factories
    "np.eye":         "mat_f",
    "np.identity":    "mat_f",
    "np.zeros":       "vec_f",        # rank not knowable from name; default to vec
    "np.zeros_like":  _transpose_like,
    "np.ones":        "vec_f",
    "np.ones_like":   _transpose_like,
    "np.empty":       "vec_f",
    "np.empty_like":  _transpose_like,
    "np.full":        "vec_f",
    "np.array":       _transpose_like,
    "np.asarray":     _transpose_like,
    "np.copy":        _transpose_like,

    # Linear algebra (matrix-shaped outputs)
    "np.linalg.inv":   "mat_cx",
    "np.linalg.pinv":  "mat_cx",
    "np.linalg.cholesky": "mat_cx",
    "np.linalg.qr":    "mat_decomp",
    "np.linalg.svd":   "mat_decomp",
    "np.linalg.eig":   "mat_decomp",
    "np.linalg.eigh":  "mat_decomp",
    "np.linalg.solve": _solve_like,
    "np.linalg.lstsq": _solve_like,

    # Linear algebra (scalar / vector outputs)
    "np.linalg.norm":  "float",
    "np.linalg.det":   _scalarlike,
    "np.dot":          _veclike,    # default: rank-shrinks like matmul
    "np.matmul":       _veclike,
    "np.inner":        _scalarlike,
    "np.outer":        _matlike,
    "np.cross":        _veclike,
    "np.kron":         _matlike,
    "np.trace":        _scalarlike,

    # Transpose / reshape
    "np.transpose":    _transpose_like,
    "np.conj":         _transpose_like,
    "np.conjugate":    _transpose_like,
    "np.real":         lambda ts: _compose(_decomp(ts[0])[0] if ts else 0, "f"),
    "np.imag":         lambda ts: _compose(_decomp(ts[0])[0] if ts else 0, "f"),
    "np.abs":          lambda ts: combine_unary_type("abs", ts[0] if ts else None),
    "np.absolute":     lambda ts: combine_unary_type("abs", ts[0] if ts else None),

    # Element-wise unary numerics �?preserve type.
    "np.sqrt":         _transpose_like,
    "np.exp":          _transpose_like,
    "np.log":          _transpose_like,
    "np.log2":         _transpose_like,
    "np.log10":        _transpose_like,
    "np.sin":          _transpose_like,
    "np.cos":          _transpose_like,
    "np.tan":          _transpose_like,
    "np.tanh":         _transpose_like,
    "np.exp2":         _transpose_like,

    # Reductions �?scalar.
    "np.sum":          _scalarlike,
    "np.mean":         _scalarlike,
    "np.var":          "float",
    "np.std":          "float",
    "np.min":          _scalarlike,
    "np.max":          _scalarlike,
    "np.argmin":       "int",
    "np.argmax":       "int",
    "np.amin":         _scalarlike,
    "np.amax":         _scalarlike,
    "np.prod":         _scalarlike,
    "np.median":       "float",
    "np.count_nonzero":"int",

    # Diagonal / structural.
    "np.diag":         _diag_like,
    "np.diagflat":     "mat_f",
    "np.tril":         _transpose_like,
    "np.triu":         _transpose_like,
    "np.flip":         _transpose_like,
    "np.flatten":      _veclike,
    "np.squeeze":      _transpose_like,
    "np.expand_dims":  _transpose_like,
    "np.reshape":      _transpose_like,

    # Comparison / selection.
    "np.where":        _transpose_like,
    "np.clip":         _transpose_like,

    # Decomposition utilities (scipy commonly used)
    "scipy.linalg.solve":     _solve_like,
    "scipy.linalg.inv":       "mat_cx",
    "scipy.linalg.pinv":      "mat_cx",
    "scipy.linalg.cholesky":  "mat_cx",
    "scipy.linalg.qr":        "mat_decomp",
    "scipy.linalg.svd":       "mat_decomp",
    "scipy.linalg.lu":        "mat_decomp",
    "scipy.linalg.norm":      "float",

    # Python builtins commonly seen.
    "len":             "int",
    "abs":             lambda ts: combine_unary_type("abs", ts[0] if ts else None),
    "int":             "int",
    "float":           "float",
    "complex":         "cx",
    "bool":            "bool",
    "min":             _scalarlike,
    "max":             _scalarlike,
    "sum":             _scalarlike,
    "range":           "list<int>",
    "list":            "list<any>",
    "tuple":           "tuple<any>",
    "dict":            "dict<any>",
    "set":             "open_set",

    # Constellation / detector helpers seen in the pool.
    "qam16_constellation": "vec_cx",
    "qpsk_constellation":  "vec_cx",
}

for _name, _ret in _BUILTIN_REGISTRY.items():
    register_callable_return(_name, _ret)
