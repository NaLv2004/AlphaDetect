"""Type lattice for type-aware GP search-space pruning.

Per code_review.md §4.1.

The lattice is a finite set of typed labels with three core operations:

    is_subtype(a, b)        — membership check used during generalization
    unify(a, b)             — least common supertype for crossover compatibility
    available_ops_for_type(t) — opcode names (in our `algorithm_ir.ir`
                                 vocabulary) that can produce values of
                                 type ``t``
    default_value(t)         — a concrete fallback value matching ``t``

Type strings are deliberately kept as small atomic identifiers so they
can live inside ``Value.attrs["_type"]`` without schema changes. The
lattice is intentionally **not a full Hindley–Milner system** — its
sole purpose is to prune the search space of GP operators.

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
