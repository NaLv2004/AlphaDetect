"""Backward-compatible facade over the lattice in ``algorithm_ir.ir.type_lattice``.

This module preserves the historical ``TypeInfo`` dataclass API used by
the lifter and the xDSL serialization bridge, but every operation now
delegates to the canonical lattice.  As a result ``Value.type_hint``
fields produced by the lifter become lattice-conformant strings
(``vec_f``, ``mat_cx``, ``cx``, 鈥? instead of the previous
Python-class-name garbage (``ndarray``, ``function``, ``ufunc`` 鈥?.

The dataclass is kept as a one-field carrier (``kind``) so that existing
``to_dict()`` / ``type_info_from_dict()`` round-trips continue to work
without schema migration.  Composite information (``elem``, ``key``,
``value``) is encoded into the ``kind`` string itself using the lattice's
composite grammar (``list<int>``, ``tuple<vec_f,float>``, ``dict<float>``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from algorithm_ir.ir.type_lattice import (
    combine_binary_type,
    combine_unary_type,
    infer_call_return_type,
    infer_value_type,
    is_subtype,
    parse_composite,
    unify,
)


@dataclass(frozen=True)
class TypeInfo:
    """Single-field carrier wrapping a lattice type string.

    The legacy fields ``elem`` / ``key`` / ``value`` / ``arity`` /
    ``shape`` are preserved to avoid breaking callers that read them,
    but they are derived on demand from the lattice ``kind`` string.
    """

    kind: str
    elem: "TypeInfo | None" = None
    key: "TypeInfo | None" = None
    value: "TypeInfo | None" = None
    arity: int | None = None
    shape: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        # Round-trip-safe serialization: the lattice ``kind`` string
        # already encodes any composite structure, so we only need to
        # persist that.  Legacy callers reading ``elem`` / ``key`` /
        # ``value`` from the dict will not find them, but
        # ``type_info_from_dict`` reconstructs them lazily via
        # ``parse_composite``.
        payload: dict[str, Any] = {"kind": self.kind}
        if self.shape is not None:
            payload["shape"] = list(self.shape)
        return payload

    def __post_init__(self) -> None:
        # Hydrate legacy attribute fields from the lattice string so
        # callers reading ``info.elem`` etc. observe consistent values.
        if self.elem is None and self.key is None and self.value is None:
            pc = parse_composite(self.kind)
            if pc is not None:
                head, parts = pc
                if head == "tuple" and parts:
                    object.__setattr__(self, "elem", TypeInfo(parts[0]))
                    object.__setattr__(self, "arity", len(parts))
                elif head == "list" and parts:
                    object.__setattr__(self, "elem", TypeInfo(parts[0]))
                elif head == "dict" and parts:
                    object.__setattr__(self, "value", TypeInfo(parts[0]))


def type_info_for_python_value(value: Any) -> TypeInfo:
    """Lattice-aware introspection of a Python runtime value."""
    return TypeInfo(infer_value_type(value))


def unify_type_infos(*infos: TypeInfo | None) -> TypeInfo:
    """Least common supertype across a list of TypeInfos (skipping ``None``)."""
    cleaned = [info.kind for info in infos if info is not None]
    if not cleaned:
        return TypeInfo("object")
    out = cleaned[0]
    for k in cleaned[1:]:
        out = unify(out, k)
    return TypeInfo(out)


def combine_binary_type_info(
    operator: str, lhs: TypeInfo | None, rhs: TypeInfo | None,
) -> TypeInfo:
    """Lattice-aware result type of a binary AST op."""
    return TypeInfo(combine_binary_type(
        operator,
        lhs.kind if lhs is not None else None,
        rhs.kind if rhs is not None else None,
    ))


def combine_unary_type_info(operator: str, operand: TypeInfo | None) -> TypeInfo:
    """Lattice-aware result type of a unary AST op."""
    return TypeInfo(combine_unary_type(
        operator,
        operand.kind if operand is not None else None,
    ))


def infer_call_type_info(
    qualified_name: str | None,
    arg_types: list[TypeInfo] | None = None,
) -> TypeInfo:
    """Lattice-aware result type of a known callable."""
    args = [a.kind for a in arg_types] if arg_types else []
    return TypeInfo(infer_call_return_type(qualified_name, args))


def type_hint_from_info(info: TypeInfo | None) -> str:
    """Project a TypeInfo onto its lattice string (used as ``Value.type_hint``)."""
    if info is None:
        return "object"
    return info.kind or "object"


def type_info_from_dict(payload: dict[str, Any]) -> TypeInfo:
    """Restore a TypeInfo from a previously serialised payload.

    Legacy payloads carrying ``elem`` / ``key`` / ``value`` sub-dicts are
    accepted and projected back onto a lattice composite string.
    """
    kind = payload.get("kind", "object")
    if "elem" in payload and kind in {"list", "tuple"}:
        elem_kind = type_info_from_dict(payload["elem"]).kind
        if kind == "tuple":
            n = int(payload.get("arity") or 1)
            kind = f"tuple<{','.join([elem_kind] * n)}>"
        else:
            kind = f"list<{elem_kind}>"
    elif "value" in payload and kind == "dict":
        val_kind = type_info_from_dict(payload["value"]).kind
        kind = f"dict<{val_kind}>"
    shape = tuple(payload["shape"]) if "shape" in payload else None
    return TypeInfo(kind=kind, shape=shape)


def is_subtype_info(a: TypeInfo | None, b: TypeInfo | None) -> bool:
    if a is None or b is None:
        return False
    return is_subtype(a.kind, b.kind)


__all__ = [
    "TypeInfo",
    "type_info_for_python_value",
    "unify_type_infos",
    "combine_binary_type_info",
    "combine_unary_type_info",
    "infer_call_type_info",
    "type_hint_from_info",
    "type_info_from_dict",
    "is_subtype_info",
]
