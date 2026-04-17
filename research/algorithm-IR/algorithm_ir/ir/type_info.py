from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TypeInfo:
    kind: str
    elem: "TypeInfo | None" = None
    key: "TypeInfo | None" = None
    value: "TypeInfo | None" = None
    arity: int | None = None
    shape: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.elem is not None:
            payload["elem"] = self.elem.to_dict()
        if self.key is not None:
            payload["key"] = self.key.to_dict()
        if self.value is not None:
            payload["value"] = self.value.to_dict()
        if self.arity is not None:
            payload["arity"] = self.arity
        if self.shape is not None:
            payload["shape"] = list(self.shape)
        return payload


def type_info_for_python_value(value: Any) -> TypeInfo:
    if value is None:
        return TypeInfo("none")
    if isinstance(value, bool):
        return TypeInfo("bool")
    if isinstance(value, int):
        return TypeInfo("int")
    if isinstance(value, float):
        return TypeInfo("float")
    if isinstance(value, complex):
        return TypeInfo("complex")
    if isinstance(value, str):
        return TypeInfo("str")
    if isinstance(value, tuple):
        elem = _unify_many(type_info_for_python_value(item) for item in value)
        return TypeInfo("tuple", elem=elem, arity=len(value), shape=(len(value),))
    if isinstance(value, list):
        elem = _unify_many(type_info_for_python_value(item) for item in value)
        return TypeInfo("list", elem=elem, shape=(len(value),))
    if isinstance(value, dict):
        key = _unify_many(type_info_for_python_value(item) for item in value.keys())
        val = _unify_many(type_info_for_python_value(item) for item in value.values())
        return TypeInfo("dict", key=key, value=val, shape=(len(value),))
    return TypeInfo(type(value).__name__)


def unify_type_infos(*infos: TypeInfo | None) -> TypeInfo:
    cleaned = [info for info in infos if info is not None]
    if not cleaned:
        return TypeInfo("object")
    if len(cleaned) == 1:
        return cleaned[0]
    if all(info.kind == cleaned[0].kind for info in cleaned):
        exemplar = cleaned[0]
        if exemplar.kind in {"list", "tuple"}:
            return TypeInfo(
                exemplar.kind,
                elem=unify_type_infos(*(info.elem for info in cleaned)),
                arity=exemplar.arity,
                shape=exemplar.shape,
            )
        if exemplar.kind == "dict":
            return TypeInfo(
                "dict",
                key=unify_type_infos(*(info.key for info in cleaned)),
                value=unify_type_infos(*(info.value for info in cleaned)),
                shape=exemplar.shape,
            )
        return exemplar

    kinds = {info.kind for info in cleaned}
    if kinds <= {"bool", "int"}:
        return TypeInfo("int")
    if kinds <= {"bool", "int", "float"}:
        return TypeInfo("float")
    if kinds <= {"bool", "int", "float", "complex"}:
        return TypeInfo("complex")
    return TypeInfo("object")


def combine_binary_type_info(operator: str, lhs: TypeInfo | None, rhs: TypeInfo | None) -> TypeInfo:
    if operator in {"Add", "Sub", "Mult", "Div", "FloorDiv", "Mod"}:
        return unify_type_infos(lhs, rhs)
    return TypeInfo("object")


def type_hint_from_info(info: TypeInfo | None) -> str:
    if info is None:
        return "object"
    if info.kind in {"bool", "int", "float", "complex", "str", "list", "dict", "tuple"}:
        return info.kind
    return info.kind or "object"


def type_info_from_dict(payload: dict[str, Any]) -> TypeInfo:
    elem = type_info_from_dict(payload["elem"]) if "elem" in payload else None
    key = type_info_from_dict(payload["key"]) if "key" in payload else None
    value = type_info_from_dict(payload["value"]) if "value" in payload else None
    shape = tuple(payload["shape"]) if "shape" in payload else None
    return TypeInfo(
        kind=payload["kind"],
        elem=elem,
        key=key,
        value=value,
        arity=payload.get("arity"),
        shape=shape,
    )


def _unify_many(infos) -> TypeInfo:
    return unify_type_infos(*list(infos))
