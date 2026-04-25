"""SlotContract — typed multi-port description of a slot region.

Phase H+4 §4.1. Built from BoundaryContract + SlotSpec when available;
otherwise inferred from the resolved region (entry/exit values).

Phase H+5 R4: contract type vocabulary is delegated to
``algorithm_ir.ir.type_lattice`` (single source of truth for types).
The legacy short tokens used in ``ProgramSpec`` (e.g. "mat", "list_int",
"tuple", "object", "vec") are normalized to lattice tokens via
``_normalize_lattice_type``; unknown tokens fall back to ``TYPE_TOP``
("any") rather than being passed through as fake placeholders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from algorithm_ir.ir.type_lattice import (
    ALL_ATOMIC_TYPES,
    TYPE_TOP,
    is_dict_type,
    is_list_type,
    is_tuple_type,
    parse_composite,
    tuple_components,
)

if TYPE_CHECKING:
    from algorithm_ir.ir.model import FunctionIR
    from evolution.pool_types import SlotPopulation


@dataclass(frozen=True)
class TypedPort:
    name: str
    type: str
    role: Literal["data", "state", "control"] = "data"


@dataclass(frozen=True)
class SlotContract:
    slot_key: str                       # e.g. "lmmse.regularizer"
    short_name: str                     # e.g. "regularizer"
    input_ports: tuple[TypedPort, ...]
    output_ports: tuple[TypedPort, ...]
    state_ports: tuple[TypedPort, ...] = ()
    effects: frozenset[str] = field(default_factory=lambda: frozenset({"pure"}))
    complexity_cap: int = 64
    constants_budget: int = 8


# ---------------------------------------------------------------------------
# Type-spec normalization (R4)
# ---------------------------------------------------------------------------

# Aliases from legacy ProgramSpec short tokens to type_lattice vocabulary.
# Keys are the strings actually used in ``evolution/algorithm_pool.py``.
_SPEC_ALIAS: dict[str, str] = {
    "mat": "mat_cx",          # MIMO defaults to complex matrices
    "vec": "vec_cx",
    "list_int": "list<int>",
    "list_float": "list<float>",
    "list_cx": "list<cx>",
    "list": "list<any>",
    "tuple": "any",           # arity unknown — keep neutral
    "object": "any",
    "scalar": "float",
    "bool": "bool",
    "int": "int",
    "float": "float",
    "cx": "cx",
}


def _normalize_lattice_type(spec_type: str | None) -> str:
    """Map a ProgramSpec type string to a type_lattice token.

    The result is guaranteed to be either an entry of ``ALL_ATOMIC_TYPES``
    or a composite parsable by ``type_lattice.parse_composite``.
    Unknown tokens collapse to ``TYPE_TOP``.
    """
    if not isinstance(spec_type, str) or not spec_type.strip():
        return TYPE_TOP
    s = spec_type.strip()
    if s in ALL_ATOMIC_TYPES:
        return s
    # Composite (list<...>, tuple<...>, dict<...>) — accept if parseable.
    if is_list_type(s) or is_tuple_type(s) or is_dict_type(s) or parse_composite(s):
        return s
    if s in _SPEC_ALIAS:
        return _SPEC_ALIAS[s]
    return TYPE_TOP


def build_slot_contract(
    pop: "SlotPopulation",
    *,
    slot_key: str,
    complexity_cap: int = 64,
    constants_budget: int = 8,
) -> SlotContract:
    """Derive a SlotContract from the slot's ProgramSpec.

    R4: this is the production replacement for the legacy
    ``_make_contract_from_region`` placeholder that emitted
    ``TypedPort("in","any") → TypedPort("out","any")`` regardless of
    the slot's actual signature. Input ports come from
    ``spec.param_names``/``spec.param_types``; output ports come from
    ``spec.return_type`` (with tuple decomposition via
    ``type_lattice.tuple_components`` when applicable).
    """
    short = slot_key.split(".")[-1]
    spec = pop.spec
    # Inputs: zip names + types, falling back to positional names.
    names = list(getattr(spec, "param_names", []) or [])
    types = list(getattr(spec, "param_types", []) or [])
    n = max(len(names), len(types))
    in_ports: list[TypedPort] = []
    for i in range(n):
        nm = names[i] if i < len(names) else f"arg{i}"
        tp = _normalize_lattice_type(types[i] if i < len(types) else None)
        in_ports.append(TypedPort(nm, tp))

    # Outputs: tuple types decompose into one port per component.
    ret = _normalize_lattice_type(getattr(spec, "return_type", None))
    out_ports: list[TypedPort]
    if is_tuple_type(ret):
        comps = tuple_components(ret) or [TYPE_TOP]
        out_ports = [
            TypedPort(f"out{i}", _normalize_lattice_type(c))
            for i, c in enumerate(comps)
        ]
    else:
        out_ports = [TypedPort("out", ret)]

    return SlotContract(
        slot_key=slot_key,
        short_name=short,
        input_ports=tuple(in_ports),
        output_ports=tuple(out_ports),
        complexity_cap=complexity_cap,
        constants_budget=constants_budget,
    )
