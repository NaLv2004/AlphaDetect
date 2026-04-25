"""SlotContract — typed multi-port description of a slot region.

Phase H+4 §4.1. Built from BoundaryContract + SlotSpec when available;
otherwise inferred from the resolved region (entry/exit values).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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
