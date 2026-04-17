from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Projection:
    proj_id: str
    region_id: str
    family: str
    node_set: list[str]
    edge_set: list[tuple[str, str]]
    evidence: dict[str, Any]
    interface: dict[str, Any]
    score: float
    attrs: dict[str, Any] = field(default_factory=dict)

