from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeValue:
    rid: str
    static_value_id: str
    py_obj_id: int | None
    created_by_event: str
    last_writer_event: str | None
    frame_id: str
    version: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeEvent:
    event_id: str
    static_op_id: str
    frame_id: str
    timestamp: int
    input_rids: list[str]
    output_rids: list[str]
    control_context: tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)

