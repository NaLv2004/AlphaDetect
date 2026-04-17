from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeFrame:
    frame_id: str
    function_id: str
    parent_frame_id: str | None
    callsite_event_id: str | None
    locals: dict[str, str] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

