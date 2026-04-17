from __future__ import annotations

from dataclasses import dataclass, field

from algorithm_ir.ir.model import FunctionIR, Op, Value
from algorithm_ir.runtime.frames import RuntimeFrame
from algorithm_ir.runtime.tracer import RuntimeEvent, RuntimeValue


@dataclass
class FactGraph:
    static_functions: dict[str, FunctionIR]
    static_ops: dict[str, Op]
    static_values: dict[str, Value]
    runtime_events: dict[str, RuntimeEvent]
    runtime_values: dict[str, RuntimeValue]
    runtime_frames: dict[str, RuntimeFrame]
    static_edges: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    dynamic_edges: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    alignment_edges: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

