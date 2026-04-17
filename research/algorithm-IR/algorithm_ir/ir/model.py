from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SourceSpan = tuple[int, int, int, int] | None


@dataclass
class Value:
    id: str
    name_hint: str | None = None
    type_hint: str | None = None
    source_span: SourceSpan = None
    def_op: str | None = None
    use_ops: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Op:
    id: str
    opcode: str
    inputs: list[str]
    outputs: list[str]
    block_id: str
    source_span: SourceSpan = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Block:
    id: str
    op_ids: list[str] = field(default_factory=list)
    preds: list[str] = field(default_factory=list)
    succs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionIR:
    id: str
    name: str
    arg_values: list[str]
    return_values: list[str]
    values: dict[str, Value]
    ops: dict[str, Op]
    blocks: dict[str, Block]
    entry_block: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleIR:
    functions: dict[str, FunctionIR]
    global_values: dict[str, Value]
    attrs: dict[str, Any] = field(default_factory=dict)

