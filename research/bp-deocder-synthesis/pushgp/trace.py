"""Provenance-tracing VM: runs a Push program with concrete values
*and* a parallel tag-stack per type that tracks how the value at each
position was produced.  After execution, the tag at the top of the
float stack is the live symbolic expression that produced the program's
output (everything not on this provenance graph is dead code).

Implementation:
  * Wrap each TypedStack with `TracingTypedStack` that mirrors every
    push / pop / dup / swap / rot / yank / shove / clear onto a parallel
    `TaggedStack` of strings.
  * Before each instruction, the VM sets `cur_op`.  Handler pops collect
    tags into `consumed` (in pop order).  When the handler pushes, a new
    tag `cur_op(<consumed_in_reverse>)` is built.  Handlers that pop
    multiple types still combine tags in pop-order across types — this
    is the actual data-flow.
  * Control-flow handlers (Exec.*) execute their nested code_block; the
    body's pushes/pops are recorded normally, so the live expression
    naturally reflects only the instructions whose output reaches the
    final float-stack top.

Output format:
  Atomic tag: "Env.GetDeg"
  Composed:   "Float.Add(Env.GetDeg, Float.EvoConst3)"
  ?-tags appear when a pop underflowed (handler had nothing to consume).
"""
from __future__ import annotations

from typing import Any, List, Optional

from .program import Instruction
from .stacks import TypedStack, VMState
from .types import Type
from .vm import VM


class TaggedStack:
    """Parallel string-stack mirroring a TypedStack."""
    def __init__(self) -> None:
        self._tags: List[str] = []

    def push(self, t: str) -> None:
        self._tags.append(t)

    def pop(self) -> str:
        if not self._tags:
            return "?"
        return self._tags.pop()

    def peek(self, off: int = 0) -> str:
        if 0 <= off < len(self._tags):
            return self._tags[-1 - off]
        return "?"

    def dup(self) -> None:
        if self._tags:
            self._tags.append(self._tags[-1])

    def swap(self) -> None:
        if len(self._tags) >= 2:
            self._tags[-1], self._tags[-2] = self._tags[-2], self._tags[-1]

    def rot(self) -> None:
        if len(self._tags) >= 3:
            a = self._tags.pop(-3)
            self._tags.append(a)

    def yank(self, n: int) -> None:
        if 0 < n < len(self._tags):
            v = self._tags.pop(-1 - n)
            self._tags.append(v)

    def shove(self, n: int) -> None:
        if 0 < n < len(self._tags):
            v = self._tags.pop()
            self._tags.insert(len(self._tags) - n + 1, v)

    def clear(self) -> None:
        self._tags.clear()

    def depth(self) -> int:
        return len(self._tags)


class _TracingTypedStack:
    """Proxy around a TypedStack that mirrors stack ops on a TaggedStack.

    The underlying stack is mutated normally; the tag stack is mirrored.
    Used to instrument the VM without touching any handler code.
    """
    __slots__ = ("_real", "_tags", "_tracer")

    def __init__(self, real: TypedStack, tags: TaggedStack, tracer: "Tracer") -> None:
        self._real = real
        self._tags = tags
        self._tracer = tracer

    # ----- TypedStack interface (delegating + tag mirror) -----
    @property
    def type(self):
        return self._real.type

    @property
    def max_depth(self):
        return self._real.max_depth

    def depth(self) -> int:
        return self._real.depth()

    def is_empty(self) -> bool:
        return self._real.is_empty()

    def clear(self) -> None:
        self._real.clear()
        self._tags.clear()

    def as_list(self):
        return self._real.as_list()

    def push(self, value) -> None:
        # Build a tag for this push from currently-consumed tags + cur_op.
        tag = self._tracer.make_tag()
        self._real.push(value)
        self._tags.push(tag)

    def pop(self):
        v = self._real.pop()
        if v is None:
            return None
        t = self._tags.pop()
        self._tracer.consumed.append(t)
        return v

    def peek(self, offset: int = 0):
        return self._real.peek(offset)

    def dup(self) -> None:
        self._real.dup()
        self._tags.dup()

    def swap(self) -> None:
        self._real.swap()
        self._tags.swap()

    def rot(self) -> None:
        self._real.rot()
        self._tags.rot()

    def yank(self, n: int) -> None:
        self._real.yank(n)
        self._tags.yank(n)

    def shove(self, n: int) -> None:
        self._real.shove(n)
        self._tags.shove(n)


class Tracer:
    """Holds the cur_op + consumed buffer that wrapper stacks reference."""
    def __init__(self) -> None:
        self.cur_op: str = ""
        self.consumed: List[str] = []
        # If a handler pushes multiple times, only the FIRST push consumes
        # the buffered tags; subsequent pushes get atomic cur_op tag.
        self._first_push_done = False

    def reset_for_op(self, op: str) -> None:
        self.cur_op = op
        self.consumed = []
        self._first_push_done = False

    def make_tag(self) -> str:
        if not self._first_push_done and self.consumed:
            args = ", ".join(reversed(self.consumed))
            self._first_push_done = True
            self.consumed = []
            return f"{self.cur_op}({args})"
        # Atomic: nothing consumed (e.g. constants, env getters)
        return self.cur_op


class TraceVM(VM):
    """VM subclass that records production lineage for stack values."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tracer = Tracer()
        # Build tag stacks and wrap typed stacks.
        self._tag_stacks = {t: TaggedStack() for t in Type}
        # Replace stacks on the VMState with wrappers.
        for t in Type:
            real = self.state.stack_for(t)
            wrapper = _TracingTypedStack(real, self._tag_stacks[t], self._tracer)
            self._set_stack(t, wrapper)

    def _set_stack(self, t: Type, wrapper) -> None:
        # Map Type enum to attribute name on VMState.
        name = {
            Type.FLOAT: "floats",
            Type.INT: "ints",
            Type.BOOL: "bools",
            Type.FLOATVEC: "fvecs",
            Type.INTVEC: "ivecs",
            Type.BOOLVEC: "bvecs",
            Type.FLOATMAT: "fmats",
        }[t]
        setattr(self.state, name, wrapper)

    def reset(self) -> None:
        # Reset underlying real stacks via wrappers (which clear tags too).
        for t in Type:
            self.state.stack_for(t).clear()
        self.state.memory[:] = 0.0
        self.state.step_count = 0
        self.state.flop_count = 0
        self.state.fault = False
        self.state.fault_reason = ""
        self._recur_depth = 0
        self._tracer = Tracer()
        # Re-wrap (real stacks already wrapped, but tracer reference changed).
        for t in Type:
            wrap = self.state.stack_for(t)
            if isinstance(wrap, _TracingTypedStack):
                wrap._tracer = self._tracer
            else:  # pragma: no cover
                pass

    # Override _step to install cur_op for the wrapper before handler runs.
    def _step(self, ins: Instruction) -> None:
        self._tracer.reset_for_op(ins.name)
        super()._step(ins)

    def get_top_float_tag(self) -> Optional[str]:
        if self.state.floats.depth() == 0:
            return None
        return self._tag_stacks[Type.FLOAT].peek(0)


def trace_program(
    program: List[Instruction],
    *,
    ctx_channel_llr: float = 0.5,
    ctx_incoming=None,
    ctx_noise_var: float = 1.0,
    ctx_iter: int = 1,
    ctx_max_iter: int = 8,
    ctx_deg: int = 6,
    ctx_edge_index: int = 3,
    ctx_evo_constants=None,
    ctx_has_channel_llr: bool = True,
) -> dict:
    """Run a program in trace mode and return {'value': float|None,
    'live_expr': str|None, 'fault': str}."""
    import numpy as np
    if ctx_incoming is None:
        ctx_incoming = np.array([0.4, -0.3, 0.7, -0.2, 0.1, -0.5], dtype=np.float64)
    if ctx_evo_constants is None:
        ctx_evo_constants = np.ones(8, dtype=np.float64)

    vm = TraceVM()
    vm.state.ctx_channel_llr = ctx_channel_llr
    vm.state.ctx_incoming = ctx_incoming
    vm.state.ctx_noise_var = ctx_noise_var
    vm.state.ctx_iter = ctx_iter
    vm.state.ctx_max_iter = ctx_max_iter
    vm.state.ctx_deg = ctx_deg
    vm.state.ctx_edge_index = ctx_edge_index
    vm.state.ctx_evo_constants = ctx_evo_constants
    vm.state.ctx_has_channel_llr = ctx_has_channel_llr

    val = vm.run(program)
    expr = vm.get_top_float_tag()
    return {
        "value": val,
        "live_expr": expr,
        "fault": vm.state.fault_reason if vm.state.fault else "",
    }


__all__ = ["TraceVM", "TaggedStack", "trace_program"]
