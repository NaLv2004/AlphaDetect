"""Push-GP virtual machine.

The VM holds a `VMState` and dispatches `Instruction`s to handlers
registered in `pushgp.instructions.HANDLERS`.  It enforces three hard
resource budgets (steps, FLOPs, recursion depth) and a wall-clock
optional cap.  When any budget is exceeded the VM marks itself aborted
and stops executing further instructions; in-progress code blocks
unwind cleanly because every control-flow handler checks `vm.aborted()`
inside its loop.

Public entry points:

* `vm.run(program)`  — execute a top-level program; returns the float
  on top of the float stack on success or `None` on fault / empty
  result.

* `vm.execute_block(block)` — used by control-flow handlers to recurse
  into nested code blocks.  Not intended to be called externally.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from .instructions import HANDLERS
from .program import Instruction
from .stacks import VMState


class VM:
    """Stack-based interpreter for evolved Push-GP programs."""

    DEFAULT_STEP_MAX = 2000
    DEFAULT_FLOP_MAX = 50_000
    DEFAULT_RECUR_MAX = 32  # nested code-block depth
    DEFAULT_TIME_MAX_S: Optional[float] = None  # disabled by default

    def __init__(
        self,
        step_max: int = DEFAULT_STEP_MAX,
        flop_max: int = DEFAULT_FLOP_MAX,
        recur_max: int = DEFAULT_RECUR_MAX,
        time_max_s: Optional[float] = DEFAULT_TIME_MAX_S,
    ) -> None:
        self.state = VMState()
        self.step_max = step_max
        self.flop_max = flop_max
        self.recur_max = recur_max
        self.time_max_s = time_max_s

        self._recur_depth = 0
        self._t_start = 0.0

    # --------------------------------------------------------------- public API
    def reset(self) -> None:
        self.state.reset_stacks()
        self._recur_depth = 0
        self._t_start = 0.0

    def aborted(self) -> bool:
        return self.state.fault

    def charge_flops(self, n: int) -> None:
        self.state.flop_count += n
        if self.state.flop_count > self.flop_max:
            self._abort("flop_max exceeded")

    def run(self, program: List[Instruction]) -> Optional[float]:
        """Execute `program` from a clean state. Returns the top of the
        float stack on success, or None if a fault occurred or the float
        stack is empty / non-finite."""
        self._t_start = time.perf_counter()
        try:
            self.execute_block(program)
        except _HardwareFault:
            return None
        if self.state.fault:
            return None
        top = self.state.floats.peek()
        if top is None:
            return None
        try:
            v = float(top)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(v):
            return None
        return v

    def execute_block(self, block: List[Instruction]) -> None:
        if self._recur_depth >= self.recur_max:
            self._abort("recur_max exceeded")
            return
        self._recur_depth += 1
        try:
            for ins in block:
                if self.state.fault:
                    return
                self._step(ins)
        finally:
            self._recur_depth -= 1

    # ------------------------------------------------------------- internals
    def _step(self, ins: Instruction) -> None:
        self.state.step_count += 1
        if self.state.step_count > self.step_max:
            self._abort("step_max exceeded")
            return
        if self.time_max_s is not None and (
            time.perf_counter() - self._t_start
        ) > self.time_max_s:
            self._abort("time_max_s exceeded")
            return
        handler = HANDLERS.get(ins.name)
        if handler is None:
            # Unknown instruction names are silently ignored for forward-compat.
            return
        try:
            handler(self, ins)
        except Exception as exc:  # never let a buggy handler crash a fitness eval
            self._abort(f"handler exception: {type(exc).__name__}: {exc}")

    def _abort(self, reason: str) -> None:
        self.state.fault = True
        if not self.state.fault_reason:
            self.state.fault_reason = reason


class _HardwareFault(Exception):
    """Reserved for future hard aborts (currently unused; soft `fault`
    flag is checked at every step)."""


__all__ = ["VM"]
