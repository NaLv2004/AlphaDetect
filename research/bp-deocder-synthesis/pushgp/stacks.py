"""Typed stacks and VM state container.

`TypedStack` is a thin, depth-bounded LIFO container with the usual
Push-GP utility operations (pop, dup, swap, rot, yank, shove).  Every
operation is *forgiving*: underflow / out-of-range index is silently
treated as a no-op so randomly generated programs almost always run to
completion without raising.

`VMState` bundles one stack per `Type` plus the per-call execution
context (channel LLR, incoming-message vector, BP iteration index,
node degree, evolved constants, working memory cells).  The VM
(`pushgp.vm`) operates on a `VMState` instance; instructions are
implemented in `pushgp.instructions`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, List, Optional, TypeVar

import numpy as np

from .types import DEFAULT_MAX_DEPTHS, Type

T = TypeVar("T")


class TypedStack(Generic[T]):
    """Bounded LIFO stack for one Push type.

    All mutating operations clamp / no-op rather than raise, except for
    `push` which silently *drops the bottom* element when the stack is
    full so that random programs never crash from overflow.
    """

    __slots__ = ("_data", "_max_depth", "_type")

    def __init__(self, type_: Type, max_depth: Optional[int] = None) -> None:
        self._type = type_
        self._max_depth = (
            max_depth if max_depth is not None else DEFAULT_MAX_DEPTHS[type_]
        )
        self._data: List[T] = []

    # ------------------------------------------------------------------ basics
    @property
    def type(self) -> Type:
        return self._type

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def depth(self) -> int:
        return len(self._data)

    def is_empty(self) -> bool:
        return not self._data

    def clear(self) -> None:
        self._data.clear()

    def as_list(self) -> List[T]:
        """Return a *copy* of the underlying list (top of stack is last)."""
        return list(self._data)

    # ---------------------------------------------------------------- push/pop
    def push(self, value: T) -> None:
        if len(self._data) >= self._max_depth:
            # Drop bottom to keep the most recently produced values.
            del self._data[0]
        self._data.append(value)

    def pop(self) -> Optional[T]:
        if not self._data:
            return None
        return self._data.pop()

    def peek(self, offset: int = 0) -> Optional[T]:
        """Return the element `offset` from the top (0 = top), or None."""
        if offset < 0 or offset >= len(self._data):
            return None
        return self._data[-1 - offset]

    # --------------------------------------------------------- forth-style ops
    def dup(self) -> None:
        if not self._data:
            return
        # For ndarrays we copy to avoid aliasing surprises in evolved programs.
        top = self._data[-1]
        if isinstance(top, np.ndarray):
            self.push(top.copy())  # type: ignore[arg-type]
        else:
            self.push(top)

    def swap(self) -> None:
        if len(self._data) < 2:
            return
        self._data[-1], self._data[-2] = self._data[-2], self._data[-1]

    def rot(self) -> None:
        """Forth ROT: (a b c -- b c a)."""
        if len(self._data) < 3:
            return
        a = self._data.pop(-3)
        self._data.append(a)

    def yank(self, n: int) -> None:
        """Bring the element at depth n (0 = top) to the top."""
        if n <= 0 or n >= len(self._data):
            return
        v = self._data.pop(-1 - n)
        self._data.append(v)

    def shove(self, n: int) -> None:
        """Push current top down so it lands at depth n (0 = stays on top)."""
        if n <= 0 or n >= len(self._data):
            return
        v = self._data.pop()
        self._data.insert(len(self._data) - n + 1, v)

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TypedStack({self._type.value}, depth={len(self._data)}/{self._max_depth})"


# ---------------------------------------------------------------------- VMState

@dataclass
class VMState:
    """Container for all stacks plus per-invocation execution context.

    The VM (see `pushgp.vm`) constructs / resets one of these for every
    edge-message update.  Fields named with `ctx_` are inputs supplied by
    the caller (the LDPC adapter) and are exposed to evolved programs
    only through `Env.*` instructions.
    """

    # --- typed stacks --------------------------------------------------------
    floats: TypedStack[float] = field(default_factory=lambda: TypedStack(Type.FLOAT))
    ints: TypedStack[int] = field(default_factory=lambda: TypedStack(Type.INT))
    bools: TypedStack[bool] = field(default_factory=lambda: TypedStack(Type.BOOL))
    fvecs: TypedStack[np.ndarray] = field(
        default_factory=lambda: TypedStack(Type.FLOATVEC)
    )
    bvecs: TypedStack[np.ndarray] = field(
        default_factory=lambda: TypedStack(Type.BOOLVEC)
    )
    ivecs: TypedStack[np.ndarray] = field(
        default_factory=lambda: TypedStack(Type.INTVEC)
    )
    fmats: TypedStack[np.ndarray] = field(
        default_factory=lambda: TypedStack(Type.FLOATMAT)
    )

    # --- read-only execution context (set by adapter, not by program) -------
    ctx_channel_llr: float = 0.0
    ctx_incoming: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    ctx_noise_var: float = 1.0
    ctx_iter: int = 0
    ctx_max_iter: int = 25
    ctx_deg: int = 0
    ctx_edge_index: int = 0
    ctx_code_rate: float = 0.5
    ctx_evo_constants: np.ndarray = field(
        default_factory=lambda: np.ones(8, dtype=np.float64)
    )
    ctx_has_channel_llr: bool = True  # False for C2V context

    # --- working memory (16 cells, persists for one program invocation) -----
    memory: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float64))

    # --- bookkeeping --------------------------------------------------------
    step_count: int = 0
    flop_count: int = 0
    fault: bool = False
    fault_reason: str = ""

    # --------------------------------------------------------------- helpers
    def stack_for(self, t: Type) -> TypedStack[Any]:
        return _STACK_GETTERS[t](self)

    def reset_stacks(self) -> None:
        for t in Type:
            self.stack_for(t).clear()
        self.memory[:] = 0.0
        self.step_count = 0
        self.flop_count = 0
        self.fault = False
        self.fault_reason = ""


_STACK_GETTERS = {
    Type.FLOAT: lambda s: s.floats,
    Type.INT: lambda s: s.ints,
    Type.BOOL: lambda s: s.bools,
    Type.FLOATVEC: lambda s: s.fvecs,
    Type.BOOLVEC: lambda s: s.bvecs,
    Type.INTVEC: lambda s: s.ivecs,
    Type.FLOATMAT: lambda s: s.fmats,
}


__all__ = ["TypedStack", "VMState"]
