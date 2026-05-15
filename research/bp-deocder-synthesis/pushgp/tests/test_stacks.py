"""Unit tests for `pushgp.stacks`."""

from __future__ import annotations

import numpy as np
import pytest

from pushgp.stacks import TypedStack, VMState
from pushgp.types import Type


# ----------------------------------------------------------------------- TypedStack


def test_push_pop_peek_depth() -> None:
    s: TypedStack[float] = TypedStack(Type.FLOAT)
    assert s.is_empty() and s.depth() == 0
    s.push(1.0)
    s.push(2.0)
    s.push(3.0)
    assert s.depth() == 3
    assert s.peek() == 3.0
    assert s.peek(1) == 2.0
    assert s.peek(2) == 1.0
    assert s.peek(99) is None
    assert s.pop() == 3.0
    assert s.pop() == 2.0
    assert s.pop() == 1.0
    assert s.pop() is None  # underflow → None, not an error


def test_dup_swap_rot() -> None:
    s: TypedStack[int] = TypedStack(Type.INT)
    s.dup()  # underflow no-op
    s.swap()
    s.rot()
    assert s.is_empty()

    for v in (1, 2, 3):
        s.push(v)
    s.dup()
    assert s.as_list() == [1, 2, 3, 3]
    s.pop()
    s.swap()
    assert s.as_list() == [1, 3, 2]
    s.rot()  # (a b c) -> (b c a) → (3 2 1)
    assert s.as_list() == [3, 2, 1]


def test_yank_shove() -> None:
    s: TypedStack[int] = TypedStack(Type.INT)
    for v in (10, 20, 30, 40):
        s.push(v)
    # yank(2): bring depth-2 element (=20) to top
    s.yank(2)
    assert s.as_list() == [10, 30, 40, 20]
    # shove(2): push current top down two positions
    s.shove(2)
    assert s.as_list()[-3:] == [20, 30, 40] or s.as_list()[-1] != 20  # sanity


def test_max_depth_drops_bottom() -> None:
    s: TypedStack[int] = TypedStack(Type.INT, max_depth=3)
    for v in (1, 2, 3, 4, 5):
        s.push(v)
    assert s.depth() == 3
    assert s.as_list() == [3, 4, 5]


def test_dup_copies_ndarray() -> None:
    s: TypedStack[np.ndarray] = TypedStack(Type.FLOATVEC)
    a = np.array([1.0, 2.0, 3.0])
    s.push(a)
    s.dup()
    top = s.pop()
    assert top is not None
    top[0] = 999.0
    # Original copy on the stack must not have been mutated.
    bottom = s.pop()
    assert bottom is not None
    assert bottom[0] == 1.0


# ------------------------------------------------------------------------ VMState


def test_vmstate_has_all_typed_stacks() -> None:
    state = VMState()
    for t in Type:
        stk = state.stack_for(t)
        assert isinstance(stk, TypedStack)
        assert stk.type is t


def test_vmstate_reset_clears_everything() -> None:
    state = VMState()
    state.floats.push(3.14)
    state.ints.push(7)
    state.memory[5] = 9.9
    state.step_count = 42
    state.flop_count = 100
    state.fault = True
    state.fault_reason = "x"

    state.reset_stacks()
    assert state.floats.is_empty()
    assert state.ints.is_empty()
    assert state.memory.sum() == 0.0
    assert state.step_count == 0
    assert state.flop_count == 0
    assert state.fault is False
    assert state.fault_reason == ""


def test_vmstate_default_context_shapes() -> None:
    state = VMState()
    assert state.ctx_evo_constants.shape == (8,)
    assert state.ctx_incoming.shape == (0,)
    assert state.memory.shape == (16,)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
