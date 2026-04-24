"""Pytest unit tests for evolution.const_lifter (S1)."""
from __future__ import annotations

import ast
import textwrap

import pytest

from evolution.const_lifter import (
    EXEMPT_VALUES,
    LiftResult,
    is_idempotent,
    lift_source,
    make_const_slot_default_source,
)


SRC_KBEST = textwrap.dedent("""
def kbest_template(H, y):
    K = 16
    sigma2 = 0.001
    Nt = H.shape[1]
    return K * Nt
""")

SRC_LOOPS = textwrap.dedent("""
def stack_walker(graph):
    max_nodes = 2000
    i = 0
    while i < 8:
        i += 1
    return i
""")

SRC_DAMPING = textwrap.dedent("""
def damping_fixed(x_old, x_new):
    alpha = 0.5
    return alpha * x_old + (1.0 - alpha) * x_new
""")

SRC_NESTED_SLOT_CALL = textwrap.dedent("""
def fancy(H):
    return _slot_inner(42, 3.14)
""")


def test_lifts_int_and_float_literals():
    res = lift_source(SRC_KBEST)
    values = sorted(L.original_value for L in res.lifted)
    assert 16 in values
    assert 0.001 in values


def test_default_exempt_skips_trivial_values():
    res = lift_source(SRC_KBEST)
    for L in res.lifted:
        assert L.original_value not in EXEMPT_VALUES


def test_loop_bound_lift_under_aggressive_exempt():
    res = lift_source(SRC_LOOPS, exempt={0, 1, -1})
    roles = [L.role for L in res.lifted]
    assert "loop_bound" in roles
    bounds = [L.original_value for L in res.lifted if L.role == "loop_bound"]
    assert 8 in bounds


def test_loop_bound_skipped_when_lift_loops_false():
    res = lift_source(SRC_LOOPS, exempt={0, 1, -1}, lift_loops=False)
    roles = [L.role for L in res.lifted]
    assert "loop_bound" not in roles


def test_lift_is_idempotent():
    assert is_idempotent(SRC_KBEST)
    assert is_idempotent(SRC_LOOPS)
    once = lift_source(SRC_KBEST)
    twice = lift_source(once.new_source)
    assert twice.lifted == []


def test_does_not_recurse_into_existing_slot_calls():
    res = lift_source(SRC_NESTED_SLOT_CALL)
    # Constant 42 / 3.14 inside _slot_inner(...) must NOT be lifted
    assert all(L.original_value not in (42, 3.14) for L in res.lifted)


def test_rewritten_source_parses_back():
    res = lift_source(SRC_KBEST)
    ast.parse(res.new_source)  # must not raise


def test_make_const_slot_default_source_int():
    res = lift_source(SRC_KBEST)
    L = next(x for x in res.lifted if x.original_value == 16)
    src = make_const_slot_default_source(L)
    ns: dict = {}
    exec(src, ns)
    assert ns[L.slot_name]() == 16


def test_make_const_slot_default_source_float():
    res = lift_source(SRC_KBEST)
    L = next(x for x in res.lifted if x.original_value == 0.001)
    src = make_const_slot_default_source(L)
    ns: dict = {}
    exec(src, ns)
    assert ns[L.slot_name]() == pytest.approx(0.001)


def test_max_lifts_per_func_cap():
    # 30 distinct float literals, none in EXEMPT_VALUES.
    body_lines = "    " + "\n    ".join(
        f"x{i} = {0.123 + i}" for i in range(30)
    )
    src = "def big():\n" + body_lines + "\n    return x0\n"
    res = lift_source(src)
    assert len(res.lifted) <= 16


def test_diagnostic_string_contains_count():
    res = lift_source(SRC_KBEST)
    assert str(len(res.lifted)) in res.diagnostic
