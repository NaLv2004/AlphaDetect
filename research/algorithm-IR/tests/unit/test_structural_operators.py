"""R5: 5 structural typed-GP operators.

Each operator is exercised on a small synthesized FunctionIR built from
the frontend so that validate_function_ir + canonical_ir_hash agree on
"this is a valid IR with a non-trivial structure".
"""
from __future__ import annotations

import numpy as np
import pytest

from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.ir.validator import validate_function_ir

from evolution.gp.canonical_hash import canonical_ir_hash
from evolution.gp.contract import SlotContract, TypedPort
from evolution.gp.operators.base import (
    GPContext,
    OPERATOR_REGISTRY,
    run_operator_with_gates,
)


SAMPLE_SRC = """
def f(x: vec_cx, y: vec_cx):
    a = x + y
    b = a * x
    c = b - y
    d = c + x
    return d
"""


@pytest.fixture
def parent_ir():
    ir = compile_source_to_ir(SAMPLE_SRC, "f")
    assert validate_function_ir(ir) == []
    return ir


@pytest.fixture
def parent2_ir():
    src2 = """
def g(p: vec_cx, q: vec_cx):
    u = p * q
    v = u + p
    w = v - q
    return w
"""
    ir = compile_source_to_ir(src2, "g")
    assert validate_function_ir(ir) == []
    return ir


def _ctx(rng: np.random.Generator, slot_key: str = "test.slot") -> GPContext:
    contract = SlotContract(
        slot_key=slot_key,
        short_name=slot_key.split(".")[-1],
        input_ports=(TypedPort("x", "vec_cx"), TypedPort("y", "vec_cx")),
        output_ports=(TypedPort("out", "vec_cx"),),
        complexity_cap=1024,
    )
    return GPContext(contract=contract, region_op_ids=frozenset(), rng=rng)


def _try_operator(name: str, ctx: GPContext, parent_ir, parent2_ir=None,
                  attempts: int = 20):
    """Try the operator up to ``attempts`` times; return the first
    structurally accepted result, or None."""
    factory, _w, _cx = OPERATOR_REGISTRY[name]
    op = factory()
    parent_hash = canonical_ir_hash(parent_ir)
    last = None
    for _ in range(attempts):
        result = run_operator_with_gates(
            op, ctx, parent_ir, parent_hash, parent2_ir=parent2_ir,
        )
        last = result
        if result.accepted_structurally:
            return result
    return last


# ---------------------------------------------------------------------------
# 5 operators: at least one structurally-accepted proposal each
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", [
    "mut_insert_typed",
    "mut_delete_typed",
    "mut_subtree_replace",
    "mut_primitive_inject",
])
def test_structural_operator_accepts(op_name, parent_ir):
    rng = np.random.default_rng(0)
    ctx = _ctx(rng)
    result = _try_operator(op_name, ctx, parent_ir)
    assert result is not None
    assert result.accepted_structurally, \
        f"{op_name} never accepted; last reason: {result.rejection_reason}"
    assert result.child_ir is not None
    # IR must validate.
    assert validate_function_ir(result.child_ir) == []
    # Canonical hash must differ.
    assert result.child_hash != canonical_ir_hash(parent_ir)


def test_cx_subtree_typed_accepts(parent_ir, parent2_ir):
    rng = np.random.default_rng(0)
    ctx = _ctx(rng)
    result = _try_operator("cx_subtree_typed", ctx, parent_ir,
                           parent2_ir=parent2_ir)
    assert result is not None
    assert result.accepted_structurally, \
        f"cx_subtree_typed never accepted; last reason: {result.rejection_reason}"
    assert validate_function_ir(result.child_ir) == []
    assert result.child_hash != canonical_ir_hash(parent_ir)


def test_all_five_operators_registered():
    expected = {
        "mut_insert_typed",
        "mut_delete_typed",
        "mut_subtree_replace",
        "cx_subtree_typed",
        "mut_primitive_inject",
    }
    assert expected.issubset(set(OPERATOR_REGISTRY.keys())), \
        f"missing R5 operators: {expected - set(OPERATOR_REGISTRY.keys())}"


def test_mut_delete_decreases_op_count(parent_ir):
    rng = np.random.default_rng(0)
    ctx = _ctx(rng)
    result = _try_operator("mut_delete_typed", ctx, parent_ir)
    assert result is not None and result.accepted_structurally
    assert len(result.child_ir.ops) < len(parent_ir.ops)


def test_mut_insert_increases_op_count(parent_ir):
    rng = np.random.default_rng(0)
    ctx = _ctx(rng)
    result = _try_operator("mut_insert_typed", ctx, parent_ir)
    assert result is not None and result.accepted_structurally
    assert len(result.child_ir.ops) > len(parent_ir.ops)
