"""Phase H+4 S2.3-S2.5 — typed operator tests.

Each operator must:
  * be in OPERATOR_REGISTRY
  * either return an IR-distinct child or a non-None rejection_reason
    on every call (no silent failure)
  * never mutate its parent IR
  * pass validate_function_ir on the produced child
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from algorithm_ir.ir.validator import validate_function_ir

from evolution.gp.canonical_hash import canonical_ir_hash
from evolution.gp.contract import SlotContract, TypedPort
from evolution.gp.operators.base import (
    GPContext,
    OPERATOR_REGISTRY,
    OperatorStats,
    run_operator_with_gates,
)


def _lmmse_ir():
    from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
    spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "lmmse")
    return compile_detector_template(spec)


def _ctx(rng_seed: int = 0) -> GPContext:
    contract = SlotContract(
        slot_key="lmmse.regularizer",
        short_name="regularizer",
        input_ports=(TypedPort("sigma2", "float"),),
        output_ports=(TypedPort("out", "float"),),
        complexity_cap=10_000,
    )
    return GPContext(
        contract=contract,
        region_op_ids=frozenset(),    # operate over whole IR
        rng=np.random.default_rng(rng_seed),
    )


REQUIRED_OPERATORS = (
    "mut_const",
    "mut_binary_swap",
    "mut_compare_swap",
    "mut_argswap",
    "mut_unary_flip",
    "mut_const_to_var",
)


@pytest.mark.parametrize("op_name", REQUIRED_OPERATORS)
def test_operator_is_registered(op_name):
    assert op_name in OPERATOR_REGISTRY, (
        f"{op_name} missing from OPERATOR_REGISTRY — "
        "typed_mutations import probably failed."
    )


@pytest.mark.parametrize("op_name", REQUIRED_OPERATORS)
def test_operator_does_not_mutate_parent(op_name):
    parent = _lmmse_ir()
    parent_hash_before = canonical_ir_hash(parent)
    factory, _, _ = OPERATOR_REGISTRY[op_name]
    op = factory()
    ctx = _ctx(rng_seed=42)
    for seed in range(8):
        ctx = GPContext(
            contract=ctx.contract,
            region_op_ids=ctx.region_op_ids,
            rng=np.random.default_rng(seed),
        )
        op.propose(ctx, parent)
    assert canonical_ir_hash(parent) == parent_hash_before, (
        f"{op_name} mutated its parent IR in place"
    )


@pytest.mark.parametrize("op_name", REQUIRED_OPERATORS)
def test_operator_returns_either_valid_child_or_rejection(op_name):
    parent = _lmmse_ir()
    parent_hash = canonical_ir_hash(parent)
    factory, _, _ = OPERATOR_REGISTRY[op_name]
    op = factory()

    n_accepted = 0
    n_rejected_with_reason = 0
    for seed in range(40):
        ctx = _ctx(rng_seed=seed)
        stats = OperatorStats(name=op_name)
        result = run_operator_with_gates(op, ctx, parent, parent_hash, stats=stats)
        if result.child_ir is not None:
            assert result.accepted_structurally
            assert result.child_hash != parent_hash
            errs = validate_function_ir(result.child_ir)
            assert not errs, f"{op_name} produced invalid child: {errs[:3]}"
            n_accepted += 1
        else:
            assert result.rejection_reason, (
                f"{op_name}: child_ir is None but no rejection_reason"
            )
            n_rejected_with_reason += 1

    # Each operator must either accept at least one or always reject
    # with a clear reason (the stronger property: no silent crash).
    assert (n_accepted + n_rejected_with_reason) == 40


def test_high_value_operators_accept_at_least_once_on_lmmse():
    """mut_const must produce at least one accepted variant on LMMSE.

    The LMMSE template has multiple float constants; if mut_const can't
    produce a single variant in 100 tries, the operator is broken.
    """
    parent = _lmmse_ir()
    parent_hash = canonical_ir_hash(parent)
    factory, _, _ = OPERATOR_REGISTRY["mut_const"]
    op = factory()
    n_accepted = 0
    for seed in range(100):
        ctx = _ctx(rng_seed=seed)
        result = run_operator_with_gates(op, ctx, parent, parent_hash)
        if result.child_ir is not None:
            n_accepted += 1
    assert n_accepted >= 50, (
        f"mut_const accepted only {n_accepted}/100 on LMMSE — "
        "expected at least 50% acceptance rate"
    )


def test_total_operator_weight_is_non_zero():
    total = sum(w for (_, w, _) in OPERATOR_REGISTRY.values())
    assert total > 0
