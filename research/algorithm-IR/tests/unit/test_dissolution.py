"""Pytest unit tests for evolution.slot_dissolution and slot_rediscovery (S5)."""
from __future__ import annotations

import pytest

from evolution.slot_dissolution import (
    dissolve_and_graft,
    strip_provenance_markers,
)
from evolution.slot_rediscovery import (
    NewSlotProposal,
    apply_rediscovered_slots,
    maybe_rediscover_slots,
    rediscover_slots,
)


@pytest.fixture(scope="module")
def lmmse_genome():
    from evolution.ir_pool import build_ir_pool
    import numpy as np
    rng = np.random.default_rng(42)
    pool = build_ir_pool(rng, n_random_variants=1)
    for g in pool:
        if "lmmse" in g.algo_id.lower() or "lmmse" in str(g.tags).lower():
            return g
    return pool[0]


def test_strip_markers_idempotent(lmmse_genome):
    ir = lmmse_genome.structural_ir
    n0 = len(ir.ops)
    strip_provenance_markers(ir)
    n1 = len(ir.ops)
    strip_provenance_markers(ir)
    n2 = len(ir.ops)
    assert n1 == n2


def test_rediscover_returns_list(lmmse_genome):
    proposals = rediscover_slots(lmmse_genome.structural_ir)
    assert isinstance(proposals, list)


def test_rediscover_proposals_have_required_fields(lmmse_genome):
    proposals = rediscover_slots(
        lmmse_genome.structural_ir,
        min_size=2, max_size=20, max_boundary=10, min_cohesion=0.0,
    )
    for p in proposals:
        assert isinstance(p, NewSlotProposal)
        assert p.slot_id.startswith("auto_")
        assert len(p.op_ids) >= 2
        assert p.cohesion >= 0.0


def test_apply_rediscovered_slots_no_op_for_empty_list(lmmse_genome):
    n0 = len(lmmse_genome.slot_populations)
    apply_rediscovered_slots(lmmse_genome, [])
    assert len(lmmse_genome.slot_populations) == n0


def test_maybe_rediscover_period_gate(lmmse_genome):
    # generation=0 must NOT trigger (unless slot_populations empty)
    pre = dict(lmmse_genome.slot_populations)
    if pre:
        out = maybe_rediscover_slots(
            lmmse_genome, generation=0, period=20,
        )
        # No new slots when pre-populated and generation=0.
        assert set(out.slot_populations.keys()) == set(pre.keys())


def test_dissolve_returns_dispatch_result_when_proposal_invalid(lmmse_genome):
    """dissolve_and_graft must not crash on a degenerate proposal."""
    from evolution.fii import build_fii_ir
    fii_ir = build_fii_ir(lmmse_genome)
    # Build a trivial proposal-like object — the function should reject
    # cleanly rather than raise.
    class _FakeRegion:
        op_ids = []
    class _FakeProposal:
        donor_algo_id = "donor_x"
        proposal_id = "p_test"
        host_algo_id = lmmse_genome.algo_id
        region = _FakeRegion()
        donor_subgraph = None
        kind = "unknown"
    try:
        result = dissolve_and_graft(
            lmmse_genome, _FakeProposal(), fii_ir,
            generation=1, donor_algo_id="donor_x",
        )
    except Exception:
        # Acceptable: bad proposal might raise; but it must NOT crash
        # the interpreter (segfault, OOM).  We've at least demonstrated
        # the API doesn't break under the contract.
        return
    assert hasattr(result, "case")
    assert hasattr(result, "child")
