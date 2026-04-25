"""Unit tests for evolution.slot_evolution (Phase H+5).

Verifies:
1. ``map_pop_key_to_from_slot_ids`` finds annotated ops on real genomes.
2. ``collect_slot_region`` returns a non-None region for lmmse.regularizer.
3. ``apply_slot_variant`` with the default variant yields a validate-clean IR.
4. SlotMicroStats invariants: validated ≤ attempted, improved ≤ evaluated.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from algorithm_ir.ir.validator import validate_function_ir
from evolution.ir_pool import build_ir_pool
from evolution.slot_evolution import (
    SlotMicroStats,
    apply_slot_variant,
    collect_slot_region,
    map_pop_key_to_from_slot_ids,
)


@pytest.fixture(scope="module")
def lmmse_genome():
    pool = build_ir_pool(np.random.default_rng(42))
    return next(g for g in pool if g.algo_id == "lmmse")


def test_map_pop_key_finds_lmmse_regularizer(lmmse_genome):
    sids = map_pop_key_to_from_slot_ids(lmmse_genome, "lmmse.regularizer")
    assert len(sids) >= 1
    assert all(s.startswith("_slot_regularizer_") for s in sids)


def test_collect_slot_region_returns_region(lmmse_genome):
    sids = map_pop_key_to_from_slot_ids(lmmse_genome, "lmmse.regularizer")
    region = collect_slot_region(lmmse_genome.ir, sids)
    assert region is not None
    assert len(region.op_ids) > 0
    assert len(region.entry_values) > 0
    assert len(region.exit_values) > 0


def test_apply_default_variant_is_valid_and_compiles(lmmse_genome):
    pop = lmmse_genome.slot_populations["lmmse.regularizer"]
    new_ir = apply_slot_variant(lmmse_genome, "lmmse.regularizer", pop.variants[0])
    assert new_ir is not None, "default splice must succeed"
    assert validate_function_ir(new_ir) == [], "post-graft IR must validate"


def test_slot_micro_stats_invariants():
    s = SlotMicroStats(
        slot_pop_key="x.y",
        n_attempted=10, n_validated=8, n_evaluated=6, n_improved=2,
        best_before=0.5, best_after=0.3,
    )
    d = s.as_dict()
    assert d["n_validated"] <= d["n_attempted"]
    assert d["n_evaluated"] <= d["n_validated"]
    assert d["n_improved"] <= d["n_evaluated"]
    assert d["best_delta"] == pytest.approx(-0.2)
