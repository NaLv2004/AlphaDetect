"""Phase H+4 S3 — micro-population step tests.

Verifies the typed GP step actually runs through the slot population
and tracks per-operator stats. Uses a stub evaluator (no subprocess)
so the test runs in-process and is fast/deterministic.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from evolution.gp.canonical_hash import canonical_ir_hash
from evolution.gp.population import micro_population_step


class _StubEvaluator:
    """Pretends to evaluate by returning ser=hash(ir)/MAX, monotonically.

    Provides ``evaluate_source_quick`` so evaluate_slot_variant takes the
    subprocess code path (we override it to bypass actual subprocess).
    """

    def __init__(self, base: float = 0.5):
        self.base = base
        self.calls = 0

    def evaluate_source_quick(self, source, fn_name, *,
                              algo_id, n_trials, timeout_sec, snr_db):
        self.calls += 1
        # Return a SER decreasing with call count to simulate improvement.
        return max(0.001, self.base - 0.0001 * self.calls)


@pytest.fixture
def lmmse_pop():
    from evolution.ir_pool import build_ir_pool
    pool = build_ir_pool(np.random.default_rng(7), n_random_variants=1)
    lmmse = next(g for g in pool if g.algo_id == "lmmse")
    # Pick a slot population that the resolver can find.
    from evolution.gp.region_resolver import resolve_slot_region
    pop_key = None
    for k in list(lmmse.slot_populations.keys()):
        if resolve_slot_region(lmmse, k) is not None:
            pop_key = k
            break
    if pop_key is None:
        pytest.skip("no resolvable slot pop on lmmse genome")
    return lmmse, pop_key, lmmse.slot_populations[pop_key]


def test_micro_step_runs_and_tracks_attempts(lmmse_pop):
    genome, pop_key, pop = lmmse_pop
    rng = np.random.default_rng(123)
    eval_ = _StubEvaluator(base=0.5)
    stats = micro_population_step(
        genome, pop_key, pop,
        evaluator=eval_, rng=rng,
        n_children=8, n_trials=1, timeout_sec=0.5, snr_db=16.0,
        max_pop_size=16,
    )
    assert stats.n_attempted == 8
    # Non-skipped path:
    assert stats.skipped_no_sids == 0
    assert stats.skipped_no_variants == 0
    # Per-operator stats should exist.
    per_op = getattr(stats, "per_operator", {})
    assert isinstance(per_op, dict) and len(per_op) >= 1


def test_micro_step_skips_unresolvable_slot(lmmse_pop):
    genome, _pop_key, _pop = lmmse_pop
    # Inject a fake pop with a key that cannot be resolved.
    from evolution.pool_types import SlotPopulation
    from evolution.skeleton_registry import ProgramSpec
    fake_pop = SlotPopulation(
        slot_id="fake",
        spec=ProgramSpec(name="fake", param_names=[], param_types=[]),
        variants=[], fitness=[], best_idx=0,
    )
    rng = np.random.default_rng(0)
    stats = micro_population_step(
        genome, "nonexistent.slot_key", fake_pop,
        evaluator=_StubEvaluator(), rng=rng, n_children=4,
    )
    # Empty variants triggers skipped_no_variants before resolver check.
    assert stats.skipped_no_variants == 1


def test_step_slot_population_delegates_to_typed_gp(lmmse_pop):
    """Full integration — calling step_slot_population uses typed GP."""
    from evolution import slot_evolution
    genome, pop_key, pop = lmmse_pop
    rng = np.random.default_rng(42)
    stats = slot_evolution.step_slot_population(
        genome, pop_key, pop,
        evaluator=_StubEvaluator(),
        rng=rng,
        n_children=4, n_trials=1, timeout_sec=0.5, snr_db=16.0,
    )
    assert stats.n_attempted == 4
    # per_operator is the new typed-GP-only attribute.
    assert hasattr(stats, "per_operator")


def test_legacy_path_still_works_when_use_typed_gp_false(lmmse_pop):
    from evolution import slot_evolution
    genome, pop_key, pop = lmmse_pop
    rng = np.random.default_rng(0)
    stats = slot_evolution.step_slot_population(
        genome, pop_key, pop,
        evaluator=_StubEvaluator(),
        rng=rng,
        n_children=2, n_trials=1, timeout_sec=0.5, snr_db=16.0,
        use_typed_gp=False,
    )
    assert stats.n_attempted == 2
