"""R6: behavior signature gate.

If a structurally-distinct child produces SER identical to its parent
(a behavioral synonym), it must NOT be added to the population.

We monkeypatch ``apply_slot_variant`` and ``evaluate_slot_variant`` in
``evolution.gp.population`` so the test exercises ONLY the gate logic
inside ``micro_population_step`` without dragging in the grafting and
evaluator machinery.
"""
from __future__ import annotations

import numpy as np
import pytest

from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.ir.validator import validate_function_ir

import evolution.slot_evolution as se_mod
from evolution.gp.population import micro_population_step
from evolution.pool_types import AlgorithmGenome, SlotPopulation
from evolution.skeleton_registry import ProgramSpec
from evolution.slot_evolution import SlotMicroStats


HOST_SRC = """
def f(x: vec_cx, y: vec_cx):
    a = x + y
    b = a * x
    c = b + 1.0
    d = c * 2.0
    return d
"""


def _make_genome():
    ir = compile_source_to_ir(HOST_SRC, "f")
    assert validate_function_ir(ir) == []
    # Tag every op with provenance so resolve_slot_region finds the region.
    for op in ir.ops.values():
        if not isinstance(op.attrs, dict):
            op.attrs = {}
        prov = op.attrs.get("_provenance")
        if not isinstance(prov, dict):
            prov = {}
            op.attrs["_provenance"] = prov
        prov["from_slot_id"] = "_slot_body"
    spec = ProgramSpec(
        name="body", param_names=["x", "y"],
        param_types=["vec_cx", "vec_cx"], return_type="vec_cx",
    )
    pop = SlotPopulation(slot_id="f.body", spec=spec)
    pop.variants = [compile_source_to_ir(HOST_SRC, "f")]
    pop.fitness = [0.1]
    pop.best_idx = 0
    g = AlgorithmGenome(
        algo_id="dummy",
        ir=ir,
        slot_populations={"f.body": pop},
    )
    return g, pop


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Make apply_slot_variant a no-op (returns the parent IR) and
    make evaluate_slot_variant return the constant SER 0.1."""
    monkeypatch.setattr(
        se_mod, "apply_slot_variant",
        lambda genome, pop_key, child, **_kw: child,
    )
    monkeypatch.setattr(
        se_mod, "evaluate_slot_variant",
        lambda genome, pop_key, child, **_kw: (0.1, None),
    )
    yield


def test_behavior_gate_drops_synonym(stub_pipeline):
    g, pop = _make_genome()
    rng = np.random.default_rng(0)
    n_before = len(pop.variants)
    stats: SlotMicroStats = micro_population_step(
        g, "f.body", pop,
        evaluator=object(), rng=rng,
        n_children=12, n_trials=1, timeout_sec=0.5,
        snr_db=16.0, max_pop_size=64, complexity_cap=2048,
    )
    assert stats.n_attempted > 0
    # Every IR-distinct child has identical SER -> behavior gate must
    # reject all of them.
    assert stats.n_evaluated == 0, (
        f"behavior-noop children leaked into evaluated count: "
        f"n_evaluated={stats.n_evaluated} n_noop_behavior={stats.n_noop_behavior}"
    )
    assert len(pop.variants) == n_before, (
        f"behavior-noop children leaked into pop: before={n_before} after={len(pop.variants)}"
    )
    assert stats.n_noop_behavior > 0


def test_behavior_gate_records_per_operator(stub_pipeline):
    g, pop = _make_genome()
    rng = np.random.default_rng(1)
    stats: SlotMicroStats = micro_population_step(
        g, "f.body", pop,
        evaluator=object(), rng=rng,
        n_children=20, n_trials=1, timeout_sec=0.5,
        snr_db=16.0, max_pop_size=64, complexity_cap=2048,
    )
    per_op = getattr(stats, "per_operator", {})
    total_per_op = sum(s.n_noop_behavior for s in per_op.values())
    assert total_per_op == stats.n_noop_behavior


def test_behavior_gate_passes_real_improvement(monkeypatch):
    """If the child's SER is strictly better than parent's, it must be kept."""
    g, pop = _make_genome()
    monkeypatch.setattr(
        se_mod, "apply_slot_variant",
        lambda genome, pop_key, child, **_kw: child,
    )
    # Each call returns a slightly better SER so every child is an improvement.
    counter = {"n": 0}

    def _eval(genome, pop_key, child, **_kw):
        counter["n"] += 1
        return (0.1 - 1e-3 * counter["n"], None)

    monkeypatch.setattr(se_mod, "evaluate_slot_variant", _eval)
    rng = np.random.default_rng(2)
    n_before = len(pop.variants)
    stats: SlotMicroStats = micro_population_step(
        g, "f.body", pop,
        evaluator=object(), rng=rng,
        n_children=4, n_trials=1, timeout_sec=0.5,
        snr_db=16.0, max_pop_size=64, complexity_cap=2048,
    )
    # All accepted children should land in the pool (not gated as noop).
    assert stats.n_evaluated >= 1
    assert stats.n_noop_behavior == 0
    assert len(pop.variants) > n_before

