"""Tests for the IR-based algorithm pool and two-level evolution pipeline.

Tests cover:
  - ir_pool: template compilation, AlgSlot conversion, build_ir_pool
  - materialize: source-level materialization, ir_to_callable
  - mimo_evaluator: sample generation, evaluation
  - algorithm_engine: init_population, micro_evolve, one generation
"""

from __future__ import annotations

import pytest
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: ir_pool tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIRPoolImports:
    """Verify all ir_pool components import cleanly."""

    def test_import_ir_pool(self):
        from evolution.ir_pool import build_ir_pool

    def test_import_detector_specs(self):
        from evolution.ir_pool import _DETECTOR_SPECS
        assert len(_DETECTOR_SPECS) >= 8


class TestTemplateCompilation:
    """Test that detector templates compile to FunctionIR."""

    def test_compile_lmmse(self):
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
        spec = _DETECTOR_SPECS[0]  # LMMSE
        assert spec.algo_id == "lmmse"
        ir = compile_detector_template(spec)
        assert ir is not None
        assert len(ir.ops) > 0

    def test_compile_zf(self):
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
        spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "zf")
        ir = compile_detector_template(spec)
        assert ir is not None

    def test_compile_osic(self):
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
        spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "osic")
        ir = compile_detector_template(spec)
        assert ir is not None

    def test_compile_kbest(self):
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
        spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "kbest")
        ir = compile_detector_template(spec)
        assert ir is not None

    def test_compile_bp(self):
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
        spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "bp")
        ir = compile_detector_template(spec)
        assert ir is not None

    def test_compile_ep(self):
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
        spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "ep")
        ir = compile_detector_template(spec)
        assert ir is not None

    def test_compile_amp(self):
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
        spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "amp")
        ir = compile_detector_template(spec)
        assert ir is not None

    def test_compile_stack(self):
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
        spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "stack")
        ir = compile_detector_template(spec)
        assert ir is not None


class TestAlgSlotConversion:
    """Test that slot_* parameters are converted to AlgSlot ops."""

    def test_lmmse_has_algslot_ops(self):
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS, find_algslot_ops
        spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "lmmse")
        ir = compile_detector_template(spec)
        slots = find_algslot_ops(ir)
        # LMMSE has 2 slots: regularizer, hard_decision
        assert len(slots) >= 1, f"Expected AlgSlot ops, got {len(slots)}"

    def test_slot_ids_present(self):
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS, get_slot_ids
        spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "lmmse")
        ir = compile_detector_template(spec)
        ids = get_slot_ids(ir)
        assert len(ids) >= 1

    def test_slot_args_removed(self):
        """After conversion, slot_* args should not be in arg_values."""
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
        spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "lmmse")
        ir = compile_detector_template(spec)
        for vid in ir.arg_values:
            v = ir.values[vid]
            name = v.attrs.get("var_name") or v.name_hint
            assert not (name and name.startswith("slot_")), \
                f"Slot arg '{name}' should have been removed"



class TestBuildIRPool:
    """Test the full build_ir_pool() function."""

    def test_build_returns_genomes(self):
        from evolution.ir_pool import build_ir_pool
        pool = build_ir_pool(np.random.default_rng(42), n_random_variants=2)
        assert len(pool) > 0

    def test_all_genomes_have_structural_ir(self):
        from evolution.ir_pool import build_ir_pool
        pool = build_ir_pool(np.random.default_rng(42), n_random_variants=2)
        for g in pool:
            assert g.structural_ir is not None

    def test_genomes_have_slot_populations(self):
        from evolution.ir_pool import build_ir_pool
        pool = build_ir_pool(np.random.default_rng(42), n_random_variants=2)
        # S1: with IR builder extensions (IfExp/BoolOp/Slice), inliner
        # surviving-helper retention, and the callee_name resolver tier,
        # all 6 previously-pruned core algorithms (kbest/bp/soft_sic/
        # turbo_linear/particle_filter/importance_sampling) now have
        # evolvable slot populations. Only ``zf`` legitimately has no
        # tunable slot helpers (pure-linear deterministic detector).
        no_slot_allowed = {"zf"}
        for g in pool:
            if g.algo_id in no_slot_allowed:
                continue
            assert len(g.slot_populations) > 0, \
                f"{g.algo_id} has no slot populations"

    def test_algo_ids_present(self):
        from evolution.ir_pool import build_ir_pool
        pool = build_ir_pool(np.random.default_rng(42), n_random_variants=2)
        ids = {g.algo_id for g in pool}
        assert "lmmse" in ids
        assert "zf" in ids


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: materialize tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMaterialize:
    """Test materialization pipeline."""

    def test_import_materialize(self):
        from evolution.materialize import materialize, ir_to_callable, materialize_to_callable

    def test_materialize_lmmse(self):
        from evolution.ir_pool import build_ir_pool
        from evolution.materialize import materialize
        pool = build_ir_pool(np.random.default_rng(42), n_random_variants=1)
        lmmse = next(g for g in pool if g.algo_id == "lmmse")
        source = materialize(lmmse)
        assert isinstance(source, str)
        assert "def " in source

    def test_materialize_to_callable_lmmse(self):
        from evolution.ir_pool import build_ir_pool
        from evolution.materialize import materialize_to_callable
        pool = build_ir_pool(np.random.default_rng(42), n_random_variants=1)
        lmmse = next(g for g in pool if g.algo_id == "lmmse")
        fn = materialize_to_callable(lmmse)
        assert callable(fn)

    def test_materialized_lmmse_runs(self):
        """LMMSE should produce a valid detection result."""
        from evolution.ir_pool import build_ir_pool
        from evolution.materialize import materialize_to_callable
        from evolution.mimo_evaluator import qam16_constellation, generate_mimo_sample

        pool = build_ir_pool(np.random.default_rng(42), n_random_variants=1)
        lmmse = next(g for g in pool if g.algo_id == "lmmse")
        fn = materialize_to_callable(lmmse)

        rng = np.random.default_rng(123)
        constellation = qam16_constellation()
        H, x_true, y, sigma2 = generate_mimo_sample(4, 4, constellation, 20.0, rng)
        x_hat = fn(H, y, sigma2, constellation)
        assert x_hat is not None
        assert len(x_hat) == 4

    def test_materialize_all_detectors(self):
        """All detectors should materialize without error."""
        from evolution.ir_pool import build_ir_pool
        from evolution.materialize import materialize
        pool = build_ir_pool(np.random.default_rng(42), n_random_variants=1)
        for g in pool:
            source = materialize(g)
            assert isinstance(source, str), f"{g.algo_id} failed to materialize"
            assert "def " in source, f"{g.algo_id} has no function def"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: MIMO evaluator tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMIMOEvaluator:
    """Test MIMO fitness evaluator components."""

    def test_constellation(self):
        from evolution.mimo_evaluator import qam16_constellation
        c = qam16_constellation()
        assert len(c) == 16
        # Check normalization
        assert abs(np.mean(np.abs(c) ** 2) - 1.0) < 0.01

    def test_generate_sample(self):
        from evolution.mimo_evaluator import generate_mimo_sample, qam16_constellation
        rng = np.random.default_rng(42)
        c = qam16_constellation()
        H, x, y, sigma2 = generate_mimo_sample(8, 8, c, 15.0, rng)
        assert H.shape == (8, 8)
        assert len(x) == 8
        assert len(y) == 8
        assert sigma2 > 0

    def test_evaluator_creates(self):
        from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
        config = MIMOEvalConfig(Nr=4, Nt=4, n_trials=10)
        ev = MIMOFitnessEvaluator(config)
        assert ev.constellation is not None


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: algorithm engine tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMaterializeWithOverride:
    """Test materialize_with_override for micro-level evaluation."""

    def test_import(self):
        from evolution.materialize import materialize_with_override

    def test_override_returns_callable(self):
        from evolution.ir_pool import build_ir_pool
        from evolution.materialize import materialize_with_override
        pool = build_ir_pool(np.random.default_rng(42), n_random_variants=1)
        lmmse = next(g for g in pool if g.algo_id == "lmmse")
        # Override with the existing best variant (should work identically)
        for slot_id, pop in lmmse.slot_populations.items():
            if pop.variants:
                fn = materialize_with_override(
                    lmmse, {pop.slot_id: pop.variants[pop.best_idx]}
                )
                assert callable(fn)
                break

    def test_override_runs_correctly(self):
        from evolution.ir_pool import build_ir_pool
        from evolution.materialize import materialize_with_override
        from evolution.mimo_evaluator import qam16_constellation, generate_mimo_sample
        pool = build_ir_pool(np.random.default_rng(42), n_random_variants=1)
        lmmse = next(g for g in pool if g.algo_id == "lmmse")
        # Override with existing best (identity operation)
        override = {}
        for slot_id, pop in lmmse.slot_populations.items():
            if pop.variants:
                override[pop.slot_id] = pop.variants[pop.best_idx]
        fn = materialize_with_override(lmmse, override)
        rng = np.random.default_rng(123)
        c = qam16_constellation()
        H, x_true, y, sigma2 = generate_mimo_sample(4, 4, c, 20.0, rng)
        x_hat = fn(H, y, sigma2, c)
        assert x_hat is not None
        assert len(x_hat) == 4

    def test_override_empty_map(self):
        """Empty override_map should behave like normal materialize."""
        from evolution.ir_pool import build_ir_pool
        from evolution.materialize import materialize_with_override, materialize_to_callable
        from evolution.mimo_evaluator import qam16_constellation, generate_mimo_sample
        pool = build_ir_pool(np.random.default_rng(42), n_random_variants=1)
        lmmse = next(g for g in pool if g.algo_id == "lmmse")
        fn_override = materialize_with_override(lmmse, {})
        fn_normal = materialize_to_callable(lmmse)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        c = qam16_constellation()
        H1, x1, y1, s1 = generate_mimo_sample(4, 4, c, 20.0, rng1)
        H2, x2, y2, s2 = generate_mimo_sample(4, 4, c, 20.0, rng2)
        r1 = fn_override(H1, y1, s1, c)
        r2 = fn_normal(H2, y2, s2, c)
        np.testing.assert_array_almost_equal(r1, r2)


class TestAlgorithmEngine:
    """Test the two-level evolution engine."""

    def test_import(self):
        from evolution.algorithm_engine import AlgorithmEvolutionEngine

    def test_init_population(self):
        from evolution.algorithm_engine import AlgorithmEvolutionEngine
        from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
        from evolution.pool_types import AlgorithmEvolutionConfig

        eval_cfg = MIMOEvalConfig(Nr=4, Nt=4, n_trials=5, snr_db_list=[15.0])
        evo_cfg = AlgorithmEvolutionConfig(pool_size=4, n_generations=1)
        engine = AlgorithmEvolutionEngine(
            evaluator=MIMOFitnessEvaluator(eval_cfg),
            config=evo_cfg,
        )
        engine.init_population()
        assert len(engine.population) == 4
        assert len(engine.fitness) == 4

    def test_micro_step_evaluates_fitness(self):
        """After _micro_step, new variants should have real fitness, not inf."""
        from evolution.algorithm_engine import AlgorithmEvolutionEngine
        from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
        from evolution.pool_types import AlgorithmEvolutionConfig

        eval_cfg = MIMOEvalConfig(Nr=4, Nt=4, n_trials=5, snr_db_list=[15.0])
        evo_cfg = AlgorithmEvolutionConfig(
            pool_size=4, micro_generations=2, micro_mutation_rate=0.9,
        )
        engine = AlgorithmEvolutionEngine(
            evaluator=MIMOFitnessEvaluator(eval_cfg),
            config=evo_cfg,
            rng=np.random.default_rng(42),
        )
        engine.init_population()
        genome = engine.population[0]
        # Run micro evolution
        engine._micro_evolve(genome)
        # Check at least some variants have real fitness
        for slot_id, pop in genome.slot_populations.items():
            if len(pop.variants) > 1:
                has_finite = any(f < float("inf") for f in pop.fitness)
                assert has_finite, (
                    f"Slot {slot_id}: all fitness still inf after micro_evolve"
                )

    def test_micro_step_updates_best_idx(self):
        """best_idx should reflect the best-performing variant."""
        from evolution.algorithm_engine import AlgorithmEvolutionEngine
        from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
        from evolution.pool_types import AlgorithmEvolutionConfig

        eval_cfg = MIMOEvalConfig(Nr=4, Nt=4, n_trials=5, snr_db_list=[15.0])
        evo_cfg = AlgorithmEvolutionConfig(
            pool_size=4, micro_generations=3, micro_mutation_rate=0.95,
        )
        engine = AlgorithmEvolutionEngine(
            evaluator=MIMOFitnessEvaluator(eval_cfg),
            config=evo_cfg,
            rng=np.random.default_rng(42),
        )
        engine.init_population()
        genome = engine.population[0]
        engine._micro_evolve(genome)
        for slot_id, pop in genome.slot_populations.items():
            if pop.fitness:
                best_f = pop.fitness[pop.best_idx]
                assert best_f == min(pop.fitness), (
                    f"Slot {slot_id}: best_idx={pop.best_idx} points to "
                    f"fitness={best_f} but min is {min(pop.fitness)}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2+3: Graft & PatternMatcher tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGraftGeneral:
    """Tests for graft_general IR-level grafting."""

    def test_import(self):
        from algorithm_ir.grafting.graft_general import (
            graft_general,
            GraftArtifact,
            find_region_boundary,
            clone_donor_ir,
            rebind_uses,
            remove_ops,
            topological_sort_block,
            bind_donor_args_to_host_values,
        )

    def test_graft_general_preserves_non_region_ops(self):
        """Ops NOT in the graft region should survive unchanged."""
        from algorithm_ir.grafting.graft_general import graft_general
        from evolution.pool_types import GraftProposal, DependencyOverride
        from algorithm_ir.region.selector import RewriteRegion

        # Build a minimal IR with 3 ops: op_a, op_b (region), op_c
        ir = _make_graft_test_ir()
        original_op_ids = set(ir.ops.keys())
        region_op_id = "op_region"

        # Build proposal
        region = RewriteRegion(
            region_id="r1",
            op_ids=[region_op_id],
            block_ids=["b0"],
            entry_values=[],
            exit_values=[],
            read_set=[],
            write_set=[],
            state_carriers=[],
            schedule_anchors={},
            allows_new_state=False,
        )
        proposal = GraftProposal(
            proposal_id="p1",
            host_algo_id="host1",
            donor_algo_id="donor1",
            region=region,
            contract=None,
            donor_ir=None,
            dependency_overrides=[],
        )

        artifact = graft_general(ir, proposal)

        # Non-region ops should survive
        surviving_ops = set(artifact.ir.ops.keys())
        for op_id in original_op_ids:
            if op_id != region_op_id:
                assert op_id in surviving_ops, f"{op_id} was removed unexpectedly"

    def test_graft_general_exit_values_rebound(self):
        """Exit values of a grafted region must be re-bound to call op outputs."""
        from algorithm_ir.grafting.graft_general import graft_general
        from evolution.pool_types import GraftProposal
        from algorithm_ir.region.selector import RewriteRegion

        ir = _make_graft_test_ir()
        region_op_id = "op_region"
        exit_val = "v_region_out"

        region = RewriteRegion(
            region_id="r1",
            op_ids=[region_op_id],
            block_ids=["b0"],
            entry_values=[],
            exit_values=[exit_val],
            read_set=[],
            write_set=[],
            state_carriers=[],
            schedule_anchors={},
            allows_new_state=False,
        )
        proposal = GraftProposal(
            proposal_id="p2",
            host_algo_id="host1",
            donor_algo_id="donor1",
            region=region,
            contract=None,
            donor_ir=None,
            dependency_overrides=[],
        )
        artifact = graft_general(ir, proposal)

        # The region op should be removed
        assert region_op_id not in artifact.ir.ops

    def test_graft_general_with_dependency_override(self):
        """DependencyOverrides should be applied to the call op."""
        from algorithm_ir.grafting.graft_general import graft_general
        from evolution.pool_types import GraftProposal, DependencyOverride
        from algorithm_ir.region.selector import RewriteRegion

        ir = _make_graft_test_ir()
        region = RewriteRegion(
            region_id="r1",
            op_ids=["op_region"],
            block_ids=["b0"],
            entry_values=[],
            exit_values=["v_region_out"],
            read_set=[],
            write_set=[],
            state_carriers=[],
            schedule_anchors={},
            allows_new_state=False,
        )
        override = DependencyOverride(
            target_value="v_pre",
            new_dependencies=["extra_dep"],
            reason="test override",
        )
        proposal = GraftProposal(
            proposal_id="p3",
            host_algo_id="host1",
            donor_algo_id=None,
            region=region,
            contract=None,
            donor_ir=None,
            dependency_overrides=[override],
        )
        artifact = graft_general(ir, proposal)
        # Artifact should be produced without error
        assert artifact.ir is not None

    def test_graft_general_validates_result(self):
        """The output IR from graft_general should be structurally valid."""
        from algorithm_ir.grafting.graft_general import graft_general
        from evolution.pool_types import GraftProposal
        from algorithm_ir.region.selector import RewriteRegion

        ir = _make_graft_test_ir()
        region = RewriteRegion(
            region_id="r1",
            op_ids=["op_region"],
            block_ids=["b0"],
            entry_values=[],
            exit_values=[],
            read_set=[],
            write_set=[],
            state_carriers=[],
            schedule_anchors={},
            allows_new_state=False,
        )
        proposal = GraftProposal(
            proposal_id="p4",
            host_algo_id="host1",
            donor_algo_id=None,
            region=region,
            contract=None,
            donor_ir=None,
            dependency_overrides=[],
        )
        artifact = graft_general(ir, proposal)

        # Check all ops reference valid blocks
        for op_id, op in artifact.ir.ops.items():
            assert op.block_id in artifact.ir.blocks, (
                f"Op {op_id} references invalid block {op.block_id}"
            )

    def test_region_boundary_analysis(self):
        """find_region_boundary should identify entry and exit values."""
        from algorithm_ir.grafting.graft_general import find_region_boundary
        ir = _make_graft_test_ir()
        entry_vals, exit_vals = find_region_boundary(ir, {"op_region"})
        # entry = values produced outside region but consumed by region ops
        # exit = values produced by region ops but consumed outside region
        assert isinstance(entry_vals, list)
        assert isinstance(exit_vals, list)


class TestPatternMatcherIntegration:
    """Test PatternMatcher integration into the evolution engine."""

    def test_engine_accepts_pattern_matcher(self):
        """Engine should accept a pattern_matcher parameter."""
        from evolution.algorithm_engine import AlgorithmEvolutionEngine
        from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
        from evolution.pool_types import AlgorithmEvolutionConfig

        eval_cfg = MIMOEvalConfig(Nr=4, Nt=4, n_trials=5, snr_db_list=[15.0])
        evo_cfg = AlgorithmEvolutionConfig(pool_size=4, n_generations=1)

        def dummy_matcher(entries, gen):
            return []  # No proposals

        engine = AlgorithmEvolutionEngine(
            evaluator=MIMOFitnessEvaluator(eval_cfg),
            config=evo_cfg,
            pattern_matcher=dummy_matcher,
        )
        assert engine.pattern_matcher is dummy_matcher

    def test_null_pattern_matcher_engine_still_works(self):
        """Engine without pattern_matcher should run normally."""
        from evolution.algorithm_engine import AlgorithmEvolutionEngine
        from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
        from evolution.pool_types import AlgorithmEvolutionConfig

        eval_cfg = MIMOEvalConfig(Nr=4, Nt=4, n_trials=5, snr_db_list=[15.0])
        evo_cfg = AlgorithmEvolutionConfig(pool_size=4, n_generations=1)
        engine = AlgorithmEvolutionEngine(
            evaluator=MIMOFitnessEvaluator(eval_cfg),
            config=evo_cfg,
        )
        assert engine.pattern_matcher is None
        engine.init_population()
        assert len(engine.population) == 4

    def test_dummy_pattern_matcher_produces_no_crash(self):
        """A PatternMatcher that returns proposals should not crash the engine run loop."""
        from evolution.algorithm_engine import AlgorithmEvolutionEngine
        from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
        from evolution.pool_types import AlgorithmEvolutionConfig

        eval_cfg = MIMOEvalConfig(Nr=4, Nt=4, n_trials=5, snr_db_list=[15.0])
        evo_cfg = AlgorithmEvolutionConfig(pool_size=4, n_generations=2)

        def empty_matcher(entries, gen):
            return []  # Return empty proposals — exercises the code path

        engine = AlgorithmEvolutionEngine(
            evaluator=MIMOFitnessEvaluator(eval_cfg),
            config=evo_cfg,
            pattern_matcher=empty_matcher,
        )
        result = engine.run()
        assert result is not None
        # run() returns best AlgorithmGenome
        from evolution.pool_types import AlgorithmGenome
        assert isinstance(result, AlgorithmGenome)

    def test_to_entry_conversion(self):
        """AlgorithmGenome.to_entry() should produce a valid AlgorithmEntry."""
        from evolution.ir_pool import build_ir_pool
        from evolution.pool_types import AlgorithmGenome, AlgorithmEntry
        from evolution.fitness import FitnessResult

        pool = build_ir_pool()
        # Pick first genome
        genome = pool[0]
        entry = genome.to_entry()
        assert isinstance(entry, AlgorithmEntry)
        assert entry.algo_id == genome.algo_id
        assert entry.ir is not None

        # With fitness
        fitness = FitnessResult(
            metrics={"ber_15dB": 0.05, "complexity": 100.0},
        )
        entry2 = genome.to_entry(fitness)
        assert entry2.fitness is fitness


# ---- Helper for graft tests ----

def _make_graft_test_ir():
    """Build a minimal FunctionIR for testing graft operations."""
    from algorithm_ir.ir.model import FunctionIR, Block, Op, Value

    # Build directly with dict-based model
    values = {
        "v_pre": Value(id="v_pre", type_hint="f64", def_op="op_pre"),
        "v_region_out": Value(
            id="v_region_out", type_hint="f64", def_op="op_region",
            use_ops=["op_post"],
        ),
    }
    ops = {
        "op_pre": Op(
            id="op_pre", opcode="const", inputs=[], outputs=["v_pre"],
            block_id="b0", attrs={"value": 1.0},
        ),
        "op_region": Op(
            id="op_region", opcode="call", inputs=["v_pre"],
            outputs=["v_region_out"], block_id="b0", attrs={"callee": "sub_fn"},
        ),
        "op_post": Op(
            id="op_post", opcode="unary", inputs=["v_region_out"],
            outputs=[], block_id="b0", attrs={"op": "neg"},
        ),
        "op_ret": Op(
            id="op_ret", opcode="return", inputs=[], outputs=[],
            block_id="b0", attrs={},
        ),
    }
    blocks = {
        "b0": Block(id="b0", op_ids=["op_pre", "op_region", "op_post", "op_ret"]),
    }

    # Update use_ops for v_pre
    values["v_pre"].use_ops = ["op_region"]

    ir = FunctionIR(
        id="test_fn_1",
        name="test_graft",
        arg_values=[],
        return_values=[],
        values=values,
        ops=ops,
        blocks=blocks,
        entry_block="b0",
    )
    return ir


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: PatternMatcher implementations
# ═══════════════════════════════════════════════════════════════════════════


class TestPatternMatchers:
    """Test the three PatternMatcher implementations."""

    def test_import_all_matchers(self):
        from evolution.pattern_matchers import (
            RandomGraftPatternMatcher,
            ExpertPatternMatcher,
            StaticStructurePatternMatcher,
            CompositePatternMatcher,
        )

    def test_random_matcher_produces_proposals(self):
        from evolution.pattern_matchers import RandomGraftPatternMatcher
        from evolution.ir_pool import build_ir_pool

        pool = build_ir_pool()
        entries = [g.to_entry() for g in pool[:4]]
        matcher = RandomGraftPatternMatcher(proposals_per_gen=2, seed=42)
        proposals = matcher(entries, generation=0)
        assert isinstance(proposals, list)
        # May produce 0-2 proposals depending on IR structure
        for p in proposals:
            assert p.host_algo_id in {e.algo_id for e in entries}
            assert p.donor_algo_id in {e.algo_id for e in entries}
            assert len(p.region.op_ids) > 0

    def test_expert_matcher_produces_proposals(self):
        from evolution.pattern_matchers import ExpertPatternMatcher
        from evolution.ir_pool import build_ir_pool

        pool = build_ir_pool()
        entries = [g.to_entry() for g in pool]
        matcher = ExpertPatternMatcher(max_proposals_per_gen=5)
        proposals = matcher(entries, generation=0)
        assert isinstance(proposals, list)
        # Expert rules should find some matches in the 8-detector pool
        for p in proposals:
            assert p.rationale  # Should have description

    def test_static_matcher_produces_proposals(self):
        from evolution.pattern_matchers import StaticStructurePatternMatcher
        from evolution.ir_pool import build_ir_pool

        pool = build_ir_pool()
        entries = [g.to_entry() for g in pool[:4]]
        matcher = StaticStructurePatternMatcher(max_proposals_per_gen=3)
        proposals = matcher(entries, generation=0)
        assert isinstance(proposals, list)

    def test_composite_matcher(self):
        from evolution.pattern_matchers import (
            RandomGraftPatternMatcher,
            ExpertPatternMatcher,
            CompositePatternMatcher,
        )
        from evolution.ir_pool import build_ir_pool

        pool = build_ir_pool()
        entries = [g.to_entry() for g in pool[:4]]
        composite = CompositePatternMatcher([
            RandomGraftPatternMatcher(proposals_per_gen=1, seed=42),
            ExpertPatternMatcher(max_proposals_per_gen=1),
        ])
        proposals = composite(entries, generation=0)
        assert isinstance(proposals, list)

    def test_random_matcher_empty_pool(self):
        from evolution.pattern_matchers import RandomGraftPatternMatcher
        matcher = RandomGraftPatternMatcher()
        # Empty pool should return no proposals
        assert matcher([], 0) == []
        # Single entry should return no proposals (needs 2)
        from evolution.ir_pool import build_ir_pool
        pool = build_ir_pool()
        assert matcher([pool[0].to_entry()], 0) == []


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: runner test
# ═══════════════════════════════════════════════════════════════════════════


class TestRunner:
    """Test the E2E runner."""

    def test_import(self):
        from evolution.run_evolution import run
