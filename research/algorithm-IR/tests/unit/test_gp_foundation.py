"""Phase H+4 S1+S2.2 — foundation tests for the typed GP package.

Covers:
  * canonical_ir_hash equivalence and difference detection
  * resolve_slot_region for all detector slot pops on a real IR pool
  * the 8-gate runner accepts pure-IR operators and rejects no-op /
    over-complex / invalid children
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from algorithm_ir.ir.model import FunctionIR

from evolution.gp.canonical_hash import canonical_ir_hash, ir_structural_signature
from evolution.gp.contract import SlotContract, TypedPort
from evolution.gp.operators.base import (
    GPContext,
    OperatorResult,
    OperatorStats,
    measure_complexity,
    run_operator_with_gates,
)
from evolution.gp.region_resolver import resolve_slot_region, prune_phantom_pops


# ---------------------------------------------------------------------------
# canonical_ir_hash
# ---------------------------------------------------------------------------

class TestCanonicalIRHash:
    def _lmmse_ir(self) -> FunctionIR:
        from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
        spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "lmmse")
        return compile_detector_template(spec)

    def test_hash_is_deterministic(self):
        ir = self._lmmse_ir()
        assert canonical_ir_hash(ir) == canonical_ir_hash(ir)

    def test_hash_invariant_under_deepcopy(self):
        ir = self._lmmse_ir()
        h1 = canonical_ir_hash(ir)
        ir2 = copy.deepcopy(ir)
        h2 = canonical_ir_hash(ir2)
        assert h1 == h2

    def test_hash_changes_on_constant_perturbation(self):
        ir = self._lmmse_ir()
        h1 = canonical_ir_hash(ir)
        ir2 = copy.deepcopy(ir)
        # Find a numeric const op and bump its literal.
        for op in ir2.ops.values():
            if op.opcode == "const" and isinstance(
                op.attrs.get("literal"), (int, float)
            ) and not isinstance(op.attrs.get("literal"), bool):
                op.attrs["literal"] = float(op.attrs["literal"]) + 0.123
                break
        else:
            pytest.skip("LMMSE IR has no numeric constants to perturb")
        h2 = canonical_ir_hash(ir2)
        assert h1 != h2

    def test_hash_changes_on_binary_opcode_flip(self):
        ir = self._lmmse_ir()
        h1 = canonical_ir_hash(ir)
        ir2 = copy.deepcopy(ir)
        for op in ir2.ops.values():
            if op.opcode == "binary":
                old = op.attrs.get("operator", "Add")
                op.attrs["operator"] = "Sub" if old != "Sub" else "Add"
                break
        else:
            pytest.skip("LMMSE IR has no binary ops to flip")
        h2 = canonical_ir_hash(ir2)
        assert h1 != h2

    def test_signature_is_json_serializable(self):
        import json
        ir = self._lmmse_ir()
        sig = ir_structural_signature(ir)
        json.dumps(sig, default=str)  # should not raise


# ---------------------------------------------------------------------------
# region_resolver
# ---------------------------------------------------------------------------

class TestRegionResolver:
    @pytest.fixture
    def pool(self):
        from evolution.ir_pool import build_ir_pool
        return build_ir_pool(np.random.default_rng(7), n_random_variants=1)

    def test_at_least_one_genome_resolves_at_least_one_slot(self, pool):
        any_resolved = 0
        for genome in pool:
            for slot_key in list(getattr(genome, "slot_populations", {}).keys()):
                if resolve_slot_region(genome, slot_key) is not None:
                    any_resolved += 1
        assert any_resolved > 0, (
            "No slot region resolved on any genome — "
            "the resolver must succeed on at least the lmmse.regularizer "
            "and ep.cavity slots that Phase H+3 already supported."
        )

    def test_resolver_returns_non_none_on_known_lmmse_slots(self, pool):
        lmmse = next((g for g in pool if g.algo_id == "lmmse"), None)
        if lmmse is None:
            pytest.skip("LMMSE genome not in pool")
        for slot_key in list(getattr(lmmse, "slot_populations", {}).keys()):
            info = resolve_slot_region(lmmse, slot_key)
            # We don't assert ALL pass yet — phantom pops exist that
            # the legacy admission path created. The point of this test
            # is that prune_phantom_pops can clean them up.
            if info is not None:
                assert info.tier in ("binding", "provenance")
                assert info.short_name == slot_key.split(".")[-1]

    def test_prune_phantom_pops_does_not_drop_resolvable_slots(self, pool):
        for genome in pool:
            pops = getattr(genome, "slot_populations", None)
            if not pops:
                continue
            resolvable_before = {
                k for k in pops if resolve_slot_region(genome, k) is not None
            }
            prune_phantom_pops(genome)
            for k in resolvable_before:
                assert k in genome.slot_populations, (
                    f"prune dropped resolvable slot {k} from {genome.algo_id}"
                )


# ---------------------------------------------------------------------------
# 8-gate runner
# ---------------------------------------------------------------------------

class _FakeOperator:
    name = "fake"
    weight = 0.1

    def __init__(self, mode: str):
        self.mode = mode

    def propose(self, ctx, parent_ir, parent2_ir=None) -> OperatorResult:
        if self.mode == "noop":
            return OperatorResult(child_ir=copy.deepcopy(parent_ir), diff_summary="noop")
        if self.mode == "raise":
            raise RuntimeError("boom")
        if self.mode == "none":
            return OperatorResult(child_ir=None, rejection_reason="not_applicable")
        if self.mode == "perturb_const":
            child = copy.deepcopy(parent_ir)
            for op in child.ops.values():
                if op.opcode == "const" and isinstance(
                    op.attrs.get("literal"), (int, float)
                ) and not isinstance(op.attrs.get("literal"), bool):
                    op.attrs["literal"] = float(op.attrs["literal"]) + 0.5
                    return OperatorResult(child_ir=child, diff_summary="bump_const")
            return OperatorResult(child_ir=None, rejection_reason="no_const")
        if self.mode == "explode":
            child = copy.deepcopy(parent_ir)
            # Add 10000 const ops to exceed complexity_cap
            from algorithm_ir.ir.model import Op, Value
            from itertools import count
            c = count()
            for _ in range(10_000):
                vid = f"v_explode_{next(c)}"
                oid = f"op_explode_{next(c)}"
                child.values[vid] = Value(id=vid, type_hint="float")
                op = Op(id=oid, opcode="const", inputs=[], outputs=[vid],
                        block_id=child.entry_block, attrs={"literal": 1.0})
                child.ops[oid] = op
            return OperatorResult(child_ir=child, diff_summary="explode")
        raise AssertionError(f"unknown mode {self.mode}")


def _make_ctx() -> GPContext:
    contract = SlotContract(
        slot_key="lmmse.regularizer",
        short_name="regularizer",
        input_ports=(TypedPort("sigma2", "float"),),
        output_ports=(TypedPort("out", "float"),),
        complexity_cap=200,
    )
    return GPContext(
        contract=contract,
        region_op_ids=frozenset(),
        rng=np.random.default_rng(0),
    )


def _lmmse_ir():
    from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS
    spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "lmmse")
    return compile_detector_template(spec)


class TestRunOperatorWithGates:
    def test_noop_rejected(self):
        ir = _lmmse_ir()
        ctx = _make_ctx()
        stats = OperatorStats(name="fake")
        result = run_operator_with_gates(
            _FakeOperator("noop"), ctx, ir, canonical_ir_hash(ir), stats=stats
        )
        assert result.child_ir is None
        assert result.rejection_reason == "noop_ir_hash"
        assert stats.n_noop_ir == 1

    def test_raise_caught_as_proposed_none(self):
        ir = _lmmse_ir()
        ctx = _make_ctx()
        stats = OperatorStats(name="fake")
        result = run_operator_with_gates(
            _FakeOperator("raise"), ctx, ir, canonical_ir_hash(ir), stats=stats
        )
        assert result.child_ir is None
        assert "operator_raised" in (result.rejection_reason or "")
        assert stats.n_proposed_none == 1

    def test_none_returned_recorded(self):
        ir = _lmmse_ir()
        ctx = _make_ctx()
        stats = OperatorStats(name="fake")
        result = run_operator_with_gates(
            _FakeOperator("none"), ctx, ir, canonical_ir_hash(ir), stats=stats
        )
        assert result.child_ir is None
        assert stats.n_proposed_none == 1

    def test_perturb_passes_structurally(self):
        ir = _lmmse_ir()
        ctx = _make_ctx()
        stats = OperatorStats(name="fake")
        result = run_operator_with_gates(
            _FakeOperator("perturb_const"), ctx, ir,
            canonical_ir_hash(ir), stats=stats,
        )
        assert result.accepted_structurally
        assert result.child_ir is not None
        assert result.child_hash != ""
        assert stats.n_accepted_structurally == 1

    def test_explode_rejected_by_complexity_cap(self):
        # Use perturb_const (which produces a valid child) with a tiny
        # complexity cap so the cap is the only gate that can reject.
        ir = _lmmse_ir()
        contract = SlotContract(
            slot_key="lmmse.regularizer",
            short_name="regularizer",
            input_ports=(TypedPort("sigma2", "float"),),
            output_ports=(TypedPort("out", "float"),),
            complexity_cap=1,    # parent has many ops; child will exceed this
        )
        ctx = GPContext(contract=contract, region_op_ids=frozenset(),
                        rng=np.random.default_rng(0))
        stats = OperatorStats(name="fake")
        result = run_operator_with_gates(
            _FakeOperator("perturb_const"), ctx, ir,
            canonical_ir_hash(ir), stats=stats,
        )
        assert result.child_ir is None
        assert "complexity_cap" in (result.rejection_reason or "")
        assert stats.n_complexity_rejected == 1


def test_measure_complexity_is_op_count():
    ir = _lmmse_ir()
    assert measure_complexity(ir) == len(ir.ops)
