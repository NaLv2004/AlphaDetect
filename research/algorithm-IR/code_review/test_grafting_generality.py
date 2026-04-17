"""
Test grafting/rewriting capabilities — the core value proposition.
Focus: Can we actually graft one algorithm's logic into another?
       Are the grafting mechanisms general or hardcoded?
"""
from __future__ import annotations
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import unittest
from algorithm_ir.frontend import compile_function_to_ir
from algorithm_ir.ir import validate_function_ir, render_function_ir
from algorithm_ir.grafting import graft_skeleton, make_bp_summary_skeleton, make_bp_tree_runtime_skeleton
from algorithm_ir.grafting.skeletons import Skeleton
from algorithm_ir.grafting.matcher import match_skeleton
from algorithm_ir.region import define_rewrite_region, infer_boundary_contract
from algorithm_ir.runtime import execute_ir
from examples.algorithms import (
    bp_summary_update, bp_tree_runtime_update,
    stack_decoder_host, stack_decoder_runtime_host,
)


def latest_value_for_var(func_ir, var_name: str) -> str:
    matches = [
        value_id for value_id, value in func_ir.values.items()
        if value.attrs.get("var_name") == var_name
    ]
    return sorted(matches, key=lambda v: func_ir.values[v].attrs.get("version", -1))[-1]


class TestSkeletonMatching(unittest.TestCase):
    """Test skeleton matching logic."""

    def test_bp_summary_matches_stack_decoder(self):
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [[0.5, 0.25], 3])
        region = define_rewrite_region(ir, source_span=(15, 8, 16, 63))
        contract = infer_boundary_contract(ir, region, trace, rv)
        skel = make_bp_summary_skeleton(bp_summary_update)
        self.assertTrue(match_skeleton(ir, region, contract, skel))

    def test_mismatch_when_inputs_missing(self):
        """A skeleton requiring inputs not present should not match."""
        def simple_fn(x: int) -> int:
            return x + 1
        ir = compile_function_to_ir(simple_fn)
        all_ops = list(ir.ops.keys())
        region = define_rewrite_region(ir, op_ids=all_ops)
        _, trace, rv = execute_ir(ir, [5])
        contract = infer_boundary_contract(ir, region, trace, rv)
        skel = make_bp_summary_skeleton(bp_summary_update)
        # simple_fn doesn't have frontier/costs/candidate
        self.assertFalse(match_skeleton(ir, region, contract, skel))


class TestGraftingGenerality(unittest.TestCase):
    """Test whether grafting is truly general or just hardcoded for 2 skeletons."""

    def test_only_two_skeletons_supported(self):
        """The graft_skeleton function only supports bp_summary_update and bp_tree_runtime_update.
        This test documents that limitation."""
        custom_skel = Skeleton(
            skel_id="skel_custom",
            name="custom_update",
            required_contract={
                "needs_inputs": ["frontier", "costs", "candidate"],
                "requires_scalar_output": True,
            },
            transform_rules=[{"kind": "custom_transform"}],
            lowering_template={"damping": 0.1},
            donor_callable=bp_summary_update,
            donor_ir=compile_function_to_ir(bp_summary_update),
        )
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [[0.5, 0.25], 3])
        region = define_rewrite_region(ir, source_span=(15, 8, 16, 63))
        contract = infer_boundary_contract(ir, region, trace, rv)
        # Matching should succeed (it only checks inputs)
        self.assertTrue(match_skeleton(ir, region, contract, custom_skel))
        # But grafting should fail because only 2 names are supported
        with self.assertRaises(NotImplementedError):
            graft_skeleton(ir, region, contract, custom_skel)

    def test_grafted_ir_is_executable_and_valid(self):
        """After grafting, the IR should be both valid and executable."""
        costs = [0.5, 0.25, 0.75]
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [costs, 4])
        region = define_rewrite_region(ir, source_span=(15, 8, 16, 63))
        contract = infer_boundary_contract(ir, region, trace, rv)
        skel = make_bp_summary_skeleton(bp_summary_update, damping=0.1)
        artifact = graft_skeleton(ir, region, contract, skel)
        errors = validate_function_ir(artifact.ir)
        self.assertEqual(errors, [], "\n".join(errors))
        result, trace2, _ = execute_ir(artifact.ir, [costs, 4])
        self.assertIsInstance(result, float)

    def test_grafted_result_differs_from_original(self):
        """Grafting should actually change the algorithm behavior."""
        costs = [0.5, 0.25, 0.75]
        ir = compile_function_to_ir(stack_decoder_host)
        orig_result, _, _ = execute_ir(ir, [costs, 4])
        _, trace, rv = execute_ir(ir, [costs, 4])
        region = define_rewrite_region(ir, source_span=(15, 8, 16, 63))
        contract = infer_boundary_contract(ir, region, trace, rv)
        skel = make_bp_summary_skeleton(bp_summary_update, damping=0.1)
        artifact = graft_skeleton(ir, region, contract, skel)
        new_result, _, _ = execute_ir(artifact.ir, [costs, 4])
        self.assertNotEqual(orig_result, new_result,
                          "Grafting should change the result, but it didn't")

    def test_runtime_bp_grafting(self):
        """Test the runtime tree BP skeleton grafting."""
        costs = [0.5, 0.25, 0.75]
        ir = compile_function_to_ir(stack_decoder_runtime_host)
        _, trace, rv = execute_ir(ir, [costs, 3, []])
        # Find the expansion block
        region_ops = None
        for block in ir.blocks.values():
            targets = {ir.ops[op_id].attrs.get("target") for op_id in block.op_ids}
            if {"left", "right"} <= targets:
                region_ops = list(block.op_ids)
                break
        self.assertIsNotNone(region_ops, "Could not find expansion block")
        region = define_rewrite_region(ir, op_ids=region_ops)
        contract = infer_boundary_contract(ir, region, trace, rv)
        skel = make_bp_tree_runtime_skeleton(bp_tree_runtime_update, damping=0.1)
        artifact = graft_skeleton(ir, region, contract, skel)
        errors = validate_function_ir(artifact.ir)
        self.assertEqual(errors, [], "\n".join(errors))
        audit = []
        result, _, _ = execute_ir(artifact.ir, [costs, 3, audit])
        self.assertIsInstance(result, float)
        self.assertGreater(len(audit), 0, "Runtime BP should have recorded audit entries")


class TestCodegenQuality(unittest.TestCase):
    """Test that generated artifacts have meaningful source code / IR text."""

    def test_artifact_source_code_is_not_empty(self):
        costs = [0.5, 0.25, 0.75]
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [costs, 4])
        region = define_rewrite_region(ir, source_span=(15, 8, 16, 63))
        contract = infer_boundary_contract(ir, region, trace, rv)
        skel = make_bp_summary_skeleton(bp_summary_update, damping=0.1)
        artifact = graft_skeleton(ir, region, contract, skel)
        self.assertGreater(len(artifact.source_code), 0)

    def test_artifact_source_is_not_python(self):
        """The codegen only emits IR text, NOT executable Python.
        This tests whether emit_artifact_source produces real code or just IR dumps."""
        costs = [0.5, 0.25, 0.75]
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [costs, 4])
        region = define_rewrite_region(ir, source_span=(15, 8, 16, 63))
        contract = infer_boundary_contract(ir, region, trace, rv)
        skel = make_bp_summary_skeleton(bp_summary_update, damping=0.1)
        artifact = graft_skeleton(ir, region, contract, skel)
        from algorithm_ir.regeneration.codegen import emit_artifact_source
        source = emit_artifact_source(artifact)
        # It should be IR text, not Python
        self.assertIn("FunctionIR", source)
        self.assertNotIn("def stack_decoder", source)

    def test_provenance_tracking(self):
        """Provenance should record host, donor, region, contract."""
        costs = [0.5, 0.25, 0.75]
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [costs, 4])
        region = define_rewrite_region(ir, source_span=(15, 8, 16, 63))
        contract = infer_boundary_contract(ir, region, trace, rv)
        skel = make_bp_summary_skeleton(bp_summary_update, damping=0.1)
        artifact = graft_skeleton(ir, region, contract, skel)
        self.assertIn("host_function", artifact.provenance)
        self.assertIn("donor_function", artifact.provenance)
        self.assertIn("rewrite_region", artifact.provenance)
        self.assertIn("boundary_contract", artifact.provenance)
        self.assertEqual(artifact.provenance["donor_function"], "bp_summary_update")


class TestGraftingWithDifferentAlgorithms(unittest.TestCase):
    """Try grafting with novel donor functions to test generality."""

    def test_custom_donor_function_rejected_by_graft_skeleton(self):
        """Even if a custom donor meets the contract, graft_skeleton hard-checks skeleton.name."""
        def custom_score(frontier, costs, damping):
            return sum(c * damping for c in costs)
        custom_skel = Skeleton(
            skel_id="skel_custom_score",
            name="custom_score_update",  # Not bp_summary_update or bp_tree_runtime_update
            required_contract={
                "needs_inputs": ["frontier", "costs", "candidate"],
                "requires_scalar_output": True,
            },
            transform_rules=[],
            lowering_template={"damping": 0.1},
            donor_callable=custom_score,
            donor_ir=None,
        )
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [[0.5, 0.25], 3])
        region = define_rewrite_region(ir, source_span=(15, 8, 16, 63))
        contract = infer_boundary_contract(ir, region, trace, rv)
        with self.assertRaises(NotImplementedError):
            graft_skeleton(ir, region, contract, custom_skel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
