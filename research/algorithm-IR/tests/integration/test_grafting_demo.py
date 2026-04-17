from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from algorithm_ir.frontend import compile_function_to_ir
from algorithm_ir.grafting import graft_skeleton, make_bp_summary_skeleton, make_bp_tree_runtime_skeleton
from algorithm_ir.ir import validate_function_ir
from algorithm_ir.region import define_rewrite_region, infer_boundary_contract
from algorithm_ir.runtime import execute_ir
from examples.algorithms import (
    bp_summary_update,
    bp_tree_runtime_update,
    stack_decoder_host,
    stack_decoder_runtime_host,
)


class GraftingDemoTests(unittest.TestCase):
    def test_stack_decoder_bp_grafting_demo(self) -> None:
        costs = [0.5, 0.25, 0.75]
        host_ir = compile_function_to_ir(stack_decoder_host)
        donor_ir = compile_function_to_ir(bp_summary_update)

        self.assertEqual(validate_function_ir(host_ir), [])
        self.assertEqual(validate_function_ir(donor_ir), [])

        _, trace, runtime_values = execute_ir(host_ir, [costs, 4])
        region = define_rewrite_region(host_ir, source_span=(15, 8, 16, 63))
        contract = infer_boundary_contract(host_ir, region, trace, runtime_values)
        skeleton = make_bp_summary_skeleton(bp_summary_update, damping=0.1)
        artifact = graft_skeleton(host_ir, region, contract, skeleton)

        errors = validate_function_ir(artifact.ir)
        self.assertEqual(errors, [], msg="\n".join(errors))
        result, new_trace, _ = execute_ir(artifact.ir, [costs, 4])
        self.assertIsInstance(result, float)
        self.assertGreater(len(new_trace), 0)
        self.assertEqual(artifact.provenance["donor_function"], "bp_summary_update")
        self.assertTrue(any("grafted" in op.attrs for op in artifact.ir.ops.values()))
        self.assertIn("xdsl_module", artifact.ir.attrs)
        self.assertIn("bp_summary_update", artifact.ir.attrs["xdsl_text"])

        cloned_ir = artifact.ir.clone()
        cloned_result, cloned_trace, _ = execute_ir(cloned_ir, [costs, 4])
        self.assertEqual(cloned_result, result)
        self.assertGreater(len(cloned_trace), 0)

    def test_stack_decoder_runtime_tree_bp_grafting_demo(self) -> None:
        costs = [0.5, 0.25, 0.75]
        baseline_audit: list[int] = []
        host_ir = compile_function_to_ir(stack_decoder_runtime_host)
        donor_ir = compile_function_to_ir(bp_tree_runtime_update)

        self.assertEqual(validate_function_ir(host_ir), [])
        self.assertEqual(validate_function_ir(donor_ir), [])

        baseline_result, trace, runtime_values = execute_ir(host_ir, [costs, 3, baseline_audit])
        self.assertEqual(baseline_audit, [])

        region = define_rewrite_region(host_ir, op_ids=_runtime_expansion_region_ops(host_ir))
        contract = infer_boundary_contract(host_ir, region, trace, runtime_values)
        skeleton = make_bp_tree_runtime_skeleton(bp_tree_runtime_update, damping=0.1)
        artifact = graft_skeleton(host_ir, region, contract, skeleton)

        errors = validate_function_ir(artifact.ir)
        self.assertEqual(errors, [], msg="\n".join(errors))

        rewritten_audit: list[int] = []
        rewritten_result, new_trace, _ = execute_ir(artifact.ir, [costs, 3, rewritten_audit])
        self.assertIsInstance(rewritten_result, float)
        self.assertGreater(len(new_trace), 0)
        self.assertEqual(artifact.provenance["donor_function"], "bp_tree_runtime_update")
        self.assertTrue(any(op.attrs.get("runtime_bp_pass") for op in artifact.ir.ops.values()))
        self.assertEqual(rewritten_audit, [3, 5, 7])
        self.assertEqual(baseline_result, 0.5)
        self.assertIn("runtime_bp_pass", artifact.ir.attrs["xdsl_text"])

        cloned_ir = artifact.ir.clone()
        cloned_audit: list[int] = []
        cloned_result, cloned_trace, _ = execute_ir(cloned_ir, [costs, 3, cloned_audit])
        self.assertEqual(cloned_result, rewritten_result)
        self.assertEqual(cloned_audit, [3, 5, 7])
        self.assertGreater(len(cloned_trace), 0)


def _runtime_expansion_region_ops(func_ir):
    for block in func_ir.blocks.values():
        targets = {func_ir.ops[op_id].attrs.get("target") for op_id in block.op_ids}
        if {"left", "right"} <= targets:
            return list(block.op_ids)
    raise AssertionError("Could not locate expansion block for runtime BP graft")


if __name__ == "__main__":
    unittest.main(verbosity=2)
