"""
End-to-end stress tests for the full pipeline.
Focus: Can the system handle realistic algorithm patterns?
"""
from __future__ import annotations
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import unittest
from algorithm_ir.frontend import compile_function_to_ir
from algorithm_ir.ir import validate_function_ir, render_function_ir
from algorithm_ir.runtime import execute_ir
from algorithm_ir.factgraph import build_factgraph
from algorithm_ir.region import define_rewrite_region, infer_boundary_contract
from algorithm_ir.projection import annotate_region


class TestRealisticAlgorithms(unittest.TestCase):
    """Test with algorithms that resemble MIMO detection patterns."""

    def test_simple_mmse_like(self):
        """Test a simplified MMSE-like computation pattern."""
        def mmse_like(h_real: float, h_imag: float, y_real: float, y_imag: float, noise: float) -> float:
            h_sq = h_real * h_real + h_imag * h_imag
            denom = h_sq + noise
            x_real = (h_real * y_real + h_imag * y_imag) / denom
            x_imag = (h_real * y_imag - h_imag * y_real) / denom
            return x_real * x_real + x_imag * x_imag
        ir = compile_function_to_ir(mmse_like)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [1.0, 0.5, 2.0, 1.0, 0.1])
        expected = mmse_like(1.0, 0.5, 2.0, 1.0, 0.1)
        self.assertAlmostEqual(result, expected)

    def test_greedy_search_pattern(self):
        """Test a greedy search pattern similar to K-best detection."""
        def greedy_search(costs: list, k: int) -> float:
            best = 9999.0
            i = 0
            while i < len(costs):
                if costs[i] < best:
                    best = costs[i]
                i = i + 1
            return best
        ir = compile_function_to_ir(greedy_search)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [[3.0, 1.0, 2.0], 1])
        self.assertEqual(result, 1.0)

    def test_iterative_refinement(self):
        """Test an iterative refinement pattern (like iterative detection)."""
        def refine(x: float, target: float, steps: int) -> float:
            i = 0
            while i < steps:
                error = target - x
                x = x + error * 0.5
                i = i + 1
            return x
        ir = compile_function_to_ir(refine)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [0.0, 10.0, 20])
        self.assertAlmostEqual(result, 10.0, places=2)


class TestFullPipelineStress(unittest.TestCase):
    """Stress test the full compile -> execute -> region -> contract -> factgraph pipeline."""

    def test_large_loop_count(self):
        """Test with a moderate number of loop iterations."""
        def fn(n: int) -> int:
            s = 0
            i = 0
            while i < n:
                s = s + i
                i = i + 1
            return s
        ir = compile_function_to_ir(fn)
        result, trace, rv = execute_ir(ir, [50])
        self.assertEqual(result, fn(50))
        self.assertGreater(len(trace), 100)

    def test_full_pipeline_on_custom_algorithm(self):
        """Full pipeline: compile -> execute -> factgraph -> region -> contract -> projection."""
        def priority_search(costs: list, max_iter: int) -> float:
            frontier = [{"node": 0, "cost": 0.0}]
            best = 9999.0
            step = 0
            while step < max_iter:
                if len(frontier) == 0:
                    return best
                # Find minimum
                min_idx = 0
                scan = 1
                while scan < len(frontier):
                    if frontier[scan]["cost"] < frontier[min_idx]["cost"]:
                        min_idx = scan
                    scan = scan + 1
                current = frontier.pop(min_idx)
                score = current["cost"] + costs[current["node"]]
                if score < best:
                    best = score
                next_node = current["node"] + 1
                if next_node < len(costs):
                    frontier.append({"node": next_node, "cost": score})
                step = step + 1
            return best

        ir = compile_function_to_ir(priority_search)
        self.assertEqual(validate_function_ir(ir), [])
        costs = [0.5, 0.3, 0.7, 0.1]
        result, trace, rv = execute_ir(ir, [costs, 10])
        self.assertEqual(result, priority_search(costs, 10))
        fg = build_factgraph(ir, trace, rv)
        self.assertGreater(len(fg.static_edges["def_use"]), 0)
        self.assertGreater(len(fg.dynamic_edges["temporal"]), 0)

    def test_multi_variable_phi_merge(self):
        """Test that phi nodes correctly merge multiple variables."""
        def fn(x: int) -> int:
            a = 0
            b = 0
            c = 0
            if x > 10:
                a = 1
                b = 2
                c = 3
            else:
                a = 4
                b = 5
                c = 6
            return a + b + c
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        for val, expected in [(15, 6), (5, 15)]:
            result, _, _ = execute_ir(ir, [val])
            self.assertEqual(result, expected, f"Failed for x={val}")


class TestArtifactReusability(unittest.TestCase):
    """Test that artifacts can be fed back into the pipeline."""

    def test_grafted_ir_can_be_analyzed(self):
        """After grafting, the resulting IR should support full analysis pipeline."""
        from examples.algorithms import stack_decoder_host, bp_summary_update
        from algorithm_ir.grafting import graft_skeleton, make_bp_summary_skeleton

        costs = [0.5, 0.25, 0.75]
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [costs, 4])
        region = define_rewrite_region(ir, source_span=(15, 8, 16, 63))
        contract = infer_boundary_contract(ir, region, trace, rv)
        skel = make_bp_summary_skeleton(bp_summary_update, damping=0.1)
        artifact = graft_skeleton(ir, region, contract, skel)

        # Now analyze the grafted artifact
        new_result, new_trace, new_rv = execute_ir(artifact.ir, [costs, 4])
        new_fg = build_factgraph(artifact.ir, new_trace, new_rv)
        self.assertGreater(len(new_fg.static_edges["def_use"]), 0)
        self.assertGreater(len(new_fg.dynamic_edges["temporal"]), 0)

    def test_grafted_ir_can_define_regions(self):
        """After grafting, we should be able to define new regions on the result."""
        from examples.algorithms import stack_decoder_host, bp_summary_update
        from algorithm_ir.grafting import graft_skeleton, make_bp_summary_skeleton

        costs = [0.5, 0.25, 0.75]
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [costs, 4])
        region = define_rewrite_region(ir, source_span=(15, 8, 16, 63))
        contract = infer_boundary_contract(ir, region, trace, rv)
        skel = make_bp_summary_skeleton(bp_summary_update, damping=0.1)
        artifact = graft_skeleton(ir, region, contract, skel)

        # Define a new region on the grafted IR
        new_result, new_trace, new_rv = execute_ir(artifact.ir, [costs, 4])
        grafted_ops = [op.id for op in artifact.ir.ops.values() if op.attrs.get("grafted")]
        if grafted_ops:
            new_region = define_rewrite_region(artifact.ir, op_ids=grafted_ops)
            self.assertGreater(len(new_region.op_ids), 0)
            new_contract = infer_boundary_contract(artifact.ir, new_region, new_trace, new_rv)
            self.assertGreater(len(new_contract.input_ports), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
