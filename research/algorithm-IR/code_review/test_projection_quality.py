"""
Test projection detection and scoring.
Focus: Are projections meaningful or trivially generated?
"""
from __future__ import annotations
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import unittest
from algorithm_ir.frontend import compile_function_to_ir
from algorithm_ir.runtime import execute_ir
from algorithm_ir.factgraph import build_factgraph
from algorithm_ir.region import define_rewrite_region, infer_boundary_contract
from algorithm_ir.projection import annotate_region, Projection
from algorithm_ir.projection.local_interaction import detect_local_interaction_projection
from algorithm_ir.projection.scheduling import detect_scheduling_projection


def latest_value_for_var(func_ir, var_name: str) -> str:
    matches = [
        value_id for value_id, value in func_ir.values.items()
        if value.attrs.get("var_name") == var_name
    ]
    return sorted(matches, key=lambda v: func_ir.values[v].attrs.get("version", -1))[-1]


class TestProjectionDetection(unittest.TestCase):
    """Test whether projections are correctly detected."""

    def test_local_interaction_detected_for_score_region(self):
        from examples.algorithms import stack_decoder_host
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [[0.5, 0.25], 3])
        score_val = latest_value_for_var(ir, "score")
        region = define_rewrite_region(ir, exit_values=[score_val])
        proj = detect_local_interaction_projection(region)
        self.assertIsNotNone(proj)
        self.assertEqual(proj.family, "local_interaction")

    def test_scheduling_detected_for_expansion_region(self):
        from examples.algorithms import stack_decoder_host
        ir = compile_function_to_ir(stack_decoder_host)
        # Find the expansion block where left/right are created
        expansion_ops = []
        for block in ir.blocks.values():
            for op_id in block.op_ids:
                op = ir.ops[op_id]
                if op.attrs.get("target") in ("left", "right"):
                    expansion_ops.append(op_id)
        if expansion_ops:
            region = define_rewrite_region(ir, op_ids=expansion_ops)
            proj = detect_scheduling_projection(region)
            # This may or may not detect scheduling depending on write_set
            # Just ensure it doesn't crash
            if proj:
                self.assertEqual(proj.family, "scheduling")

    def test_no_projection_for_trivial_region(self):
        """A region with no entry/exit values should not get local_interaction."""
        def fn() -> int:
            return 42
        ir = compile_function_to_ir(fn)
        const_ops = [op.id for op in ir.ops.values() if op.opcode == "const"]
        if const_ops:
            region = define_rewrite_region(ir, op_ids=const_ops)
            proj = detect_local_interaction_projection(region)
            if not region.entry_values and not region.exit_values:
                self.assertIsNone(proj)

    def test_projection_scores_are_hardcoded(self):
        """Projection scores are constant (0.7 and 0.6) — they don't depend on data."""
        from examples.algorithms import stack_decoder_host
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [[0.5, 0.25], 3])
        score_val = latest_value_for_var(ir, "score")
        region = define_rewrite_region(ir, exit_values=[score_val])
        proj = detect_local_interaction_projection(region)
        self.assertEqual(proj.score, 0.7, "Score is hardcoded, not data-driven")

    def test_annotate_region_with_factgraph(self):
        from examples.algorithms import stack_decoder_host
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [[0.5, 0.25], 3])
        fg = build_factgraph(ir, trace, rv)
        score_val = latest_value_for_var(ir, "score")
        region = define_rewrite_region(ir, exit_values=[score_val])
        projections = annotate_region(region, fg)
        self.assertGreater(len(projections), 0)
        for p in projections:
            self.assertIn("factgraph_function", p.evidence)

    def test_edge_set_always_empty(self):
        """Projection edge_set is always empty — no actual graph structure is captured."""
        from examples.algorithms import stack_decoder_host
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [[0.5, 0.25], 3])
        score_val = latest_value_for_var(ir, "score")
        region = define_rewrite_region(ir, exit_values=[score_val])
        projections = annotate_region(region)
        for p in projections:
            self.assertEqual(p.edge_set, [], "edge_set is always empty — no real graph detected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
