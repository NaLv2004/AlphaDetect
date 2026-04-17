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

from algorithm_ir.factgraph import build_factgraph
from algorithm_ir.frontend import compile_function_to_ir
from algorithm_ir.projection import annotate_region
from algorithm_ir.region import define_rewrite_region, infer_boundary_contract
from algorithm_ir.runtime import execute_ir
from examples.algorithms import stack_decoder_host


def latest_value_for_var(func_ir, var_name: str) -> str:
    matches = [
        value_id
        for value_id, value in func_ir.values.items()
        if value.attrs.get("var_name") == var_name
    ]
    return sorted(matches, key=lambda value_id: func_ir.values[value_id].attrs["version"])[-1]


class RegionProjectionTests(unittest.TestCase):
    def test_define_region_and_infer_contract(self) -> None:
        costs = [0.5, 0.25, 0.75]
        func_ir = compile_function_to_ir(stack_decoder_host)
        _, trace, runtime_values = execute_ir(func_ir, [costs, 4])
        score_value = latest_value_for_var(func_ir, "score")
        region = define_rewrite_region(func_ir, exit_values=[score_value])
        contract = infer_boundary_contract(func_ir, region, trace, runtime_values)
        self.assertGreater(len(region.op_ids), 0)
        self.assertIn(score_value, contract.output_ports)
        self.assertGreater(len(contract.input_ports), 0)
        self.assertIn("scalar_outputs", contract.invariants)

    def test_optional_projection_annotation(self) -> None:
        costs = [0.5, 0.25, 0.75]
        func_ir = compile_function_to_ir(stack_decoder_host)
        _, trace, runtime_values = execute_ir(func_ir, [costs, 4])
        score_value = latest_value_for_var(func_ir, "score")
        region = define_rewrite_region(func_ir, exit_values=[score_value])
        fg = build_factgraph(func_ir, trace, runtime_values)
        projections = annotate_region(region, fg)
        families = {projection.family for projection in projections}
        self.assertIn("local_interaction", families)


if __name__ == "__main__":
    unittest.main(verbosity=2)
