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
from algorithm_ir.runtime import execute_ir
from examples.algorithms import complex_tuple_kernel, simple_branch_loop, stack_decoder_host


class RuntimeFactGraphTests(unittest.TestCase):
    def test_execute_simple_branch_loop(self) -> None:
        func_ir = compile_function_to_ir(simple_branch_loop)
        expected = simple_branch_loop(5)
        actual, trace, runtime_values = execute_ir(func_ir, [5])
        self.assertEqual(actual, expected)
        self.assertGreater(len(trace), 0)
        self.assertGreater(len(runtime_values), 0)

    def test_execute_stack_decoder_and_build_factgraph(self) -> None:
        costs = [0.5, 0.25, 0.75]
        func_ir = compile_function_to_ir(stack_decoder_host)
        expected = stack_decoder_host(costs, 4)
        actual, trace, runtime_values = execute_ir(func_ir, [costs, 4])
        self.assertEqual(actual, expected)
        factgraph = build_factgraph(func_ir, trace, runtime_values)
        self.assertIn("def_use", factgraph.static_edges)
        self.assertIn("event_input", factgraph.dynamic_edges)
        self.assertGreater(len(factgraph.dynamic_edges["temporal"]), 0)
        self.assertGreater(len(factgraph.alignment_edges["instantiates_op"]), 0)

    def test_execute_complex_tuple_kernel(self) -> None:
        func_ir = compile_function_to_ir(complex_tuple_kernel)
        expected = complex_tuple_kernel(0.5)
        actual, trace, runtime_values = execute_ir(func_ir, [0.5])
        self.assertEqual(actual, expected)
        self.assertGreater(len(trace), 0)
        self.assertGreater(len(runtime_values), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
