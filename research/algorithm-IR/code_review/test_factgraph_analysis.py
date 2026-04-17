"""
Test the FactGraph construction and analysis modules.
Focus: Is the FactGraph a meaningful structure or just a data dump?
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
from algorithm_ir.analysis import def_use_edges, block_uses, runtime_values_for_static, fingerprint_runtime_value


class TestFactGraphConstruction(unittest.TestCase):
    """Test FactGraph edge construction."""

    def test_def_use_edges_correct(self):
        def fn(a: int, b: int) -> int:
            x = a + b
            return x
        ir = compile_function_to_ir(fn)
        _, trace, rv = execute_ir(ir, [3, 4])
        fg = build_factgraph(ir, trace, rv)
        self.assertIn("def_use", fg.static_edges)
        self.assertGreater(len(fg.static_edges["def_use"]), 0)

    def test_cfg_edges_present(self):
        def fn(x: int) -> int:
            if x > 0:
                return x
            return 0
        ir = compile_function_to_ir(fn)
        _, trace, rv = execute_ir(ir, [5])
        fg = build_factgraph(ir, trace, rv)
        self.assertIn("cfg", fg.static_edges)
        self.assertGreater(len(fg.static_edges["cfg"]), 0)

    def test_temporal_edges(self):
        def fn(x: int) -> int:
            y = x + 1
            z = y + 2
            return z
        ir = compile_function_to_ir(fn)
        _, trace, rv = execute_ir(ir, [5])
        fg = build_factgraph(ir, trace, rv)
        self.assertIn("temporal", fg.dynamic_edges)
        self.assertGreater(len(fg.dynamic_edges["temporal"]), 0)

    def test_alignment_edges(self):
        def fn(x: int) -> int:
            return x + 1
        ir = compile_function_to_ir(fn)
        _, trace, rv = execute_ir(ir, [5])
        fg = build_factgraph(ir, trace, rv)
        self.assertIn("instantiates_op", fg.alignment_edges)
        self.assertIn("instantiates_value", fg.alignment_edges)
        self.assertGreater(len(fg.alignment_edges["instantiates_op"]), 0)

    def test_event_input_output_edges(self):
        def fn(x: int) -> int:
            return x * 2
        ir = compile_function_to_ir(fn)
        _, trace, rv = execute_ir(ir, [5])
        fg = build_factgraph(ir, trace, rv)
        self.assertGreater(len(fg.dynamic_edges["event_input"]), 0)
        self.assertGreater(len(fg.dynamic_edges["event_output"]), 0)


class TestStaticAnalysis(unittest.TestCase):
    """Test static analysis utilities."""

    def test_def_use_edges_utility(self):
        def fn(a: int) -> int:
            x = a + 1
            y = x + 2
            return y
        ir = compile_function_to_ir(fn)
        edges = def_use_edges(ir)
        self.assertGreater(len(edges), 0)
        # All edges should be (op_id, op_id) pairs
        for src, dst in edges:
            self.assertIn(src, ir.ops)
            self.assertIn(dst, ir.ops)

    def test_block_uses(self):
        def fn(a: int) -> int:
            return a + 1
        ir = compile_function_to_ir(fn)
        entry_uses = block_uses(ir, ir.entry_block)
        self.assertIsInstance(entry_uses, set)
        self.assertGreater(len(entry_uses), 0)


class TestDynamicAnalysis(unittest.TestCase):
    """Test dynamic analysis utilities."""

    def test_runtime_values_for_static(self):
        def fn(x: int) -> int:
            y = x + 1
            return y
        ir = compile_function_to_ir(fn)
        _, _, rv = execute_ir(ir, [5])
        # Find a static value that was instantiated at runtime
        any_static_id = list(ir.values.keys())[0]
        runtime_vals = runtime_values_for_static(rv, any_static_id)
        # Should find at least some matches
        # (not all static values will have runtime instances if they are args)
        self.assertIsInstance(runtime_vals, list)

    def test_fingerprint_runtime_value(self):
        def fn(x: int) -> int:
            return x + 1
        ir = compile_function_to_ir(fn)
        _, _, rv = execute_ir(ir, [5])
        for runtime_val in rv.values():
            fp = fingerprint_runtime_value(runtime_val)
            self.assertIsInstance(fp, tuple)
            self.assertEqual(len(fp), 4)


class TestContainerTracking(unittest.TestCase):
    """Test shadow store container membership tracking in FactGraph."""

    def test_list_container_membership(self):
        def fn() -> int:
            lst = [1, 2, 3]
            lst.append(4)
            return len(lst)
        ir = compile_function_to_ir(fn)
        _, trace, rv = execute_ir(ir, [])
        fg = build_factgraph(ir, trace, rv)
        # Container membership edges should exist
        cm_edges = fg.dynamic_edges.get("container_membership", set())
        self.assertGreater(len(cm_edges), 0,
                          "Container membership edges should be tracked for list operations")


if __name__ == "__main__":
    unittest.main(verbosity=2)
