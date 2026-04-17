"""
Test region selection, slicing, and boundary contract inference.
Focus: Can the system correctly identify and isolate computation regions?
"""
from __future__ import annotations
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import unittest
from algorithm_ir.frontend import compile_function_to_ir
from algorithm_ir.ir import validate_function_ir
from algorithm_ir.region import define_rewrite_region, infer_boundary_contract
from algorithm_ir.region.slicer import backward_slice_by_values, forward_slice_from_values
from algorithm_ir.runtime import execute_ir


def latest_value_for_var(func_ir, var_name: str) -> str:
    matches = [
        value_id for value_id, value in func_ir.values.items()
        if value.attrs.get("var_name") == var_name
    ]
    return sorted(matches, key=lambda v: func_ir.values[v].attrs.get("version", -1))[-1]


def all_values_for_var(func_ir, var_name: str) -> list[str]:
    return [
        value_id for value_id, value in func_ir.values.items()
        if value.attrs.get("var_name") == var_name
    ]


class TestBackwardSlice(unittest.TestCase):
    """Test backward slice computation."""

    def test_backward_slice_captures_dependencies(self):
        def fn(a: int, b: int) -> int:
            x = a + 1
            y = b + 2
            z = x + y
            return z
        ir = compile_function_to_ir(fn)
        z_val = latest_value_for_var(ir, "z")
        sliced = backward_slice_by_values(ir, [z_val])
        # The slice for z should include ops that compute x, y, and z
        sliced_opcodes = {ir.ops[op_id].opcode for op_id in sliced}
        self.assertIn("binary", sliced_opcodes)
        # Should include at least 3 binary ops (x=a+1, y=b+2, z=x+y)
        self.assertGreaterEqual(len(sliced), 3)

    def test_backward_slice_stops_at_args(self):
        def fn(a: int) -> int:
            return a + 1
        ir = compile_function_to_ir(fn)
        # Get the return value
        return_ops = [op for op in ir.ops.values() if op.opcode == "return"]
        self.assertEqual(len(return_ops), 1)
        ret_input = return_ops[0].inputs[0]
        sliced = backward_slice_by_values(ir, [ret_input])
        # Should include const(1) and binary(a+1), plus assign
        self.assertGreater(len(sliced), 0)
        self.assertLessEqual(len(sliced), 10)  # should be small


class TestForwardSlice(unittest.TestCase):
    """Test forward slice computation."""

    def test_forward_slice_captures_consumers(self):
        def fn(a: int) -> int:
            x = a + 1
            y = x * 2
            z = y + 3
            return z
        ir = compile_function_to_ir(fn)
        x_val = latest_value_for_var(ir, "x")
        sliced = forward_slice_from_values(ir, [x_val])
        # Forward slice from x should include y and z computation, plus return
        self.assertGreater(len(sliced), 1)


class TestRegionDefinition(unittest.TestCase):
    """Test region definition via different selectors."""

    def test_region_via_op_ids(self):
        def fn(a: int, b: int) -> int:
            x = a + b
            y = x * 2
            return y
        ir = compile_function_to_ir(fn)
        # Select the binary ops manually
        binary_ops = [op.id for op in ir.ops.values() if op.opcode == "binary"]
        region = define_rewrite_region(ir, op_ids=binary_ops)
        self.assertEqual(len(region.op_ids), len(binary_ops))
        self.assertGreater(len(region.entry_values), 0)

    def test_region_via_exit_values(self):
        def fn(a: int, b: int) -> int:
            x = a + b
            y = x * 2
            return y
        ir = compile_function_to_ir(fn)
        y_val = latest_value_for_var(ir, "y")
        region = define_rewrite_region(ir, exit_values=[y_val])
        self.assertIn(y_val, region.exit_values)
        self.assertGreater(len(region.op_ids), 0)

    def test_region_requires_some_selector(self):
        def fn(a: int) -> int:
            return a
        ir = compile_function_to_ir(fn)
        with self.assertRaises(ValueError):
            define_rewrite_region(ir)

    def test_region_read_write_sets(self):
        """Test that read/write sets are detected for dict operations."""
        def fn() -> int:
            d = {"x": 1}
            val = d["x"]
            d["x"] = val + 1
            return d["x"]
        ir = compile_function_to_ir(fn)
        all_ops = list(ir.ops.keys())
        region = define_rewrite_region(ir, op_ids=all_ops)
        # Should detect read_set and write_set for dict items
        has_item_read = any("item:" in s for s in region.read_set)
        has_item_write = any("item:" in s for s in region.write_set)
        self.assertTrue(has_item_read, f"Expected item read in read_set, got: {region.read_set}")
        self.assertTrue(has_item_write, f"Expected item write in write_set, got: {region.write_set}")


class TestBoundaryContract(unittest.TestCase):
    """Test boundary contract inference."""

    def test_contract_has_correct_ports(self):
        def fn(a: int, b: int) -> int:
            x = a + b
            y = x * 2
            return y
        ir = compile_function_to_ir(fn)
        y_val = latest_value_for_var(ir, "y")
        region = define_rewrite_region(ir, exit_values=[y_val])
        _, trace, rv = execute_ir(ir, [3, 4])
        contract = infer_boundary_contract(ir, region, trace, rv)
        self.assertIn(y_val, contract.output_ports)
        self.assertGreater(len(contract.input_ports), 0)

    def test_contract_with_runtime_evidence(self):
        """Test that runtime values influence contract invariants."""
        def fn(x: float) -> float:
            return x * 2.0
        ir = compile_function_to_ir(fn)
        ret_ops = [op for op in ir.ops.values() if op.opcode == "return"]
        ret_val = ret_ops[0].inputs[0]
        region = define_rewrite_region(ir, exit_values=[ret_val])
        _, trace, rv = execute_ir(ir, [3.0])
        contract = infer_boundary_contract(ir, region, trace, rv)
        self.assertIn("scalar_outputs", contract.invariants)
        self.assertTrue(contract.invariants["scalar_outputs"])

    def test_contract_reconnect_points(self):
        """Reconnect points should list ops outside region that consume exit values."""
        def fn(a: int) -> int:
            x = a + 1
            y = x * 2
            return y
        ir = compile_function_to_ir(fn)
        x_val = latest_value_for_var(ir, "x")
        # Region only includes the computation of x
        x_def_op = ir.values[x_val].def_op
        region = define_rewrite_region(ir, op_ids=[x_def_op])
        _, trace, rv = execute_ir(ir, [5])
        contract = infer_boundary_contract(ir, region, trace, rv)
        # x is used outside the region (by the y computation)
        self.assertGreater(len(contract.reconnect_points), 0)


class TestRegionOnStackDecoder(unittest.TestCase):
    """Test region operations on the actual stack decoder algorithm."""

    def test_score_region_backward_slice(self):
        from examples.algorithms import stack_decoder_host
        ir = compile_function_to_ir(stack_decoder_host)
        _, trace, rv = execute_ir(ir, [[0.5, 0.25, 0.75], 4])
        score_val = latest_value_for_var(ir, "score")
        region = define_rewrite_region(ir, exit_values=[score_val])
        self.assertGreater(len(region.op_ids), 0)
        # The region should span multiple blocks or at least the score computation
        self.assertGreater(len(region.entry_values), 0)

    def test_source_span_region(self):
        from examples.algorithms import stack_decoder_host
        ir = compile_function_to_ir(stack_decoder_host)
        region = define_rewrite_region(ir, source_span=(15, 8, 16, 63))
        self.assertGreater(len(region.op_ids), 0)
        # All ops should have source spans that overlap the given range
        for op_id in region.op_ids:
            op = ir.ops[op_id]
            self.assertIsNotNone(op.source_span)


if __name__ == "__main__":
    unittest.main(verbosity=2)
