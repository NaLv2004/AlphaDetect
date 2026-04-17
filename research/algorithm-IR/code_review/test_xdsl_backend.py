"""
Test xDSL bridge — the actual backend of the IR.
Focus: Is xDSL truly the substrate, or just a mirror?
"""
from __future__ import annotations
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import unittest
from algorithm_ir.frontend import compile_function_to_ir
from algorithm_ir.ir import validate_function_ir, render_function_ir
from algorithm_ir.ir.xdsl_bridge import render_xdsl_module
from algorithm_ir.runtime import execute_ir


class TestXDSLBackend(unittest.TestCase):
    """Test xDSL integration."""

    def test_xdsl_module_exists(self):
        def fn(x: int) -> int:
            return x + 1
        ir = compile_function_to_ir(fn)
        self.assertIsNotNone(ir.xdsl_module)
        self.assertIsNotNone(ir.xdsl_func)

    def test_xdsl_text_in_attrs(self):
        def fn(x: int) -> int:
            return x + 1
        ir = compile_function_to_ir(fn)
        self.assertIn("xdsl_text", ir.attrs)
        self.assertIn("func.func", ir.attrs["xdsl_text"])

    def test_xdsl_op_map_populated(self):
        def fn(x: int) -> int:
            y = x + 1
            return y
        ir = compile_function_to_ir(fn)
        self.assertGreater(len(ir.xdsl_op_map), 0)
        # Every op in ir.ops should have an entry in xdsl_op_map
        for op_id in ir.ops:
            self.assertIn(op_id, ir.xdsl_op_map,
                         f"Op {op_id} missing from xdsl_op_map")

    def test_xdsl_block_map_populated(self):
        def fn(x: int) -> int:
            if x > 0:
                return x
            return 0
        ir = compile_function_to_ir(fn)
        self.assertGreater(len(ir.xdsl_block_map), 0)
        for block_id in ir.blocks:
            self.assertIn(block_id, ir.xdsl_block_map,
                         f"Block {block_id} missing from xdsl_block_map")

    def test_clone_creates_new_xdsl_module(self):
        def fn(x: int) -> int:
            return x + 1
        ir = compile_function_to_ir(fn)
        cloned = ir.clone()
        self.assertIsNot(ir.xdsl_module, cloned.xdsl_module)
        # Both should render to the same text
        orig_text = render_xdsl_module(ir.xdsl_module)
        clone_text = render_xdsl_module(cloned.xdsl_module)
        self.assertEqual(orig_text, clone_text)

    def test_xdsl_render_roundtrip(self):
        """Rebuilding from xDSL should preserve the IR structure."""
        def fn(a: int, b: int) -> int:
            return a + b
        ir = compile_function_to_ir(fn)
        from algorithm_ir.ir.model import FunctionIR
        rebuilt = FunctionIR.from_xdsl(ir.xdsl_module)
        self.assertEqual(len(ir.ops), len(rebuilt.ops))
        self.assertEqual(len(ir.blocks), len(rebuilt.blocks))
        self.assertEqual(len(ir.values), len(rebuilt.values))
        self.assertEqual(ir.name, rebuilt.name)

    def test_callable_serialization_in_payload(self):
        """Test that callable objects survive xDSL payload roundtrip."""
        from algorithm_ir.ir.xdsl_bridge import _normalize_payload, _denormalize_payload
        import builtins
        payload = {"fn": len, "value": 42, "name": "test"}
        normalized = _normalize_payload(payload)
        self.assertIsInstance(normalized["fn"], dict)
        self.assertIn("__callable__", normalized["fn"])
        denormalized = _denormalize_payload(normalized)
        self.assertEqual(denormalized["fn"], len)


class TestXDSLTypeMapping(unittest.TestCase):
    """Test type mapping between Python types and xDSL types."""

    def test_int_maps_to_i64(self):
        def fn(x: int) -> int:
            return x
        ir = compile_function_to_ir(fn)
        xdsl_text = ir.attrs["xdsl_text"]
        self.assertIn("i64", xdsl_text)

    def test_float_maps_to_f64(self):
        def fn(x: float) -> float:
            return x
        ir = compile_function_to_ir(fn)
        xdsl_text = ir.attrs["xdsl_text"]
        self.assertIn("f64", xdsl_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
