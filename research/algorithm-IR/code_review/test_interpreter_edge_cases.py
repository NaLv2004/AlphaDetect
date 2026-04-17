"""
Test the interpreter's correctness boundary and edge cases.
Focus: Does execute_ir faithfully reproduce Python semantics?
"""
from __future__ import annotations
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import unittest
from algorithm_ir.frontend import compile_function_to_ir
from algorithm_ir.runtime import execute_ir


class TestInterpreterCorrectness(unittest.TestCase):
    """Verify IR execution matches Python for non-trivial programs."""

    def test_fibonacci(self):
        def fib(n: int) -> int:
            a = 0
            b = 1
            i = 0
            while i < n:
                temp = a + b
                a = b
                b = temp
                i = i + 1
            return a
        ir = compile_function_to_ir(fib)
        for n in [0, 1, 5, 10]:
            result, _, _ = execute_ir(ir, [n])
            self.assertEqual(result, fib(n), f"Fibonacci failed for n={n}")

    def test_bubble_sort(self):
        def bubble(lst: list) -> list:
            n = len(lst)
            i = 0
            while i < n:
                j = 0
                while j < n - 1 - i:
                    if lst[j] > lst[j + 1]:
                        temp = lst[j]
                        lst[j] = lst[j + 1]
                        lst[j + 1] = temp
                    j = j + 1
                i = i + 1
            return lst
        ir = compile_function_to_ir(bubble)
        test_input = [4, 2, 7, 1, 3]
        result, _, _ = execute_ir(ir, [list(test_input)])
        self.assertEqual(result, sorted(test_input))

    def test_gcd(self):
        def gcd(a: int, b: int) -> int:
            while b > 0:
                temp = b
                b = a % b
                a = temp
            return a
        ir = compile_function_to_ir(gcd)
        result, _, _ = execute_ir(ir, [48, 18])
        self.assertEqual(result, gcd(48, 18))

    def test_multiple_returns(self):
        def fn(x: int) -> int:
            if x > 10:
                return 1
            if x > 5:
                return 2
            if x > 0:
                return 3
            return 4
        ir = compile_function_to_ir(fn)
        for val, expected in [(15, 1), (7, 2), (3, 3), (-1, 4)]:
            result, _, _ = execute_ir(ir, [val])
            self.assertEqual(result, expected, f"Failed for x={val}")

    def test_nested_dict_mutation(self):
        def fn() -> int:
            d = {"a": 1, "b": 2}
            d["a"] = d["a"] + d["b"]
            d["b"] = d["a"] * 2
            return d["a"] + d["b"]
        ir = compile_function_to_ir(fn)
        result, _, _ = execute_ir(ir, [])
        self.assertEqual(result, fn())

    def test_list_of_dicts(self):
        """Test operating on list of dicts — the core data pattern in stack_decoder."""
        def fn() -> float:
            items = []
            items.append({"val": 1.0})
            items.append({"val": 2.0})
            total = 0.0
            i = 0
            while i < len(items):
                total = total + items[i]["val"]
                i = i + 1
            return total
        ir = compile_function_to_ir(fn)
        result, _, _ = execute_ir(ir, [])
        self.assertEqual(result, 3.0)


class TestTraceQuality(unittest.TestCase):
    """Test that runtime traces capture meaningful information."""

    def test_trace_has_all_events(self):
        def fn(x: int) -> int:
            y = x + 1
            z = y * 2
            return z
        ir = compile_function_to_ir(fn)
        _, trace, _ = execute_ir(ir, [5])
        opcodes_executed = [
            ir.ops[event.static_op_id].opcode
            for event in trace
            if event.static_op_id in ir.ops
        ]
        self.assertIn("binary", opcodes_executed)
        self.assertIn("return", opcodes_executed)

    def test_runtime_values_track_types(self):
        def fn(x: float) -> float:
            return x * 2.0
        ir = compile_function_to_ir(fn)
        _, _, rv = execute_ir(ir, [3.0])
        type_names = {v.metadata.get("type_name") for v in rv.values()}
        self.assertIn("float", type_names)

    def test_loop_trace_records_iterations(self):
        def fn(n: int) -> int:
            s = 0
            i = 0
            while i < n:
                s = s + i
                i = i + 1
            return s
        ir = compile_function_to_ir(fn)
        _, trace, _ = execute_ir(ir, [3])
        # Should have multiple branch events from the while loop
        branch_events = [e for e in trace if ir.ops.get(e.static_op_id, None) and ir.ops[e.static_op_id].opcode == "branch"]
        self.assertGreater(len(branch_events), 1)

    def test_shadow_store_tracks_mutations(self):
        def fn() -> int:
            d = {"x": 0}
            d["x"] = 1
            d["x"] = 2
            return d["x"]
        ir = compile_function_to_ir(fn)
        _, _, rv = execute_ir(ir, [])
        # Check that some runtime values have shadow_store metadata
        has_shadow = any(
            "shadow_store" in v.metadata
            for v in rv.values()
        )
        self.assertTrue(has_shadow, "Shadow store metadata should be present")


class TestEmptyAndDegenerate(unittest.TestCase):
    """Test edge cases: empty loops, no-op functions, etc."""

    def test_empty_body_returns_none(self):
        def fn() -> None:
            x = 0
            return None
        ir = compile_function_to_ir(fn)
        result, _, _ = execute_ir(ir, [])
        self.assertIsNone(result)

    def test_zero_iteration_loop(self):
        def fn() -> int:
            s = 0
            i = 0
            while i < 0:
                s = s + 1
                i = i + 1
            return s
        ir = compile_function_to_ir(fn)
        result, _, _ = execute_ir(ir, [])
        self.assertEqual(result, 0)

    def test_single_constant_return(self):
        def fn() -> int:
            return 42
        ir = compile_function_to_ir(fn)
        result, _, _ = execute_ir(ir, [])
        self.assertEqual(result, 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
