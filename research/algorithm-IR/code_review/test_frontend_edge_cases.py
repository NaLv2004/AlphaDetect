"""
Test the frontend's ability to compile various Python constructs.
Focus: What Python features are actually supported vs claimed?
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


class TestBasicArithmetic(unittest.TestCase):
    """Test basic arithmetic and constant propagation."""

    def test_integer_arithmetic(self):
        def fn(a: int, b: int) -> int:
            return a + b * 2 - 1
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [10, 3])
        self.assertEqual(result, fn(10, 3))

    def test_float_arithmetic(self):
        def fn(x: float) -> float:
            return x * 0.5 + 1.0
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [3.0])
        self.assertAlmostEqual(result, fn(3.0))

    def test_floor_div_and_mod(self):
        def fn(a: int, b: int) -> int:
            q = a // b
            r = a % b
            return q + r
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [17, 5])
        self.assertEqual(result, fn(17, 5))

    def test_unary_negation(self):
        def fn(x: int) -> int:
            return -x
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [42])
        self.assertEqual(result, fn(42))

    def test_complex_arithmetic(self):
        """Test complex number support (claimed in ir_plan.md)."""
        def fn(x: float) -> complex:
            c = 1.0 + 2.0j
            return c * x
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [3.0])
        self.assertEqual(result, fn(3.0))


class TestControlFlow(unittest.TestCase):
    """Test control flow constructs."""

    def test_simple_if_else(self):
        def fn(x: int) -> int:
            if x > 0:
                y = x
            else:
                y = -x
            return y
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        for val in [-3, 0, 5]:
            result, _, _ = execute_ir(ir, [val])
            self.assertEqual(result, fn(val), f"Failed for x={val}")

    def test_nested_if(self):
        def fn(x: int) -> int:
            if x > 10:
                if x > 20:
                    y = 3
                else:
                    y = 2
            else:
                y = 1
            return y
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        for val, expected in [(5, 1), (15, 2), (25, 3)]:
            result, _, _ = execute_ir(ir, [val])
            self.assertEqual(result, expected, f"Failed for x={val}")

    def test_while_loop(self):
        def fn(n: int) -> int:
            s = 0
            i = 0
            while i < n:
                s = s + i
                i = i + 1
            return s
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [5])
        self.assertEqual(result, fn(5))

    def test_for_loop(self):
        def fn(items: list) -> int:
            s = 0
            for item in items:
                s = s + item
            return s
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [[1, 2, 3, 4]])
        self.assertEqual(result, fn([1, 2, 3, 4]))

    def test_nested_while(self):
        def fn(n: int) -> int:
            total = 0
            i = 0
            while i < n:
                j = 0
                while j < i:
                    total = total + 1
                    j = j + 1
                i = i + 1
            return total
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [4])
        self.assertEqual(result, fn(4))

    def test_early_return_in_loop(self):
        def fn(x: int) -> int:
            i = 0
            while i < 100:
                if i == x:
                    return i * 2
                i = i + 1
            return -1
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [5])
        self.assertEqual(result, fn(5))

    def test_if_without_else(self):
        """Test if without else — phi merge must still work."""
        def fn(x: int) -> int:
            y = 0
            if x > 0:
                y = x
            return y
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        for val in [-1, 0, 5]:
            result, _, _ = execute_ir(ir, [val])
            self.assertEqual(result, fn(val), f"Failed for x={val}")


class TestDataStructures(unittest.TestCase):
    """Test list/dict/tuple operations."""

    def test_list_build_and_index(self):
        def fn(a: int, b: int) -> int:
            lst = [a, b, a + b]
            return lst[2]
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [3, 4])
        self.assertEqual(result, fn(3, 4))

    def test_list_append(self):
        """Test list.append — requires get_attr + call pattern."""
        def fn(n: int) -> int:
            lst = []
            i = 0
            while i < n:
                lst.append(i)
                i = i + 1
            return len(lst)
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [5])
        self.assertEqual(result, fn(5))

    def test_dict_build_and_access(self):
        def fn(x: int) -> int:
            d = {"a": x, "b": x + 1}
            return d["a"] + d["b"]
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [10])
        self.assertEqual(result, fn(10))

    def test_dict_set_item(self):
        def fn(x: int) -> int:
            d = {"val": 0}
            d["val"] = x * 2
            return d["val"]
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [5])
        self.assertEqual(result, fn(5))

    def test_tuple_build_and_index(self):
        def fn(a: int, b: int) -> int:
            t = (a, b)
            return t[0] + t[1]
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [3, 4])
        self.assertEqual(result, fn(3, 4))

    def test_nested_dict_access(self):
        def fn() -> int:
            d = {"inner": {"x": 42}}
            return d["inner"]["x"]
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [])
        self.assertEqual(result, fn())


class TestFunctionCalls(unittest.TestCase):
    """Test function call support."""

    def test_builtin_len(self):
        def fn(lst: list) -> int:
            return len(lst)
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [[1, 2, 3]])
        self.assertEqual(result, 3)

    def test_builtin_abs(self):
        def fn(x: int) -> int:
            return abs(x)
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [-5])
        self.assertEqual(result, 5)

    def test_external_function_call(self):
        """Test calling an external function referenced in globals."""
        import math
        def fn(x: float) -> float:
            return math.sqrt(x)
        ir = compile_function_to_ir(fn)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [9.0])
        self.assertAlmostEqual(result, 3.0)


class TestUnsupportedConstructs(unittest.TestCase):
    """Test that unsupported Python features raise appropriate errors."""

    def test_lambda_rejected(self):
        """Lambdas should not be compilable."""
        with self.assertRaises(Exception):
            compile_function_to_ir(lambda x: x + 1)

    def test_list_comprehension_rejected(self):
        """List comprehensions use ListComp AST node — should fail."""
        def fn(n: int) -> list:
            return [i * 2 for i in range(n)]
        with self.assertRaises(Exception):
            compile_function_to_ir(fn)

    def test_try_except_rejected(self):
        """Exception handling should not be supported."""
        def fn(x: int) -> int:
            try:
                return x // 0
            except:
                return -1
        with self.assertRaises(Exception):
            compile_function_to_ir(fn)

    def test_with_statement_rejected(self):
        def fn() -> int:
            with open("test.txt") as f:
                return 0
        with self.assertRaises(Exception):
            compile_function_to_ir(fn)

    def test_class_def_rejected(self):
        """Class definitions inside functions should fail."""
        def fn() -> int:
            class Foo:
                pass
            return 0
        with self.assertRaises(Exception):
            compile_function_to_ir(fn)

    def test_yield_rejected(self):
        """Generators should not be supported."""
        # Can't compile a generator as a regular function
        def fn(n: int):
            i = 0
            while i < n:
                yield i
                i = i + 1
        with self.assertRaises(Exception):
            compile_function_to_ir(fn)

    def test_starred_assignment_rejected(self):
        """Starred assignment (a, *b = ...) should fail."""
        def fn() -> int:
            a, *b = [1, 2, 3]
            return a
        with self.assertRaises(Exception):
            compile_function_to_ir(fn)


class TestTypeTracking(unittest.TestCase):
    """Test that the type inference / type_info system works correctly."""

    def test_int_types_propagated(self):
        def fn(x: int) -> int:
            y = x + 1
            return y
        ir = compile_function_to_ir(fn)
        # Check that binary op on ints produces int type
        binary_ops = [op for op in ir.ops.values() if op.opcode == "binary"]
        self.assertGreater(len(binary_ops), 0)
        for op in binary_ops:
            output_value = ir.values[op.outputs[0]]
            self.assertEqual(output_value.type_hint, "int")

    def test_float_types_propagated(self):
        def fn(x: float) -> float:
            y = x * 2.0
            return y
        ir = compile_function_to_ir(fn)
        binary_ops = [op for op in ir.ops.values() if op.opcode == "binary"]
        for op in binary_ops:
            output_value = ir.values[op.outputs[0]]
            # float * float or float * int → float
            self.assertIn(output_value.type_hint, ["float", "object"])

    def test_bool_comparison_type(self):
        def fn(x: int) -> bool:
            return x > 0
        ir = compile_function_to_ir(fn)
        compare_ops = [op for op in ir.ops.values() if op.opcode == "compare"]
        self.assertGreater(len(compare_ops), 0)
        for op in compare_ops:
            output_value = ir.values[op.outputs[0]]
            self.assertEqual(output_value.type_hint, "bool")


class TestCloneAndEquivalence(unittest.TestCase):
    """Test that FunctionIR.clone() produces an independent, equivalent copy."""

    def test_clone_produces_same_result(self):
        def fn(x: int) -> int:
            if x > 0:
                return x * 2
            return 0
        ir = compile_function_to_ir(fn)
        cloned = ir.clone()
        for val in [-1, 0, 5]:
            r1, _, _ = execute_ir(ir, [val])
            r2, _, _ = execute_ir(cloned, [val])
            self.assertEqual(r1, r2, f"Clone mismatch for x={val}")

    def test_clone_is_independent(self):
        """Modifying clone should not affect original."""
        def fn(x: int) -> int:
            return x + 1
        ir = compile_function_to_ir(fn)
        cloned = ir.clone()
        # Modify the clone's attrs
        cloned.attrs["test_marker"] = True
        self.assertNotIn("test_marker", ir.attrs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
