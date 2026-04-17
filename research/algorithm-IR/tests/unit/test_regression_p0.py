"""
Regression tests for P0 bugs fixed in Phase 2:
  1. for-loop phi nodes (variables modified in loop body)
  2. Function parameter type annotations
  3. Module-level attribute access (e.g. math.sqrt)
  4. Clone roundtrip with callables
"""
import math
import unittest

from algorithm_ir.frontend.ir_builder import compile_function_to_ir
from algorithm_ir.ir.validator import validate_function_ir
from algorithm_ir.runtime.interpreter import execute_ir


# ---- Test functions ----

def for_sum(items: list) -> int:
    total = 0
    for x in items:
        total = total + x
    return total


def for_nested(matrix: list) -> int:
    total = 0
    for row in matrix:
        for x in row:
            total = total + x
    return total


def for_with_branch(items: list) -> int:
    pos = 0
    neg = 0
    for x in items:
        if x > 0:
            pos = pos + x
        else:
            neg = neg + x
    return pos + neg


def typed_fn(x: int, y: float) -> float:
    return x + y


def typed_bool(flag: bool) -> bool:
    return flag


def typed_str(name: str) -> str:
    return name


def use_math_sqrt(x: float) -> float:
    return math.sqrt(x)


def use_math_floor(x: float) -> float:
    return math.floor(x) + math.ceil(x)


def use_len(items: list) -> int:
    return len(items)


def use_range(n: int) -> int:
    total = 0
    for i in range(n):
        total = total + i
    return total


class TestForLoopPhiFix(unittest.TestCase):
    """Regression: for-loop now creates phi nodes for modified variables."""

    def test_for_sum_basic(self):
        ir = compile_function_to_ir(for_sum)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [[1, 2, 3, 4, 5]])
        self.assertEqual(result, 15)

    def test_for_sum_empty(self):
        ir = compile_function_to_ir(for_sum)
        result, _, _ = execute_ir(ir, [[]])
        self.assertEqual(result, 0)

    def test_for_sum_single(self):
        ir = compile_function_to_ir(for_sum)
        result, _, _ = execute_ir(ir, [[42]])
        self.assertEqual(result, 42)

    def test_for_nested(self):
        ir = compile_function_to_ir(for_nested)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [[[1, 2], [3, 4]]])
        self.assertEqual(result, 10)

    def test_for_with_branch(self):
        ir = compile_function_to_ir(for_with_branch)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [[3, -1, 2, -4]])
        self.assertEqual(result, 0)

    def test_for_phi_exists(self):
        """Verify phi ops exist in the for-loop test block."""
        ir = compile_function_to_ir(for_sum)
        phi_ops = [op for op in ir.ops.values() if op.opcode == "phi"]
        self.assertGreater(len(phi_ops), 0, "for-loop should generate phi ops")


class TestTypeAnnotations(unittest.TestCase):
    """Regression: function arg type annotations are now parsed."""

    def test_int_float_annotations(self):
        ir = compile_function_to_ir(typed_fn)
        arg_types = {ir.values[v].name_hint: ir.values[v].type_hint for v in ir.arg_values}
        self.assertEqual(arg_types.get("x"), "int")
        self.assertEqual(arg_types.get("y"), "float")

    def test_bool_annotation(self):
        ir = compile_function_to_ir(typed_bool)
        arg_types = {ir.values[v].name_hint: ir.values[v].type_hint for v in ir.arg_values}
        self.assertEqual(arg_types.get("flag"), "bool")

    def test_str_annotation(self):
        ir = compile_function_to_ir(typed_str)
        arg_types = {ir.values[v].name_hint: ir.values[v].type_hint for v in ir.arg_values}
        self.assertEqual(arg_types.get("name"), "str")

    def test_list_annotation(self):
        ir = compile_function_to_ir(for_sum)
        arg_types = {ir.values[v].name_hint: ir.values[v].type_hint for v in ir.arg_values}
        self.assertEqual(arg_types.get("items"), "list")

    def test_execution_with_types(self):
        ir = compile_function_to_ir(typed_fn)
        result, _, _ = execute_ir(ir, [3, 2.5])
        self.assertEqual(result, 5.5)


class TestModuleAccess(unittest.TestCase):
    """Regression: module-level attribute access now works."""

    def test_math_sqrt(self):
        ir = compile_function_to_ir(use_math_sqrt)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [9.0])
        self.assertAlmostEqual(result, 3.0)

    def test_math_floor_ceil(self):
        ir = compile_function_to_ir(use_math_floor)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [3.7])
        self.assertEqual(result, math.floor(3.7) + math.ceil(3.7))

    def test_builtin_len(self):
        ir = compile_function_to_ir(use_len)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [[1, 2, 3]])
        self.assertEqual(result, 3)

    def test_builtin_range(self):
        ir = compile_function_to_ir(use_range)
        self.assertEqual(validate_function_ir(ir), [])
        result, _, _ = execute_ir(ir, [5])
        self.assertEqual(result, 10)  # 0+1+2+3+4

    def test_module_const_type_hint(self):
        """Module objects should get type_hint='module'."""
        ir = compile_function_to_ir(use_math_sqrt)
        module_values = [v for v in ir.values.values() if v.type_hint == "module"]
        self.assertGreater(len(module_values), 0)


class TestCloneRoundtrip(unittest.TestCase):
    """Verify clone() preserves callables and state correctly."""

    def test_clone_preserves_execution(self):
        ir = compile_function_to_ir(use_len)
        cloned = ir.clone()
        self.assertEqual(validate_function_ir(cloned), [])
        result, _, _ = execute_ir(cloned, [[10, 20, 30]])
        self.assertEqual(result, 3)

    def test_clone_preserves_math(self):
        ir = compile_function_to_ir(use_math_sqrt)
        cloned = ir.clone()
        self.assertEqual(validate_function_ir(cloned), [])
        result, _, _ = execute_ir(cloned, [16.0])
        self.assertAlmostEqual(result, 4.0)

    def test_clone_preserves_for_loop(self):
        ir = compile_function_to_ir(for_sum)
        cloned = ir.clone()
        self.assertEqual(validate_function_ir(cloned), [])
        result, _, _ = execute_ir(cloned, [[1, 2, 3]])
        self.assertEqual(result, 6)

    def test_clone_preserves_types(self):
        ir = compile_function_to_ir(typed_fn)
        cloned = ir.clone()
        arg_types = {cloned.values[v].name_hint: cloned.values[v].type_hint for v in cloned.arg_values}
        self.assertEqual(arg_types.get("x"), "int")
        self.assertEqual(arg_types.get("y"), "float")


if __name__ == "__main__":
    unittest.main()
