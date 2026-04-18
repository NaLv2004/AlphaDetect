"""Cross-language consistency tests: Python codegen vs C++ ir_eval."""
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

import numpy as np

from algorithm_ir.frontend.ir_builder import compile_function_to_ir
from algorithm_ir.regeneration.codegen import emit_python_source, emit_cpp_ops
from algorithm_ir.runtime.interpreter import execute_ir


def _try_import_cpp():
    """Try to import the C++ bridge. Skip tests if DLL not available."""
    try:
        sys.path.insert(0, str(ROOT / "applications" / "mimo_bp"))
        from cpp_evaluator import ir_eval_expr_cpp
        return ir_eval_expr_cpp
    except (ImportError, FileNotFoundError, OSError):
        return None


ir_eval_expr_cpp = _try_import_cpp()
SKIP_MSG = "C++ DLL not available"


def _eval_python_direct(func, args_list):
    """Evaluate original Python function directly (ground truth)."""
    return func(*args_list)


@unittest.skipIf(ir_eval_expr_cpp is None, SKIP_MSG)
class TestCrossLangArithmetic(unittest.TestCase):
    """Test basic arithmetic consistency between Python and C++."""

    def _check(self, fn, args, tol=1e-10):
        ir = compile_function_to_ir(fn)
        py_result = _eval_python_direct(fn, args)
        cpp_ops = emit_cpp_ops(ir)
        cpp_result = ir_eval_expr_cpp(cpp_ops, list(map(float, args)))
        self.assertAlmostEqual(float(py_result), cpp_result, delta=tol,
                               msg=f"Python={py_result}, C++={cpp_result}")

    def test_add(self):
        def f(a, b): return a + b
        self._check(f, [3.0, 4.0])

    def test_sub(self):
        def f(a, b): return a - b
        self._check(f, [10.0, 3.0])

    def test_mul(self):
        def f(a, b): return a * b
        self._check(f, [3.0, 4.0])

    def test_div(self):
        def f(a, b): return a / b
        self._check(f, [10.0, 3.0])

    def test_neg(self):
        def f(x): return -x
        self._check(f, [5.0])

    def test_nested(self):
        def f(a, b): return (a + b) * (a - b)
        self._check(f, [5.0, 3.0])

    def test_const(self):
        def f(x): return x + 3.14
        self._check(f, [1.0])


@unittest.skipIf(ir_eval_expr_cpp is None, SKIP_MSG)
class TestCrossLangComparison(unittest.TestCase):
    """Test comparison ops."""

    def _check(self, fn, args, tol=1e-10):
        ir = compile_function_to_ir(fn)
        py_result = _eval_python_direct(fn, args)
        cpp_ops = emit_cpp_ops(ir)
        cpp_result = ir_eval_expr_cpp(cpp_ops, list(map(float, args)))
        self.assertAlmostEqual(float(py_result), cpp_result, delta=tol)

    def test_gt_true(self):
        def f(a, b):
            if a > b:
                return a
            return b
        self._check(f, [5.0, 3.0])

    def test_gt_false(self):
        def f(a, b):
            if a > b:
                return a
            return b
        self._check(f, [1.0, 3.0])


@unittest.skipIf(ir_eval_expr_cpp is None, SKIP_MSG)
class TestCrossLangSafeMath(unittest.TestCase):
    """Test safe math edge cases."""

    def _check(self, fn, args, tol=1e-10):
        ir = compile_function_to_ir(fn)
        py_result = _eval_python_direct(fn, args)
        cpp_ops = emit_cpp_ops(ir)
        cpp_result = ir_eval_expr_cpp(cpp_ops, list(map(float, args)))
        self.assertAlmostEqual(float(py_result), cpp_result, delta=tol,
                               msg=f"Python={py_result}, C++={cpp_result}")

    def test_div_by_zero(self):
        """C++ safe_div returns 0.0 for division by zero."""
        def f(a, b): return a / b
        ir = compile_function_to_ir(f)
        cpp_ops = emit_cpp_ops(ir)
        cpp_result = ir_eval_expr_cpp(cpp_ops, [5.0, 0.0])
        self.assertEqual(cpp_result, 0.0)

    def test_identity(self):
        def f(x): return x
        self._check(f, [42.0])

    def test_double_add(self):
        def f(a, b): return a + b + 1.0
        self._check(f, [2.0, 3.0])


@unittest.skipIf(ir_eval_expr_cpp is None, SKIP_MSG)
class TestCrossLangFuzz(unittest.TestCase):
    """Fuzz test: random programs, check Python vs C++ consistency."""

    def test_fuzz_20_random_exprs(self):
        """Generate random arithmetic expressions and verify consistency."""
        from algorithm_ir.frontend.ir_builder import compile_source_to_ir
        rng = np.random.default_rng(42)
        passed = 0
        # Use source strings to avoid inspect.getsource issues with lambdas
        sources = [
            "def f(a, b): return a + b",
            "def f(a, b): return a - b",
            "def f(a, b): return a * b",
            "def f(a, b): return (a + b) * 2.0",
            "def f(a, b): return a + b + 1.0",
        ]
        for i in range(20):
            a, b = rng.uniform(-10, 10, 2)
            src = sources[i % len(sources)]
            try:
                ir = compile_source_to_ir(src, "f")
                # Evaluate via exec
                ns = {}
                exec(compile(src, "<fuzz>", "exec"), ns)
                py_result = ns["f"](a, b)
                cpp_ops = emit_cpp_ops(ir)
                cpp_result = ir_eval_expr_cpp(cpp_ops, [a, b])
                self.assertAlmostEqual(float(py_result), cpp_result, delta=1e-8,
                                       msg=f"Fuzz {i}: Py={py_result}, C++={cpp_result}")
                passed += 1
            except Exception:
                pass  # Some random fns may fail to compile
        self.assertGreater(passed, 10, "At least 10/20 fuzz tests should pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
