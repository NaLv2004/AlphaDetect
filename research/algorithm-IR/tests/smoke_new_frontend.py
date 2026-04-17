"""Quick smoke test for the new frontend."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithm_ir.frontend.ir_builder import compile_function_to_ir
from algorithm_ir.ir.validator import validate_function_ir
from algorithm_ir.runtime.interpreter import execute_ir
from tests.examples.algorithms import simple_branch_loop, complex_tuple_kernel, stack_decoder_host
import math

def for_sum(items: list) -> int:
    total = 0
    for x in items:
        total = total + x
    return total

def typed_fn(x: int, y: float) -> float:
    return x + y

def use_math(x: float) -> float:
    return math.sqrt(x) + math.floor(x)

tests_passed = 0
tests_failed = 0

def check(name, ir, args, expected):
    global tests_passed, tests_failed
    errors = validate_function_ir(ir)
    if errors:
        print(f"FAIL {name}: validation errors: {errors}")
        tests_failed += 1
        return
    result, trace, rvs = execute_ir(ir, args)
    if result == expected:
        print(f"PASS {name}: result={result}")
        tests_passed += 1
    else:
        print(f"FAIL {name}: result={result}, expected={expected}")
        tests_failed += 1

# Test 1: simple_branch_loop
ir1 = compile_function_to_ir(simple_branch_loop)
check("simple_branch_loop(5)", ir1, [5], simple_branch_loop(5))

# Test 2: complex_tuple_kernel
ir2 = compile_function_to_ir(complex_tuple_kernel)
check("complex_tuple_kernel(3.0)", ir2, [3.0], complex_tuple_kernel(3.0))

# Test 3: stack_decoder_host
ir3 = compile_function_to_ir(stack_decoder_host)
check("stack_decoder_host", ir3, [[0.5, 1.0, 0.3], 3], stack_decoder_host([0.5, 1.0, 0.3], 3))

# Test 4: for loop
ir4 = compile_function_to_ir(for_sum)
check("for_sum([1,2,3,4])", ir4, [[1,2,3,4]], 10)

# Test 5: type annotations
ir5 = compile_function_to_ir(typed_fn)
arg_types = [(ir5.values[v].name_hint, ir5.values[v].type_hint) for v in ir5.arg_values]
print(f"typed_fn arg types: {arg_types}")
assert arg_types[0][1] == "int", f"Expected int, got {arg_types[0][1]}"
assert arg_types[1][1] == "float", f"Expected float, got {arg_types[1][1]}"
check("typed_fn(3, 2.5)", ir5, [3, 2.5], 5.5)

# Test 6: math module
ir6 = compile_function_to_ir(use_math)
check("use_math(9.0)", ir6, [9.0], math.sqrt(9.0) + math.floor(9.0))

print(f"\n{tests_passed} passed, {tests_failed} failed")
if tests_failed > 0:
    sys.exit(1)
