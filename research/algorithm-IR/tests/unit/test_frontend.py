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

from algorithm_ir.frontend import compile_function_to_ir
from algorithm_ir.ir import render_function_ir, validate_function_ir
from examples.algorithms import simple_branch_loop, stack_decoder_host


class FrontendTests(unittest.TestCase):
    def test_compile_simple_branch_loop(self) -> None:
        func_ir = compile_function_to_ir(simple_branch_loop)
        errors = validate_function_ir(func_ir)
        self.assertEqual(errors, [], msg="\n".join(errors))
        opcodes = {op.opcode for op in func_ir.ops.values()}
        self.assertIn("branch", opcodes)
        self.assertIn("return", opcodes)
        self.assertGreaterEqual(len(func_ir.blocks), 4)

    def test_compile_stack_decoder_host(self) -> None:
        func_ir = compile_function_to_ir(stack_decoder_host)
        errors = validate_function_ir(func_ir)
        self.assertEqual(errors, [], msg="\n".join(errors))
        rendered = render_function_ir(func_ir)
        self.assertIn("build_dict", rendered)
        self.assertIn("call", rendered)
        self.assertIn("get_item", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
