"""Unit tests for the alg IRDL dialect and AlgType."""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64
from xdsl.ir import Block as XBlock

from algorithm_ir.ir.dialect import (
    AlgAppend,
    AlgAssign,
    AlgBinary,
    AlgBranch,
    AlgBuildDict,
    AlgBuildList,
    AlgBuildTuple,
    AlgCall,
    AlgCompare,
    AlgConst,
    AlgDialect,
    AlgGetAttr,
    AlgGetItem,
    AlgIterInit,
    AlgIterNext,
    AlgJump,
    AlgPhi,
    AlgPop,
    AlgReturn,
    AlgSetAttr,
    AlgSetItem,
    AlgSlot,
    AlgUnary,
)
from algorithm_ir.ir.types import AlgType


def _block(*n_args):
    return XBlock(arg_types=[AlgType() for _ in range(n_args[0] if n_args else 2)])


class TestAlgType(unittest.TestCase):
    def test_type_name(self):
        self.assertEqual(AlgType.name, "alg.type")

    def test_block_arg_type(self):
        block = _block(1)
        self.assertIsInstance(block.args[0].type, AlgType)


class TestAlgConst(unittest.TestCase):
    def test_build_with_value(self):
        op = AlgConst.build(
            result_types=[AlgType()],
            attributes={"value": StringAttr("42"), "type_hint": StringAttr("int")},
        )
        self.assertEqual(op.value.data, "42")
        self.assertEqual(op.type_hint.data, "int")

    def test_build_without_type_hint(self):
        op = AlgConst.build(
            result_types=[AlgType()],
            attributes={"value": StringAttr("None")},
        )
        self.assertIsNone(op.type_hint)


class TestAlgAssign(unittest.TestCase):
    def test_build(self):
        block = _block(1)
        op = AlgAssign.build(
            operands=[block.args[0]],
            result_types=[AlgType()],
            attributes={"var_name": StringAttr("x")},
        )
        self.assertEqual(op.var_name.data, "x")
        self.assertEqual(len(op.results), 1)


class TestAlgBinary(unittest.TestCase):
    def test_add(self):
        block = _block(2)
        op = AlgBinary.build(
            operands=[block.args[0], block.args[1]],
            result_types=[AlgType()],
            attributes={"operator": StringAttr("Add")},
        )
        self.assertEqual(op.operator.data, "Add")
        self.assertEqual(len(op.operands), 2)

    def test_ssa_connectivity(self):
        block = _block(2)
        op = AlgBinary.build(
            operands=[block.args[0], block.args[1]],
            result_types=[AlgType()],
            attributes={"operator": StringAttr("Mult")},
        )
        block.add_op(op)
        self.assertIs(op.lhs, block.args[0])
        self.assertIs(op.rhs, block.args[1])


class TestAlgUnary(unittest.TestCase):
    def test_usub(self):
        block = _block(1)
        op = AlgUnary.build(
            operands=[block.args[0]],
            result_types=[AlgType()],
            attributes={"operator": StringAttr("USub")},
        )
        self.assertEqual(op.operator.data, "USub")


class TestAlgCompare(unittest.TestCase):
    def test_chain(self):
        block = _block(3)
        op = AlgCompare.build(
            operands=[list(block.args)],
            result_types=[AlgType()],
            attributes={"operators": StringAttr("Lt,Lt")},
        )
        self.assertEqual(len(op.arguments), 3)
        self.assertEqual(op.operators.data, "Lt,Lt")


class TestAlgPhi(unittest.TestCase):
    def test_two_incoming(self):
        block = _block(2)
        op = AlgPhi.build(
            operands=[list(block.args)],
            result_types=[AlgType()],
            attributes={"sources": StringAttr("b0,b1"), "var_name": StringAttr("x")},
        )
        self.assertEqual(len(op.incoming), 2)
        self.assertEqual(op.var_name.data, "x")


class TestAlgCall(unittest.TestCase):
    def test_build(self):
        block = _block(3)
        op = AlgCall.build(
            operands=[list(block.args)],
            result_types=[AlgType()],
            attributes={"n_args": IntegerAttr(2, i64)},
        )
        self.assertEqual(len(op.callee_and_args), 3)
        self.assertEqual(op.n_args.value.data, 2)


class TestAlgGetAttr(unittest.TestCase):
    def test_build(self):
        block = _block(1)
        op = AlgGetAttr.build(
            operands=[block.args[0]],
            result_types=[AlgType()],
            attributes={"attr_name": StringAttr("depth")},
        )
        self.assertEqual(op.attr_name.data, "depth")


class TestAlgSetAttr(unittest.TestCase):
    def test_build(self):
        block = _block(2)
        op = AlgSetAttr.build(
            operands=[block.args[0], block.args[1]],
            attributes={"attr_name": StringAttr("x")},
        )
        self.assertEqual(len(op.operands), 2)
        self.assertEqual(len(op.results), 0)


class TestAlgGetItem(unittest.TestCase):
    def test_build(self):
        block = _block(2)
        op = AlgGetItem.build(
            operands=[block.args[0], block.args[1]],
            result_types=[AlgType()],
        )
        self.assertEqual(len(op.operands), 2)


class TestAlgSetItem(unittest.TestCase):
    def test_build(self):
        block = _block(3)
        op = AlgSetItem.build(
            operands=[block.args[0], block.args[1], block.args[2]],
        )
        self.assertEqual(len(op.operands), 3)
        self.assertEqual(len(op.results), 0)


class TestAlgBuildList(unittest.TestCase):
    def test_build(self):
        block = _block(3)
        op = AlgBuildList.build(
            operands=[list(block.args)],
            result_types=[AlgType()],
        )
        self.assertEqual(len(op.items), 3)


class TestAlgBuildTuple(unittest.TestCase):
    def test_build(self):
        block = _block(2)
        op = AlgBuildTuple.build(
            operands=[list(block.args)],
            result_types=[AlgType()],
        )
        self.assertEqual(len(op.items), 2)


class TestAlgBuildDict(unittest.TestCase):
    def test_build(self):
        block = _block(4)
        op = AlgBuildDict.build(
            operands=[list(block.args)],
            result_types=[AlgType()],
        )
        self.assertEqual(len(op.key_values), 4)


class TestAlgAppend(unittest.TestCase):
    def test_build(self):
        block = _block(2)
        op = AlgAppend.build(operands=[block.args[0], block.args[1]])
        self.assertEqual(len(op.operands), 2)
        self.assertEqual(len(op.results), 0)


class TestAlgPop(unittest.TestCase):
    def test_build(self):
        block = _block(2)
        op = AlgPop.build(
            operands=[block.args[0], block.args[1]],
            result_types=[AlgType()],
        )
        self.assertEqual(len(op.operands), 2)
        self.assertEqual(len(op.results), 1)


class TestAlgIterInit(unittest.TestCase):
    def test_build(self):
        block = _block(1)
        op = AlgIterInit.build(
            operands=[block.args[0]],
            result_types=[AlgType()],
        )
        self.assertEqual(len(op.operands), 1)


class TestAlgIterNext(unittest.TestCase):
    def test_build(self):
        block = _block(1)
        op = AlgIterNext.build(
            operands=[block.args[0]],
            result_types=[AlgType(), AlgType()],
        )
        self.assertEqual(len(op.results), 2)


class TestAlgBranch(unittest.TestCase):
    def test_build(self):
        block = _block(1)
        t = XBlock()
        f = XBlock()
        op = AlgBranch.build(operands=[block.args[0]], successors=[t, f])
        self.assertEqual(len(op.successors), 2)
        self.assertIs(op.true_block, t)
        self.assertIs(op.false_block, f)


class TestAlgJump(unittest.TestCase):
    def test_build(self):
        t = XBlock()
        op = AlgJump.build(successors=[t])
        self.assertEqual(len(op.successors), 1)
        self.assertIs(op.target, t)


class TestAlgReturn(unittest.TestCase):
    def test_build(self):
        block = _block(1)
        op = AlgReturn.build(operands=[block.args[0]])
        self.assertEqual(len(op.operands), 1)


class TestAlgSlot(unittest.TestCase):
    def test_build_with_inputs(self):
        block = _block(3)
        op = AlgSlot.build(
            operands=[list(block.args)],
            result_types=[AlgType()],
            attributes={
                "slot_id": StringAttr("slot_score_0"),
                "slot_kind": StringAttr("score"),
            },
        )
        self.assertEqual(len(op.slot_inputs), 3)
        self.assertEqual(op.slot_id.data, "slot_score_0")
        self.assertEqual(op.slot_kind.data, "score")

    def test_build_without_kind(self):
        block = _block(1)
        op = AlgSlot.build(
            operands=[[block.args[0]]],
            result_types=[AlgType()],
            attributes={"slot_id": StringAttr("slot_0")},
        )
        self.assertIsNone(op.slot_kind)

    def test_empty_inputs(self):
        op = AlgSlot.build(
            operands=[[]],
            result_types=[AlgType()],
            attributes={"slot_id": StringAttr("slot_empty")},
        )
        self.assertEqual(len(op.slot_inputs), 0)


class TestAlgDialect(unittest.TestCase):
    def test_op_count(self):
        ops = list(AlgDialect.operations)
        self.assertEqual(len(ops), 22)

    def test_contains_all_ops(self):
        op_names = {op.name for op in AlgDialect.operations}
        expected = {
            "alg.const", "alg.assign", "alg.binary", "alg.unary",
            "alg.compare", "alg.phi", "alg.call",
            "alg.get_attr", "alg.set_attr",
            "alg.get_item", "alg.set_item",
            "alg.build_list", "alg.build_tuple", "alg.build_dict",
            "alg.append", "alg.pop",
            "alg.iter_init", "alg.iter_next",
            "alg.branch", "alg.jump", "alg.return",
            "alg.slot",
        }
        self.assertEqual(op_names, expected)


class TestSSAConnectivity(unittest.TestCase):
    """Verify that xDSL SSA works properly with our ops."""

    def test_result_used_as_operand(self):
        block = _block(0)
        c1 = AlgConst.build(
            result_types=[AlgType()],
            attributes={"value": StringAttr("1")},
        )
        c2 = AlgConst.build(
            result_types=[AlgType()],
            attributes={"value": StringAttr("2")},
        )
        add = AlgBinary.build(
            operands=[c1.res, c2.res],
            result_types=[AlgType()],
            attributes={"operator": StringAttr("Add")},
        )
        block.add_op(c1)
        block.add_op(c2)
        block.add_op(add)
        self.assertIs(add.lhs, c1.res)
        self.assertIs(add.rhs, c2.res)
        self.assertEqual(len(list(c1.res.uses)), 1)

    def test_multi_result_iter_next(self):
        block = _block(1)
        init = AlgIterInit.build(
            operands=[block.args[0]],
            result_types=[AlgType()],
        )
        block.add_op(init)
        nxt = AlgIterNext.build(
            operands=[init.res],
            result_types=[AlgType(), AlgType()],
        )
        block.add_op(nxt)
        self.assertIs(nxt.iterator, init.res)
        self.assertEqual(len(nxt.results), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
