"""
alg Dialect — IRDL-based operation definitions for Algorithm IR.

Every op that the frontend emits or the grafting system manipulates is
defined here as a proper xDSL IRDLOperation so that xDSL SSA, printing,
cloning, and pattern rewriting all work natively.
"""
from __future__ import annotations

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.ir import Dialect
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    successor_def,
    var_operand_def,
)

from algorithm_ir.ir.types import AlgType


# ---------------------------------------------------------------------------
# Value-producing ops
# ---------------------------------------------------------------------------

@irdl_op_definition
class AlgConst(IRDLOperation):
    """Load a compile-time constant (int, float, str, callable, None, …)."""
    name = "alg.const"
    res = result_def(AnyAttr())
    value = attr_def(StringAttr)          # repr() of the Python literal
    type_hint = opt_attr_def(StringAttr)  # e.g. "int", "float", "function"


@irdl_op_definition
class AlgAssign(IRDLOperation):
    """SSA rename: ``target = source``."""
    name = "alg.assign"
    source = operand_def(AnyAttr())
    res = result_def(AnyAttr())
    var_name = opt_attr_def(StringAttr)


@irdl_op_definition
class AlgBinary(IRDLOperation):
    """Binary arithmetic/logic: ``lhs op rhs``."""
    name = "alg.binary"
    lhs = operand_def(AnyAttr())
    rhs = operand_def(AnyAttr())
    res = result_def(AnyAttr())
    operator = attr_def(StringAttr)       # "Add", "Sub", "Mult", …


@irdl_op_definition
class AlgUnary(IRDLOperation):
    """Unary operation: ``op operand``."""
    name = "alg.unary"
    operand = operand_def(AnyAttr())
    res = result_def(AnyAttr())
    operator = attr_def(StringAttr)       # "USub", "UAdd", "Not"


@irdl_op_definition
class AlgCompare(IRDLOperation):
    """Chained comparison: ``a < b < c`` → operators=["Lt","Lt"]."""
    name = "alg.compare"
    arguments = var_operand_def(AnyAttr())   # [left, *comparators]
    res = result_def(AnyAttr())
    operators = attr_def(StringAttr)         # comma-joined operator names


@irdl_op_definition
class AlgPhi(IRDLOperation):
    """SSA φ-node: select among incoming values."""
    name = "alg.phi"
    incoming = var_operand_def(AnyAttr())
    res = result_def(AnyAttr())
    sources = attr_def(StringAttr)           # comma-joined block ids
    var_name = opt_attr_def(StringAttr)


@irdl_op_definition
class AlgCall(IRDLOperation):
    """Function call: ``callee(arg0, arg1, …)``."""
    name = "alg.call"
    callee_and_args = var_operand_def(AnyAttr())  # [callee, *args]
    res = result_def(AnyAttr())
    n_args = attr_def(IntegerAttr)


@irdl_op_definition
class AlgGetAttr(IRDLOperation):
    """Attribute read: ``owner.attr``."""
    name = "alg.get_attr"
    owner = operand_def(AnyAttr())
    res = result_def(AnyAttr())
    attr_name = attr_def(StringAttr)


@irdl_op_definition
class AlgSetAttr(IRDLOperation):
    """Attribute write: ``owner.attr = value``."""
    name = "alg.set_attr"
    owner = operand_def(AnyAttr())
    value = operand_def(AnyAttr())
    attr_name = attr_def(StringAttr)


@irdl_op_definition
class AlgGetItem(IRDLOperation):
    """Subscript read: ``owner[index]``."""
    name = "alg.get_item"
    owner = operand_def(AnyAttr())
    index = operand_def(AnyAttr())
    res = result_def(AnyAttr())


@irdl_op_definition
class AlgSetItem(IRDLOperation):
    """Subscript write: ``owner[index] = value``."""
    name = "alg.set_item"
    owner = operand_def(AnyAttr())
    index = operand_def(AnyAttr())
    value = operand_def(AnyAttr())


@irdl_op_definition
class AlgBuildList(IRDLOperation):
    """Construct a list: ``[item0, item1, …]``."""
    name = "alg.build_list"
    items = var_operand_def(AnyAttr())
    res = result_def(AnyAttr())


@irdl_op_definition
class AlgBuildTuple(IRDLOperation):
    """Construct a tuple: ``(item0, item1, …)``."""
    name = "alg.build_tuple"
    items = var_operand_def(AnyAttr())
    res = result_def(AnyAttr())


@irdl_op_definition
class AlgBuildDict(IRDLOperation):
    """Construct a dict: ``{k0: v0, k1: v1, …}`` — items=[k0,v0,k1,v1,…]."""
    name = "alg.build_dict"
    key_values = var_operand_def(AnyAttr())
    res = result_def(AnyAttr())


@irdl_op_definition
class AlgAppend(IRDLOperation):
    """In-place list append: ``container.append(value)``."""
    name = "alg.append"
    container = operand_def(AnyAttr())
    value = operand_def(AnyAttr())


@irdl_op_definition
class AlgPop(IRDLOperation):
    """List pop: ``container.pop(index)``."""
    name = "alg.pop"
    container = operand_def(AnyAttr())
    index = operand_def(AnyAttr())
    res = result_def(AnyAttr())


@irdl_op_definition
class AlgIterInit(IRDLOperation):
    """Create iterator from iterable."""
    name = "alg.iter_init"
    iterable = operand_def(AnyAttr())
    res = result_def(AnyAttr())


@irdl_op_definition
class AlgIterNext(IRDLOperation):
    """Advance iterator: produces (next_value, has_next)."""
    name = "alg.iter_next"
    iterator = operand_def(AnyAttr())
    next_val = result_def(AnyAttr())
    has_next = result_def(AnyAttr())


# ---------------------------------------------------------------------------
# Control-flow terminators
# ---------------------------------------------------------------------------

@irdl_op_definition
class AlgBranch(IRDLOperation):
    """Conditional branch."""
    name = "alg.branch"
    cond = operand_def(AnyAttr())
    true_block = successor_def()
    false_block = successor_def()


@irdl_op_definition
class AlgJump(IRDLOperation):
    """Unconditional jump."""
    name = "alg.jump"
    target = successor_def()


@irdl_op_definition
class AlgReturn(IRDLOperation):
    """Return from function."""
    name = "alg.return"
    value = operand_def(AnyAttr())


# ---------------------------------------------------------------------------
# Skeleton slot (GP-ready placeholder)
# ---------------------------------------------------------------------------

@irdl_op_definition
class AlgSlot(IRDLOperation):
    """
    A placeholder slot in the IR that can be filled with a skeleton fragment.

    During skeleton grafting, a region's ops are replaced by an AlgSlot.
    ``fill_slot()`` (backed by xDSL PatternRewriter) then replaces the
    slot with the skeleton ops, reconnecting SSA automatically.

    For GP:
      - crossover = extract slot content from donor, fill into host slot
      - mutation  = replace slot content with a mutated variant

    ``slot_inputs`` provides the SSA values available to the skeleton.
    ``res`` is the single output fed to downstream consumers.
    ``slot_id`` uniquely identifies this slot for pattern matching.
    ``slot_kind`` optionally categorizes the slot (e.g. "score", "schedule").
    """
    name = "alg.slot"
    slot_inputs = var_operand_def(AnyAttr())
    res = result_def(AnyAttr())
    slot_id = attr_def(StringAttr)
    slot_kind = opt_attr_def(StringAttr)


# ---------------------------------------------------------------------------
# Dialect bundle
# ---------------------------------------------------------------------------

AlgDialect = Dialect(
    "alg",
    [
        AlgConst,
        AlgAssign,
        AlgBinary,
        AlgUnary,
        AlgCompare,
        AlgPhi,
        AlgCall,
        AlgGetAttr,
        AlgSetAttr,
        AlgGetItem,
        AlgSetItem,
        AlgBuildList,
        AlgBuildTuple,
        AlgBuildDict,
        AlgAppend,
        AlgPop,
        AlgIterInit,
        AlgIterNext,
        AlgBranch,
        AlgJump,
        AlgReturn,
        AlgSlot,
    ],
    [],
)
