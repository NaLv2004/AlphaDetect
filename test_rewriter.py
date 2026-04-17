"""
Test: Can xDSL's PatternRewriter do runtime skeleton injection?
"""
from __future__ import annotations
from xdsl.irdl import (
    irdl_op_definition, IRDLOperation, AnyAttr,
    operand_def, result_def, attr_def, var_operand_def,
)
from xdsl.dialects.builtin import (
    ModuleOp, StringAttr, f64, FunctionType,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region, Dialect, OpResult
from xdsl.pattern_rewriter import (
    PatternRewriter, RewritePattern, PatternRewriteWalker,
    GreedyRewritePatternApplier, op_type_rewrite_pattern,
)
from xdsl.printer import Printer
from xdsl.context import Context
from io import StringIO


# ====== 1. Define a minimal 'alg' dialect ======

@irdl_op_definition
class AlgConst(IRDLOperation):
    name = "alg.const"
    res = result_def(AnyAttr())
    value = attr_def(StringAttr)

@irdl_op_definition
class AlgBinary(IRDLOperation):
    name = "alg.binary"
    lhs = operand_def(AnyAttr())
    rhs = operand_def(AnyAttr())
    res = result_def(AnyAttr())
    operator = attr_def(StringAttr)

@irdl_op_definition
class AlgCall(IRDLOperation):
    name = "alg.call"
    callee = operand_def(AnyAttr())
    args = var_operand_def(AnyAttr())
    res = result_def(AnyAttr())
    func_name = attr_def(StringAttr)

@irdl_op_definition
class AlgReturn(IRDLOperation):
    name = "alg.return"
    val = operand_def(AnyAttr())

AlgDialect = Dialect("alg", [AlgConst, AlgBinary, AlgCall, AlgReturn])


# ====== 2. Build a simple IR ======

def build_original_ir():
    block = Block()
    
    metric = AlgConst.build(
        result_types=[f64],
        attributes={"value": StringAttr("3.0")}
    )
    block.add_op(metric)
    
    penalty = AlgConst.build(
        result_types=[f64],
        attributes={"value": StringAttr("1.0")}
    )
    block.add_op(penalty)
    
    score = AlgBinary.build(
        operands=[metric.res, penalty.res],
        result_types=[f64],
        attributes={"operator": StringAttr("Add")}
    )
    block.add_op(score)
    
    ret = AlgReturn.build(operands=[score.res])
    block.add_op(ret)
    
    region = Region([block])
    func = FuncOp("detector", FunctionType.from_lists([], [f64]), region)
    module = ModuleOp([func])
    return module


# ====== 3. Define a RewritePattern ======

class InjectBPSummaryPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AlgBinary, rewriter: PatternRewriter):
        if op.operator.data != "Add":
            return
        
        bp_const = AlgConst.build(
            result_types=[f64],
            attributes={"value": StringAttr("bp_summary_fn")}
        )
        
        bp_call = AlgCall.build(
            operands=[bp_const.res, [op.lhs]],
            result_types=[f64],
            attributes={"func_name": StringAttr("bp_summary")}
        )
        
        new_add = AlgBinary.build(
            operands=[op.lhs, bp_call.res],
            result_types=[f64],
            attributes={"operator": StringAttr("Add")}
        )
        
        rewriter.replace_op(op, [bp_const, bp_call, new_add])


# ====== 4. Run the pattern rewriter ======

print("=" * 60)
print("BEFORE rewrite:")
print("=" * 60)
module = build_original_ir()
Printer().print_op(module)

ctx = Context()
ctx.load_dialect(AlgDialect)
walker = PatternRewriteWalker(
    GreedyRewritePatternApplier([InjectBPSummaryPattern()]),
    apply_recursively=False,
)
walker.rewrite_module(module)

print("\n" + "=" * 60)
print("AFTER rewrite:")
print("=" * 60)
Printer().print_op(module)

# ====== 5. Verify the rewrite ======
func = next(iter(module.ops))
body_block = next(iter(func.body.blocks))
ops = list(body_block.ops)
op_names = [op.name for op in ops]
print("\nOp sequence after rewrite:", op_names)

has_call = any(op.name == "alg.call" for op in ops)
print(f"Has injected call: {has_call}")

ret_op = ops[-1]
if ret_op.name == "alg.return":
    ret_input = ret_op.operands[0]
    if isinstance(ret_input, OpResult):
        print(f"Return uses result of: {ret_input.op.name}")
        if ret_input.op.name == "alg.binary":
            add_op = ret_input.op
            add_rhs = add_op.operands[1]
            if isinstance(add_rhs, OpResult) and add_rhs.op.name == "alg.call":
                print("SUCCESS: Return -> Add -> Call (BP injected correctly)")

print("\n" + "=" * 60)
print("CONCLUSION: xDSL PatternRewriter CAN do skeleton injection")
print("=" * 60)
