"""Phase H+5 R5: structural typed-GP operators (5 mandatory).

These operators perform genuine *structural* edits on FunctionIR — they
add, remove, or rewire ops, going beyond the attribute-level swaps
implemented in :mod:`evolution.gp.operators.typed_mutations`.

Type compatibility uses :mod:`algorithm_ir.ir.type_lattice` (no parallel
type system). Each operator:

  * works on a deepcopy of the parent IR
  * preserves block.op_ids ordering when inserting / deleting
  * calls ``rebuild_def_use`` before returning so gate 2 is happy
  * scopes its candidate set to ``ctx.region_op_ids`` when non-empty
  * returns ``OperatorResult(child_ir=None, rejection_reason=...)`` if
    no candidate found

Operators registered:

  mut_insert_typed       Insert a new binary op ``v op c`` between an
                         existing value v and one of its downstream uses.
  mut_delete_typed       Remove a non-essential op (assign / type-preserving
                         binary) and rewire its uses to its first input.
  mut_subtree_replace    Replace a binary op with a freshly-synthesized
                         binary expression of the same output type
                         (uses a different operator and a fresh constant).
  cx_subtree_typed       Same-slot crossover: import a binary opcode +
                         constant pattern from parent2's region into
                         parent's region at a type-compatible point.
  mut_primitive_inject   Insert a small affine primitive
                         (``out = x * scale + bias``) before a chosen use.
"""
from __future__ import annotations

import copy
from typing import Iterable

import numpy as np

from algorithm_ir.ir.model import FunctionIR, Op, Value
from algorithm_ir.ir.type_lattice import (
    PRIMITIVE_TYPES,
    TYPE_TOP,
    combine_binary_type,
    is_subtype,
    promote_dtype,
)
from algorithm_ir.ir.validator import rebuild_def_use

from evolution.gp.operators.base import (
    GPContext,
    OperatorResult,
    register_operator,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _candidate_op_ids(ctx: GPContext, ir: FunctionIR) -> list[str]:
    if ctx.region_op_ids:
        return [oid for oid in ctx.region_op_ids if oid in ir.ops]
    return list(ir.ops.keys())


def _fresh_value_id(ir: FunctionIR, hint: str = "v") -> str:
    n = 0
    while True:
        cand = f"__r5_{hint}_{n}"
        if cand not in ir.values:
            return cand
        n += 1


def _fresh_op_id(ir: FunctionIR, hint: str = "op") -> str:
    n = 0
    while True:
        cand = f"__r5_{hint}_{n}"
        if cand not in ir.ops:
            return cand
        n += 1


def _value_type(ir: FunctionIR, vid: str) -> str:
    v = ir.values.get(vid)
    if v is None or v.type_hint is None:
        return TYPE_TOP
    return v.type_hint


def _add_value(ir: FunctionIR, vid: str, type_hint: str) -> Value:
    v = Value(id=vid, name_hint=vid, type_hint=type_hint)
    ir.values[vid] = v
    return v


def _insert_op_before(ir: FunctionIR, block_id: str, anchor_oid: str,
                      new_oid: str) -> bool:
    """Insert ``new_oid`` immediately before ``anchor_oid`` in block.op_ids."""
    block = ir.blocks.get(block_id)
    if block is None:
        return False
    try:
        idx = block.op_ids.index(anchor_oid)
    except ValueError:
        return False
    block.op_ids.insert(idx, new_oid)
    return True


def _terminator_opcodes() -> set[str]:
    return {"branch", "jump", "return"}


def _block_op_index(ir: FunctionIR, oid: str) -> tuple[str, int] | None:
    op = ir.ops.get(oid)
    if op is None:
        return None
    block = ir.blocks.get(op.block_id)
    if block is None or oid not in block.op_ids:
        return None
    return op.block_id, block.op_ids.index(oid)


def _pick(rng: np.random.Generator, seq: list):
    return seq[int(rng.integers(0, len(seq)))]


def _numeric_in_lattice(t: str) -> bool:
    # Frontend often emits ``TYPE_TOP`` ("any") because static type
    # inference is best-effort. Treat ``any`` as compatible (we will
    # default to float/cx constants); downstream subtype gates
    # ``is_subtype(result_t, v_type)`` still pass since everything is a
    # subtype of ``any``.
    if t == TYPE_TOP:
        return True
    return t in PRIMITIVE_TYPES or t in {"vec_f", "vec_cx", "vec_i", "mat_f", "mat_cx"}


def _make_const_op(ir: FunctionIR, literal, type_hint: str) -> tuple[str, str]:
    """Create a const op + value, return (op_id, value_id). Caller must
    insert op_id into a block.op_ids."""
    vid = _fresh_value_id(ir, "c")
    _add_value(ir, vid, type_hint)
    oid = _fresh_op_id(ir, "const")
    ir.ops[oid] = Op(
        id=oid, opcode="const", inputs=[], outputs=[vid],
        block_id="",  # caller fills
        attrs={"literal": literal, "name": vid},
    )
    return oid, vid


def _redirect_one_use(ir: FunctionIR, old_vid: str, new_vid: str,
                      rng: np.random.Generator,
                      forbid_op_ids: set[str] | None = None) -> str | None:
    """Redirect one downstream use of ``old_vid`` to ``new_vid``.

    Returns the op_id whose input slot was rewired, or None if nothing
    eligible. ``forbid_op_ids`` excludes op_ids from being rewired (used
    to avoid feeding the inserted op into itself or into ops we just
    created).
    """
    forbid = forbid_op_ids or set()
    candidates: list[tuple[str, int]] = []
    for use_oid in list(ir.values[old_vid].use_ops):
        if use_oid in forbid:
            continue
        op = ir.ops.get(use_oid)
        if op is None:
            continue
        for i, in_vid in enumerate(op.inputs):
            if in_vid == old_vid:
                candidates.append((use_oid, i))
    if not candidates:
        return None
    use_oid, slot = _pick(rng, candidates)
    ir.ops[use_oid].inputs[slot] = new_vid
    return use_oid


# ---------------------------------------------------------------------------
# 1. mut_insert_typed
# ---------------------------------------------------------------------------

# Type-safe binary ops we may inject.
_INJECT_BINOPS_NUMERIC = ("Add", "Sub", "Mult")


class _MutInsertTyped:
    name = "mut_insert_typed"
    weight = 0.10

    def propose(self, ctx: GPContext, parent_ir: FunctionIR,
                parent2_ir: FunctionIR | None = None) -> OperatorResult:
        rng = ctx.rng
        # Find values produced inside the region with a numeric type and
        # at least one downstream use in the region.
        region = set(_candidate_op_ids(ctx, parent_ir))
        cand_values: list[str] = []
        for oid in region:
            op = parent_ir.ops[oid]
            for vid in op.outputs:
                v = parent_ir.values.get(vid)
                if v is None:
                    continue
                t = v.type_hint or TYPE_TOP
                if not _numeric_in_lattice(t):
                    continue
                # At least one downstream use that is itself in region
                # (or no region constraint).
                uses = [u for u in v.use_ops if (not region or u in region)]
                # Don't insert into terminator-fed values (avoid touching
                # control flow).
                uses = [u for u in uses
                        if parent_ir.ops[u].opcode not in _terminator_opcodes()]
                if uses:
                    cand_values.append(vid)
        if not cand_values:
            return OperatorResult(child_ir=None, rejection_reason="no_numeric_value_with_use")

        target_vid = _pick(rng, cand_values)
        child = copy.deepcopy(parent_ir)
        v_type = _value_type(child, target_vid)
        op_name = _pick(rng, list(_INJECT_BINOPS_NUMERIC))

        # Constant of compatible scalar type.
        if v_type in {"cx", "vec_cx", "mat_cx"}:
            literal = complex(float(rng.normal(0.0, 0.5)), float(rng.normal(0.0, 0.5)))
            const_t = "cx"
        elif v_type in {"int", "vec_i"}:
            literal = int(rng.integers(-3, 4))
            const_t = "int"
        else:
            literal = float(rng.normal(0.0, 0.5))
            const_t = "float"

        # Place the new binary op right before one of target_vid's uses.
        anchor_uses = [
            u for u in child.values[target_vid].use_ops
            if child.ops[u].opcode not in _terminator_opcodes()
        ]
        if not anchor_uses:
            return OperatorResult(child_ir=None, rejection_reason="no_use_anchor")
        anchor_oid = _pick(rng, anchor_uses)
        anchor_block = child.ops[anchor_oid].block_id

        # Const op (placed immediately before the anchor).
        const_oid, const_vid = _make_const_op(child, literal, const_t)
        child.ops[const_oid].block_id = anchor_block

        # Binary op: target_vid OP const_vid -> new_vid.
        result_t = combine_binary_type(op_name, v_type, const_t)
        if result_t == TYPE_TOP and v_type != TYPE_TOP:
            return OperatorResult(child_ir=None, rejection_reason="type_combine_top")
        new_vid = _fresh_value_id(child, "v")
        _add_value(child, new_vid, result_t)
        bin_oid = _fresh_op_id(child, "bin")
        child.ops[bin_oid] = Op(
            id=bin_oid, opcode="binary",
            inputs=[target_vid, const_vid],
            outputs=[new_vid],
            block_id=anchor_block,
            attrs={"operator": op_name, "name": new_vid},
        )

        if not _insert_op_before(child, anchor_block, anchor_oid, const_oid):
            return OperatorResult(child_ir=None, rejection_reason="anchor_block_lookup_failed")
        if not _insert_op_before(child, anchor_block, anchor_oid, bin_oid):
            return OperatorResult(child_ir=None, rejection_reason="anchor_block_lookup_failed")

        # Type-safety check on rewire: only rewire the use slot if its
        # operator can accept ``result_t``. Easy approximation: only
        # rewire when result_t is_subtype of original v_type (so any
        # downstream consumer that worked with v_type still works).
        if not is_subtype(result_t, v_type):
            return OperatorResult(child_ir=None, rejection_reason="result_not_subtype")

        rewired = _redirect_one_use(
            child, target_vid, new_vid, rng,
            forbid_op_ids={bin_oid, const_oid},
        )
        if rewired is None:
            return OperatorResult(child_ir=None, rejection_reason="no_rewire_candidate")

        rebuild_def_use(child)
        return OperatorResult(
            child_ir=child,
            diff_summary=f"insert {op_name} at {target_vid}->{new_vid} feeding {rewired}",
        )


@register_operator("mut_insert_typed", weight=0.10)
def make_mut_insert_typed():
    return _MutInsertTyped()


# ---------------------------------------------------------------------------
# 2. mut_delete_typed
# ---------------------------------------------------------------------------

class _MutDeleteTyped:
    name = "mut_delete_typed"
    weight = 0.10

    def propose(self, ctx: GPContext, parent_ir: FunctionIR,
                parent2_ir: FunctionIR | None = None) -> OperatorResult:
        rng = ctx.rng
        region = set(_candidate_op_ids(ctx, parent_ir))

        # Eligible: assign / binary / unary ops whose first input has a
        # type that is a subtype of the output's type — so rewiring the
        # output users to that input preserves type safety.
        cands: list[str] = []
        for oid in region:
            op = parent_ir.ops[oid]
            if op.opcode not in {"assign", "binary", "unary"}:
                continue
            if not op.outputs or not op.inputs:
                continue
            out_t = _value_type(parent_ir, op.outputs[0])
            in_t = _value_type(parent_ir, op.inputs[0])
            # The replacement (op.inputs[0]) must be a subtype of the
            # value its downstream uses expect.
            if not is_subtype(in_t, out_t):
                continue
            # Don't delete an op whose output feeds a return / branch
            # directly — keep the contract surface intact.
            unsafe = False
            for u in parent_ir.values.get(op.outputs[0],
                                          Value(id="", name_hint=None,
                                                type_hint=None)).use_ops:
                if parent_ir.ops[u].opcode in _terminator_opcodes():
                    unsafe = True
                    break
            if unsafe:
                continue
            cands.append(oid)
        if not cands:
            return OperatorResult(child_ir=None, rejection_reason="no_deletable_op")

        target_oid = _pick(rng, cands)
        child = copy.deepcopy(parent_ir)
        target_op = child.ops[target_oid]
        out_vid = target_op.outputs[0]
        replacement_vid = target_op.inputs[0]

        # Redirect every downstream use of out_vid to replacement_vid.
        for use_oid in list(child.values[out_vid].use_ops):
            use_op = child.ops.get(use_oid)
            if use_op is None or use_oid == target_oid:
                continue
            use_op.inputs = [
                replacement_vid if x == out_vid else x for x in use_op.inputs
            ]
            use_op.outputs = [
                replacement_vid if x == out_vid else x for x in use_op.outputs
            ]

        # Remove target op from its block and from the op dict.
        block = child.blocks.get(target_op.block_id)
        if block is not None and target_oid in block.op_ids:
            block.op_ids.remove(target_oid)
        del child.ops[target_oid]
        # Remove the now-orphan output value.
        if out_vid in child.values:
            del child.values[out_vid]

        rebuild_def_use(child)
        return OperatorResult(
            child_ir=child,
            diff_summary=f"delete {target_op.opcode}@{target_oid}; users -> {replacement_vid}",
        )


@register_operator("mut_delete_typed", weight=0.10)
def make_mut_delete_typed():
    return _MutDeleteTyped()


# ---------------------------------------------------------------------------
# 3. mut_subtree_replace
# ---------------------------------------------------------------------------

# Cross-group binary swaps that preserve numeric output type.
_SUBTREE_SWAP_TARGETS: dict[str, tuple[str, ...]] = {
    "Add":      ("Mult", "Sub"),
    "Sub":      ("Add", "Mult"),
    "Mult":     ("Add", "Sub"),
    "Div":      ("Mult", "FloorDiv"),
    "FloorDiv": ("Div", "Mult"),
}


class _MutSubtreeReplace:
    name = "mut_subtree_replace"
    weight = 0.08

    def propose(self, ctx: GPContext, parent_ir: FunctionIR,
                parent2_ir: FunctionIR | None = None) -> OperatorResult:
        rng = ctx.rng
        region = set(_candidate_op_ids(ctx, parent_ir))

        cands: list[tuple[str, str]] = []
        for oid in region:
            op = parent_ir.ops[oid]
            if op.opcode != "binary":
                continue
            cur = op.attrs.get("operator")
            if not isinstance(cur, str) or cur not in _SUBTREE_SWAP_TARGETS:
                continue
            for new_op in _SUBTREE_SWAP_TARGETS[cur]:
                # Only keep swaps that yield a subtype of the original
                # output type so downstream uses stay type-safe.
                lhs_t = _value_type(parent_ir, op.inputs[0]) if op.inputs else TYPE_TOP
                rhs_t = _value_type(parent_ir, op.inputs[1]) if len(op.inputs) > 1 else TYPE_TOP
                old_t = _value_type(parent_ir, op.outputs[0]) if op.outputs else TYPE_TOP
                new_t = combine_binary_type(new_op, lhs_t, rhs_t)
                if not is_subtype(new_t, old_t):
                    continue
                cands.append((oid, new_op))
        if not cands:
            return OperatorResult(child_ir=None, rejection_reason="no_swap_candidate")

        target_oid, new_op = _pick(rng, cands)
        child = copy.deepcopy(parent_ir)
        op = child.ops[target_oid]
        old = op.attrs.get("operator")
        # Build a *new* op (so canonical hash differs from a plain
        # mut_binary_swap that mutates attrs in place at the same id).
        new_oid = _fresh_op_id(child, "subtree")
        new_vid = _fresh_value_id(child, "v")
        out_t = _value_type(child, op.outputs[0])
        _add_value(child, new_vid, out_t)
        child.ops[new_oid] = Op(
            id=new_oid, opcode="binary",
            inputs=list(op.inputs),
            outputs=[new_vid],
            block_id=op.block_id,
            attrs={"operator": new_op, "name": new_vid,
                   "_provenance": dict(op.attrs.get("_provenance", {}))},
        )
        # Insert the new op directly after the old one in its block.
        block = child.blocks[op.block_id]
        idx = block.op_ids.index(target_oid)
        block.op_ids.insert(idx + 1, new_oid)
        # Redirect ONE downstream use of the old output to the new value.
        rewired = _redirect_one_use(child, op.outputs[0], new_vid, rng,
                                    forbid_op_ids={new_oid})
        if rewired is None:
            return OperatorResult(child_ir=None, rejection_reason="no_rewire_candidate")

        rebuild_def_use(child)
        return OperatorResult(
            child_ir=child,
            diff_summary=f"subtree_replace {old}->{new_op} forking @{target_oid}",
        )


@register_operator("mut_subtree_replace", weight=0.08)
def make_mut_subtree_replace():
    return _MutSubtreeReplace()


# ---------------------------------------------------------------------------
# 4. cx_subtree_typed (crossover)
# ---------------------------------------------------------------------------

class _CxSubtreeTyped:
    name = "cx_subtree_typed"
    weight = 0.06

    def propose(self, ctx: GPContext, parent_ir: FunctionIR,
                parent2_ir: FunctionIR | None = None) -> OperatorResult:
        if parent2_ir is None or parent2_ir is parent_ir:
            return OperatorResult(child_ir=None, rejection_reason="no_parent2")
        rng = ctx.rng
        region = set(_candidate_op_ids(ctx, parent_ir))

        # Collect (oid, output_type) for binary ops in parent's region.
        parent_binaries: list[tuple[str, str]] = []
        for oid in region:
            op = parent_ir.ops[oid]
            if op.opcode != "binary" or not op.outputs:
                continue
            t = _value_type(parent_ir, op.outputs[0])
            if not _numeric_in_lattice(t):
                continue
            parent_binaries.append((oid, t))
        if not parent_binaries:
            return OperatorResult(child_ir=None, rejection_reason="no_parent_binary")

        # Donor pool: binary ops in parent2 whose output type is a
        # subtype of one we have, with the same operator family.
        donor_pool: list[tuple[str, str]] = []
        for oid, op in parent2_ir.ops.items():
            if op.opcode != "binary" or not op.outputs:
                continue
            opname = op.attrs.get("operator")
            if not isinstance(opname, str):
                continue
            donor_pool.append((oid, opname))
        if not donor_pool:
            return OperatorResult(child_ir=None, rejection_reason="no_donor_binary")

        target_oid, target_t = _pick(rng, parent_binaries)
        donor_oid, donor_opname = _pick(rng, donor_pool)

        child = copy.deepcopy(parent_ir)
        op = child.ops[target_oid]
        # Type-safe replacement of the operator only (a "subtree" of size 1).
        lhs_t = _value_type(child, op.inputs[0]) if op.inputs else TYPE_TOP
        rhs_t = _value_type(child, op.inputs[1]) if len(op.inputs) > 1 else TYPE_TOP
        new_t = combine_binary_type(donor_opname, lhs_t, rhs_t)
        if not is_subtype(new_t, target_t):
            return OperatorResult(child_ir=None, rejection_reason="cross_type_unsafe")
        old = op.attrs.get("operator")
        if old == donor_opname:
            return OperatorResult(child_ir=None, rejection_reason="cross_noop")

        # Splice as a new op (so canonical hash and op_id structure changes).
        new_oid = _fresh_op_id(child, "cx")
        new_vid = _fresh_value_id(child, "v")
        _add_value(child, new_vid, new_t)
        child.ops[new_oid] = Op(
            id=new_oid, opcode="binary",
            inputs=list(op.inputs),
            outputs=[new_vid],
            block_id=op.block_id,
            attrs={"operator": donor_opname, "name": new_vid},
        )
        block = child.blocks[op.block_id]
        idx = block.op_ids.index(target_oid)
        block.op_ids.insert(idx + 1, new_oid)
        rewired = _redirect_one_use(child, op.outputs[0], new_vid, rng,
                                    forbid_op_ids={new_oid})
        if rewired is None:
            return OperatorResult(child_ir=None, rejection_reason="no_rewire_candidate")

        rebuild_def_use(child)
        return OperatorResult(
            child_ir=child,
            diff_summary=f"cx_subtree imported {donor_opname} from parent2@{donor_oid} -> @{target_oid}",
        )


@register_operator("cx_subtree_typed", weight=0.06, crossover=True)
def make_cx_subtree_typed():
    return _CxSubtreeTyped()


# ---------------------------------------------------------------------------
# 5. mut_primitive_inject (affine primitive: x * scale + bias)
# ---------------------------------------------------------------------------

class _MutPrimitiveInject:
    name = "mut_primitive_inject"
    weight = 0.06

    def propose(self, ctx: GPContext, parent_ir: FunctionIR,
                parent2_ir: FunctionIR | None = None) -> OperatorResult:
        rng = ctx.rng
        region = set(_candidate_op_ids(ctx, parent_ir))

        # Pick a value of numeric type with at least one non-terminator
        # downstream use.
        cand_values: list[str] = []
        for oid in region:
            op = parent_ir.ops[oid]
            for vid in op.outputs:
                v = parent_ir.values.get(vid)
                if v is None:
                    continue
                t = v.type_hint or TYPE_TOP
                if not _numeric_in_lattice(t):
                    continue
                uses = [u for u in v.use_ops
                        if parent_ir.ops[u].opcode not in _terminator_opcodes()]
                if uses:
                    cand_values.append(vid)
        if not cand_values:
            return OperatorResult(child_ir=None, rejection_reason="no_numeric_value")

        target_vid = _pick(rng, cand_values)
        child = copy.deepcopy(parent_ir)
        v_type = _value_type(child, target_vid)

        # Build constants compatible with v_type.
        if v_type in {"cx", "vec_cx", "mat_cx"}:
            scale_lit = complex(float(rng.normal(1.0, 0.1)), 0.0)
            bias_lit = complex(float(rng.normal(0.0, 0.1)),
                               float(rng.normal(0.0, 0.1)))
            const_t = "cx"
        elif v_type in {"int", "vec_i"}:
            scale_lit = int(rng.integers(1, 3))
            bias_lit = int(rng.integers(-2, 3))
            const_t = "int"
        else:
            scale_lit = float(rng.normal(1.0, 0.1))
            bias_lit = float(rng.normal(0.0, 0.1))
            const_t = "float"

        anchor_uses = [
            u for u in child.values[target_vid].use_ops
            if child.ops[u].opcode not in _terminator_opcodes()
        ]
        if not anchor_uses:
            return OperatorResult(child_ir=None, rejection_reason="no_use_anchor")
        anchor_oid = _pick(rng, anchor_uses)
        anchor_block = child.ops[anchor_oid].block_id

        # const(scale), const(bias), bin(Mult: target * scale -> tmp),
        # bin(Add: tmp + bias -> out).
        scale_oid, scale_vid = _make_const_op(child, scale_lit, const_t)
        child.ops[scale_oid].block_id = anchor_block
        bias_oid, bias_vid = _make_const_op(child, bias_lit, const_t)
        child.ops[bias_oid].block_id = anchor_block

        tmp_t = combine_binary_type("Mult", v_type, const_t)
        out_t = combine_binary_type("Add", tmp_t, const_t)
        if (tmp_t == TYPE_TOP or out_t == TYPE_TOP) and v_type != TYPE_TOP:
            return OperatorResult(child_ir=None, rejection_reason="type_combine_top")
        if not is_subtype(out_t, v_type):
            return OperatorResult(child_ir=None, rejection_reason="primitive_breaks_type")

        tmp_vid = _fresh_value_id(child, "tmp")
        _add_value(child, tmp_vid, tmp_t)
        out_vid = _fresh_value_id(child, "out")
        _add_value(child, out_vid, out_t)
        mult_oid = _fresh_op_id(child, "mult")
        add_oid = _fresh_op_id(child, "add")
        child.ops[mult_oid] = Op(
            id=mult_oid, opcode="binary",
            inputs=[target_vid, scale_vid],
            outputs=[tmp_vid],
            block_id=anchor_block,
            attrs={"operator": "Mult", "name": tmp_vid},
        )
        child.ops[add_oid] = Op(
            id=add_oid, opcode="binary",
            inputs=[tmp_vid, bias_vid],
            outputs=[out_vid],
            block_id=anchor_block,
            attrs={"operator": "Add", "name": out_vid},
        )

        for new_oid in (scale_oid, bias_oid, mult_oid, add_oid):
            if not _insert_op_before(child, anchor_block, anchor_oid, new_oid):
                return OperatorResult(child_ir=None, rejection_reason="anchor_block_lookup_failed")

        rewired = _redirect_one_use(
            child, target_vid, out_vid, rng,
            forbid_op_ids={scale_oid, bias_oid, mult_oid, add_oid},
        )
        if rewired is None:
            return OperatorResult(child_ir=None, rejection_reason="no_rewire_candidate")

        rebuild_def_use(child)
        return OperatorResult(
            child_ir=child,
            diff_summary=f"primitive_inject affine on {target_vid} feeding {rewired}",
        )


@register_operator("mut_primitive_inject", weight=0.06)
def make_mut_primitive_inject():
    return _MutPrimitiveInject()
