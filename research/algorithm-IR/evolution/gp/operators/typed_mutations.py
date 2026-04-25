"""Typed GP operators (Phase H+4 §6).

All operators are pure-IR: they take a parent ``FunctionIR`` and return
a new ``FunctionIR`` (deep-copied) with one localized edit. None of
them touch Python source.

Operators registered here:

  mut_const          Gaussian / categorical perturbation of a literal
  mut_binary_swap    swap a binary opcode within a same-signature group
  mut_compare_swap   swap a compare opcode within a same-signature group
  mut_argswap        swap two inputs of a non-commutative binary op
  mut_unary_flip     swap a unary opcode (USub<->UAdd / Invert)
  mut_const_to_var   redirect a use of a const value to an in-scope same-
                     type value (functional reference rewiring)

Each operator ALWAYS:

  * works on a deepcopy of the parent IR
  * scopes its candidate set to ``ctx.region_op_ids`` when non-empty
  * returns ``OperatorResult(child_ir=None, rejection_reason=...)``
    when no candidate exists (gate 1 fails downstream)
  * never touches block/control-flow ops or ops outside its remit
"""
from __future__ import annotations

import copy
from typing import Iterable

import numpy as np

from algorithm_ir.ir.model import FunctionIR, Op

from evolution.gp.operators.base import (
    GPContext,
    OperatorResult,
    register_operator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Same-signature swap groups for binary operators (commutative-equivalent
# under typed constraints — operands have the same type as in the original
# binary op). Only opcodes within the SAME group may be swapped.
_BINARY_SAME_SIG_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"Add", "Sub"}),
    frozenset({"Mult", "Div", "FloorDiv"}),
    frozenset({"Mod", "FloorDiv"}),
    frozenset({"BitAnd", "BitOr", "BitXor"}),
    frozenset({"LShift", "RShift"}),
    frozenset({"Pow", "Mult"}),
)

# Compare opcodes that are pairwise type-compatible.
_COMPARE_SAME_SIG_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"Lt", "LtE", "Gt", "GtE"}),
    frozenset({"Eq", "NotEq"}),
    frozenset({"Is", "IsNot"}),
)

# Unary operators that swap.
_UNARY_PAIRS: dict[str, tuple[str, ...]] = {
    "USub": ("UAdd",),
    "UAdd": ("USub",),
    "Invert": ("USub",),
    "Not": ("Not",),  # idempotent — caught by hash gate
}

# Non-commutative binary opcodes whose argswap changes semantics.
_NON_COMMUTATIVE_BINARY = frozenset({
    "Sub", "Div", "FloorDiv", "Mod", "Pow", "MatMult",
    "LShift", "RShift",
})


def _candidate_op_ids(ctx: GPContext, ir: FunctionIR) -> Iterable[str]:
    """Yield op_ids the operator is allowed to touch."""
    if ctx.region_op_ids:
        return [oid for oid in ctx.region_op_ids if oid in ir.ops]
    return list(ir.ops.keys())


def _filter_by_opcode(
    ctx: GPContext,
    ir: FunctionIR,
    opcodes: frozenset[str] | set[str] | tuple[str, ...],
) -> list[str]:
    s = set(opcodes)
    return [oid for oid in _candidate_op_ids(ctx, ir) if ir.ops[oid].opcode in s]


def _pick(rng: np.random.Generator, seq: list):
    return seq[int(rng.integers(0, len(seq)))]


# ---------------------------------------------------------------------------
# 1. mut_const — perturb a numeric literal
# ---------------------------------------------------------------------------

class _MutConst:
    name = "mut_const"
    weight = 0.30

    def propose(self, ctx, parent_ir, parent2_ir=None) -> OperatorResult:
        const_ids = [
            oid for oid in _filter_by_opcode(ctx, parent_ir, frozenset({"const"}))
            if isinstance(parent_ir.ops[oid].attrs.get("literal"),
                          (int, float, complex, bool))
        ]
        if not const_ids:
            return OperatorResult(child_ir=None, rejection_reason="no_const_in_region")
        target = _pick(ctx.rng, const_ids)
        child = copy.deepcopy(parent_ir)
        op = child.ops[target]
        v = op.attrs["literal"]
        if isinstance(v, bool):
            new_v = (not v)
        elif isinstance(v, int) and not isinstance(v, bool):
            # ±1 step or ±2 step half the time
            step = int(ctx.rng.choice([-2, -1, 1, 2]))
            new_v = v + step
        elif isinstance(v, float):
            scale = 0.1 if abs(v) < 1.0 else 0.1 * abs(v)
            new_v = float(v + ctx.rng.normal(0.0, scale))
            if new_v == v:
                new_v = float(v + 1e-3)
        elif isinstance(v, complex):
            new_v = complex(v.real + ctx.rng.normal(0.0, 0.1),
                            v.imag + ctx.rng.normal(0.0, 0.1))
        else:
            return OperatorResult(child_ir=None, rejection_reason="unsupported_literal")
        op.attrs["literal"] = new_v
        return OperatorResult(
            child_ir=child,
            diff_summary=f"const@{target}: {v!r} -> {new_v!r}",
        )


@register_operator("mut_const", weight=0.30)
def make_mut_const():
    return _MutConst()


# ---------------------------------------------------------------------------
# 2. mut_binary_swap
# ---------------------------------------------------------------------------

class _MutBinarySwap:
    name = "mut_binary_swap"
    weight = 0.15

    def propose(self, ctx, parent_ir, parent2_ir=None) -> OperatorResult:
        binary_ids = _filter_by_opcode(ctx, parent_ir, frozenset({"binary"}))
        if not binary_ids:
            return OperatorResult(child_ir=None, rejection_reason="no_binary_in_region")
        # Build (op_id, alternatives) candidates.
        cands: list[tuple[str, str]] = []
        for oid in binary_ids:
            cur = parent_ir.ops[oid].attrs.get("operator", "Add")
            for grp in _BINARY_SAME_SIG_GROUPS:
                if cur in grp:
                    for alt in grp:
                        if alt != cur:
                            cands.append((oid, alt))
        if not cands:
            return OperatorResult(child_ir=None, rejection_reason="no_swap_target")
        target_oid, new_op = _pick(ctx.rng, cands)
        child = copy.deepcopy(parent_ir)
        old = child.ops[target_oid].attrs.get("operator")
        child.ops[target_oid].attrs["operator"] = new_op
        return OperatorResult(
            child_ir=child,
            diff_summary=f"binary@{target_oid}: {old} -> {new_op}",
        )


@register_operator("mut_binary_swap", weight=0.15)
def make_mut_binary_swap():
    return _MutBinarySwap()


# ---------------------------------------------------------------------------
# 3. mut_compare_swap
# ---------------------------------------------------------------------------

class _MutCompareSwap:
    name = "mut_compare_swap"
    weight = 0.10

    def propose(self, ctx, parent_ir, parent2_ir=None) -> OperatorResult:
        cmp_ids = _filter_by_opcode(ctx, parent_ir, frozenset({"compare"}))
        if not cmp_ids:
            return OperatorResult(child_ir=None, rejection_reason="no_compare_in_region")
        cands: list[tuple[str, int, str]] = []
        for oid in cmp_ids:
            ops_list = parent_ir.ops[oid].attrs.get("operators") or [
                parent_ir.ops[oid].attrs.get("operator", "Eq")
            ]
            for idx, cur in enumerate(ops_list):
                for grp in _COMPARE_SAME_SIG_GROUPS:
                    if cur in grp:
                        for alt in grp:
                            if alt != cur:
                                cands.append((oid, idx, alt))
        if not cands:
            return OperatorResult(child_ir=None, rejection_reason="no_swap_target")
        target_oid, idx, new_op = _pick(ctx.rng, cands)
        child = copy.deepcopy(parent_ir)
        op = child.ops[target_oid]
        ops_list = op.attrs.get("operators")
        if isinstance(ops_list, list):
            old = ops_list[idx]
            ops_list[idx] = new_op
        else:
            old = op.attrs.get("operator")
            op.attrs["operator"] = new_op
        return OperatorResult(
            child_ir=child,
            diff_summary=f"compare@{target_oid}[{idx}]: {old} -> {new_op}",
        )


@register_operator("mut_compare_swap", weight=0.10)
def make_mut_compare_swap():
    return _MutCompareSwap()


# ---------------------------------------------------------------------------
# 4. mut_argswap (non-commutative binary inputs)
# ---------------------------------------------------------------------------

class _MutArgSwap:
    name = "mut_argswap"
    weight = 0.10

    def propose(self, ctx, parent_ir, parent2_ir=None) -> OperatorResult:
        cands: list[str] = []
        for oid in _filter_by_opcode(ctx, parent_ir, frozenset({"binary"})):
            op = parent_ir.ops[oid]
            cur = op.attrs.get("operator", "Add")
            if cur in _NON_COMMUTATIVE_BINARY and len(op.inputs) == 2:
                cands.append(oid)
        if not cands:
            return OperatorResult(child_ir=None, rejection_reason="no_noncomm_binary")
        target = _pick(ctx.rng, cands)
        child = copy.deepcopy(parent_ir)
        op = child.ops[target]
        op.inputs[0], op.inputs[1] = op.inputs[1], op.inputs[0]
        return OperatorResult(
            child_ir=child,
            diff_summary=f"argswap@{target}",
        )


@register_operator("mut_argswap", weight=0.10)
def make_mut_argswap():
    return _MutArgSwap()


# ---------------------------------------------------------------------------
# 5. mut_unary_flip
# ---------------------------------------------------------------------------

class _MutUnaryFlip:
    name = "mut_unary_flip"
    weight = 0.05

    def propose(self, ctx, parent_ir, parent2_ir=None) -> OperatorResult:
        cands: list[tuple[str, str]] = []
        for oid in _filter_by_opcode(ctx, parent_ir, frozenset({"unary"})):
            cur = parent_ir.ops[oid].attrs.get("operator", "USub")
            alts = _UNARY_PAIRS.get(cur, ())
            for alt in alts:
                if alt != cur:
                    cands.append((oid, alt))
        if not cands:
            return OperatorResult(child_ir=None, rejection_reason="no_unary_to_flip")
        target_oid, new_op = _pick(ctx.rng, cands)
        child = copy.deepcopy(parent_ir)
        old = child.ops[target_oid].attrs.get("operator")
        child.ops[target_oid].attrs["operator"] = new_op
        return OperatorResult(
            child_ir=child,
            diff_summary=f"unary@{target_oid}: {old} -> {new_op}",
        )


@register_operator("mut_unary_flip", weight=0.05)
def make_mut_unary_flip():
    return _MutUnaryFlip()


# ---------------------------------------------------------------------------
# 6. mut_const_to_var — redirect a use of a const to a same-type in-scope value
# ---------------------------------------------------------------------------

def _value_type(ir: FunctionIR, vid: str) -> str | None:
    v = ir.values.get(vid)
    if v is None:
        return None
    return v.type_hint


class _MutConstToVar:
    name = "mut_const_to_var"
    weight = 0.05

    def propose(self, ctx, parent_ir, parent2_ir=None) -> OperatorResult:
        # Find a binary/compare op inside region whose input is the
        # output of a const op; redirect it to a different in-scope
        # value of the same type.
        region_oids = set(_candidate_op_ids(ctx, parent_ir))
        if not region_oids:
            return OperatorResult(child_ir=None, rejection_reason="empty_region")

        # Build value-id -> type map; collect all values in scope (defined
        # by any op in the function — keeping it simple).
        cands: list[tuple[str, int, str, str]] = []
        # Map output->const-op for quick lookup
        const_outputs: dict[str, str] = {}
        for oid, op in parent_ir.ops.items():
            if op.opcode == "const" and op.outputs:
                const_outputs[op.outputs[0]] = oid

        # Group all in-scope values by type.
        by_type: dict[str, list[str]] = {}
        for vid, v in parent_ir.values.items():
            t = v.type_hint or "any"
            by_type.setdefault(t, []).append(vid)

        for oid in region_oids:
            op = parent_ir.ops[oid]
            if op.opcode not in ("binary", "compare", "call", "unary"):
                continue
            for idx, vid in enumerate(op.inputs):
                if vid not in const_outputs:
                    continue
                t = _value_type(parent_ir, vid) or "any"
                pool = [u for u in by_type.get(t, []) if u != vid]
                if not pool:
                    continue
                replacement = pool[int(ctx.rng.integers(0, len(pool)))]
                cands.append((oid, idx, vid, replacement))

        if not cands:
            return OperatorResult(child_ir=None, rejection_reason="no_const_use_to_redirect")

        target_oid, idx, old_vid, new_vid = _pick(ctx.rng, cands)
        child = copy.deepcopy(parent_ir)
        child.ops[target_oid].inputs[idx] = new_vid
        return OperatorResult(
            child_ir=child,
            diff_summary=f"const_to_var@{target_oid}[{idx}]: {old_vid} -> {new_vid}",
        )


@register_operator("mut_const_to_var", weight=0.05)
def make_mut_const_to_var():
    return _MutConstToVar()


# Auto-register everything by importing this module.
__all__ = [
    "make_mut_const",
    "make_mut_binary_swap",
    "make_mut_compare_swap",
    "make_mut_argswap",
    "make_mut_unary_flip",
    "make_mut_const_to_var",
]
