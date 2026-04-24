from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.region.slicer import (
    backward_slice_by_values,
    backward_slice_until_values,
    forward_slice_from_values,
)


@dataclass(frozen=True)
class BoundaryRegionSpec:
    output_values: list[str]
    cut_values: list[str] = field(default_factory=list)


@dataclass
class RewriteRegion:
    region_id: str
    op_ids: list[str]
    block_ids: list[str]
    entry_values: list[str]
    exit_values: list[str]
    read_set: list[str]
    write_set: list[str]
    state_carriers: list[str]
    schedule_anchors: dict[str, Any]
    allows_new_state: bool
    attrs: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def define_rewrite_region(
    func_ir: FunctionIR,
    *,
    boundary_spec: BoundaryRegionSpec | None = None,
    op_ids: list[str] | None = None,
    source_span: tuple[int, int, int, int] | None = None,
    exit_values: list[str] | None = None,
    state_carriers: list[str] | None = None,
) -> RewriteRegion:
    if op_ids is None:
        if boundary_spec is not None:
            raw_output_values = _dedupe_values(boundary_spec.output_values)
            raw_cut_values = _dedupe_values(boundary_spec.cut_values)
            invalid_cuts = [vid for vid in raw_cut_values if vid not in func_ir.values]
            if invalid_cuts:
                raise ValueError(f"Unknown cut values: {invalid_cuts}")
            invalid_outputs = [vid for vid in raw_output_values if vid not in func_ir.values]
            if invalid_outputs:
                raise ValueError(f"Unknown output values: {invalid_outputs}")
            output_values = [vid for vid in raw_output_values if vid in func_ir.values]
            cut_values = [vid for vid in raw_cut_values if vid in func_ir.values]
            closure = backward_slice_by_values(func_ir, output_values)
            bad_cuts = [
                vid for vid in cut_values
                if func_ir.values[vid].def_op is not None
                and func_ir.values[vid].def_op not in closure
            ]
            if bad_cuts:
                raise ValueError(
                    f"Cut values must lie in the backward closure of outputs: {bad_cuts}"
                )
            op_ids = sorted(
                backward_slice_until_values(func_ir, output_values, cut_values),
                key=lambda oid: _op_order_key(func_ir, oid),
            )
            selected_output_values = output_values
            selected_cut_values = cut_values
        elif source_span is not None:
            op_ids = [
                op.id
                for op in func_ir.ops.values()
                if _overlaps(op.source_span, source_span)
            ]
            selected_output_values = []
            selected_cut_values = []
        elif exit_values:
            selected_output_values = _normalize_values(func_ir, exit_values)
            selected_cut_values = []
            op_ids = sorted(
                backward_slice_until_values(func_ir, selected_output_values, []),
                key=lambda oid: _op_order_key(func_ir, oid),
            )
        elif state_carriers:
            op_ids = sorted(forward_slice_from_values(func_ir, state_carriers))
            selected_output_values = []
            selected_cut_values = []
        else:
            raise ValueError("One of boundary_spec, op_ids, source_span, exit_values, or state_carriers must be provided")
    else:
        selected_output_values = []
        selected_cut_values = []

    op_set = set(op_ids)
    block_ids = sorted({func_ir.ops[op_id].block_id for op_id in op_ids})

    region_entry_values: set[str] = set()
    region_exit_values: set[str] = set(exit_values or [])
    region_defined_values: set[str] = set()
    region_used_values: set[str] = set()
    read_set: set[str] = set()
    write_set: set[str] = set()

    for op_id in op_ids:
        op = func_ir.ops[op_id]
        region_defined_values.update(op.outputs)
        region_used_values.update(op.inputs)
        if op.opcode in {"get_attr"}:
            read_set.add(f"attr:{op.inputs[0]}:{op.attrs['attr']}")
        elif op.opcode in {"set_attr"}:
            write_set.add(f"attr:{op.inputs[0]}:{op.attrs['attr']}")
        elif op.opcode in {"get_item"}:
            read_set.add(f"item:{op.inputs[0]}")
        elif op.opcode in {"set_item"}:
            write_set.add(f"item:{op.inputs[0]}")
        elif op.opcode in {"append", "pop"}:
            write_set.add(f"container:{op.inputs[0]}")

    for value_id in sorted(region_used_values, key=lambda vid: _value_order_key(func_ir, vid)):
        def_op = func_ir.values[value_id].def_op
        if def_op is None or def_op not in op_set:
            region_entry_values.add(value_id)

    for value_id in sorted(region_defined_values, key=lambda vid: _value_order_key(func_ir, vid)):
        for use_op in func_ir.values[value_id].use_ops:
            if use_op not in op_set:
                region_exit_values.add(value_id)

    carrier_ids = list(state_carriers or [])
    if not carrier_ids:
        carrier_ids = sorted(
            {
                value_id
                for value_id in region_entry_values | region_defined_values
                if func_ir.values[value_id].type_hint in {"list", "dict", "object"}
            }
        )

    schedule_anchors = {
        "entry_blocks": [
            block_id
            for block_id in block_ids
            if any(pred not in block_ids for pred in func_ir.blocks[block_id].preds) or not func_ir.blocks[block_id].preds
        ],
        "exit_blocks": [
            block_id
            for block_id in block_ids
            if any(succ not in block_ids for succ in func_ir.blocks[block_id].succs) or not func_ir.blocks[block_id].succs
        ],
        "loop_blocks": [
            block_id
            for block_id in block_ids
            if any(op.attrs.get("loop_backedge") for op in _block_ops(func_ir, block_id))
        ],
    }

    return RewriteRegion(
        region_id=f"region_{func_ir.name}_{len(op_ids)}",
        op_ids=list(op_ids),
        block_ids=block_ids,
        entry_values=sorted(region_entry_values, key=lambda vid: _value_order_key(func_ir, vid)),
        exit_values=sorted(region_exit_values, key=lambda vid: _value_order_key(func_ir, vid)),
        read_set=sorted(read_set),
        write_set=sorted(write_set),
        state_carriers=carrier_ids,
        schedule_anchors=schedule_anchors,
        allows_new_state=True,
        attrs={},
        provenance={
            "selected_via": (
                "boundary_cut"
                if boundary_spec is not None or exit_values is not None
                else "op_ids"
                if op_ids is not None and source_span is None and state_carriers is None
                else "derived"
            ),
            "construction_mode": (
                "boundary_cut" if boundary_spec is not None or exit_values is not None else "legacy"
            ),
            "selected_output_values": list(selected_output_values),
            "selected_cut_values": list(selected_cut_values),
            "effective_output_values": [
                vid for vid in selected_output_values
                if vid in region_exit_values
            ] or list(selected_output_values),
            "effective_cut_values": [
                vid for vid in selected_cut_values
                if vid in region_entry_values
            ],
        },
    )


def _overlaps(lhs, rhs) -> bool:
    if lhs is None or rhs is None:
        return False
    l1, c1, l2, c2 = lhs
    r1, rc1, r2, rc2 = rhs
    if l2 < r1 or r2 < l1:
        return False
    if l1 == r2 and c1 > rc2:
        return False
    if r1 == l2 and rc1 > c2:
        return False
    return True


def _block_ops(func_ir: FunctionIR, block_id: str):
    for op_id in func_ir.blocks[block_id].op_ids:
        yield func_ir.ops[op_id]


def _normalize_values(func_ir: FunctionIR, values: list[str]) -> list[str]:
    return [value_id for value_id in _dedupe_values(values) if value_id in func_ir.values]


def _dedupe_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value_id in values:
        if value_id in seen:
            continue
        seen.add(value_id)
        ordered.append(value_id)
    return ordered


def _op_order_key(func_ir: FunctionIR, op_id: str) -> tuple[int, int, str]:
    op = func_ir.ops[op_id]
    block_index = list(func_ir.blocks.keys()).index(op.block_id)
    op_index = func_ir.blocks[op.block_id].op_ids.index(op_id)
    return (block_index, op_index, op_id)


def _value_order_key(func_ir: FunctionIR, value_id: str) -> tuple[int, int, str, int, str]:
    value = func_ir.values[value_id]
    if value.def_op is not None and value.def_op in func_ir.ops:
        op_key = _op_order_key(func_ir, value.def_op)
        return (*op_key, 0, value_id)
    use_keys = [
        _op_order_key(func_ir, use_op)
        for use_op in value.use_ops
        if use_op in func_ir.ops
    ]
    if use_keys:
        first_use = min(use_keys)
        return (*first_use, 1, value_id)
    return (10**9, 10**9, "", 2, value_id)
