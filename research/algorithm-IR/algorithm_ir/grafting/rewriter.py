from __future__ import annotations

import copy
from typing import Any

from algorithm_ir.grafting.matcher import match_skeleton
from algorithm_ir.grafting.skeletons import OverridePlan, Skeleton
from algorithm_ir.ir import render_function_ir, validate_function_ir
from algorithm_ir.ir.model import FunctionIR, Op, Value
from algorithm_ir.region.contract import BoundaryContract
from algorithm_ir.region.selector import RewriteRegion
from algorithm_ir.region.slicer import backward_slice_by_values
from algorithm_ir.regeneration.artifact import AlgorithmArtifact


def graft_skeleton(
    func_ir: FunctionIR,
    region: RewriteRegion,
    contract: BoundaryContract,
    skeleton: Skeleton,
    projection=None,
) -> AlgorithmArtifact:
    if not match_skeleton(func_ir, region, contract, skeleton):
        raise ValueError("BoundaryContract does not satisfy donor skeleton requirements")
    if skeleton.name != "bp_summary_update":
        raise NotImplementedError("Only the BP summary skeleton is implemented in the MVP")

    new_ir = copy.deepcopy(func_ir)
    target_output = _select_target_output(new_ir, contract)
    score_slice = set(backward_slice_by_values(new_ir, [target_output])) & set(region.op_ids)
    removed_ops, preserved_bindings = _partition_removable_ops(new_ir, score_slice, target_output)
    insertion_block = new_ir.ops[new_ir.values[target_output].def_op].block_id

    frontier_value = _find_named_value(new_ir, region, "frontier")
    costs_value = _find_named_value(new_ir, region, "costs")
    candidate_value = _find_named_value(new_ir, region, "candidate")
    if not (frontier_value and costs_value and candidate_value):
        raise ValueError("Could not resolve frontier/costs/candidate bindings for BP graft")

    new_op_ids, reconnect_map = _insert_bp_summary_ops(
        new_ir,
        insertion_block=insertion_block,
        target_output=target_output,
        frontier_value=frontier_value,
        costs_value=costs_value,
        candidate_value=candidate_value,
        donor_callable=skeleton.donor_callable,
        damping=float(skeleton.lowering_template["damping"]),
    )

    _remove_ops(new_ir, removed_ops)
    _splice_block_ops(new_ir, insertion_block, target_output, removed_ops, new_op_ids)
    _recompute_use_def(new_ir)

    errors = validate_function_ir(new_ir)
    if errors:
        raise ValueError("Rewritten IR is invalid:\n" + "\n".join(errors))

    override_plan = OverridePlan(
        plan_id=f"override_{region.region_id}",
        target_region_id=region.region_id,
        removed_op_ids=sorted(removed_ops),
        preserved_bindings=preserved_bindings,
        new_state_defs=[],
        schedule_insertions=[{"block_id": insertion_block, "new_op_ids": new_op_ids}],
        reconnect_map=reconnect_map,
        projection_id=getattr(projection, "proj_id", None),
    )

    return AlgorithmArtifact(
        ir=new_ir,
        source_code=render_function_ir(new_ir),
        rewritten_regions=[region],
        projections=[projection] if projection is not None else [],
        provenance={
            "host_function": func_ir.name,
            "donor_function": skeleton.donor_callable.__name__ if skeleton.donor_callable else None,
            "rewrite_region": region.region_id,
            "boundary_contract": contract.contract_id,
            "override_plan": override_plan,
        },
    )


def _select_target_output(func_ir: FunctionIR, contract: BoundaryContract) -> str:
    for value_id in contract.output_ports:
        if func_ir.values[value_id].attrs.get("var_name") == "score":
            return value_id
    if len(contract.output_ports) == 1:
        return contract.output_ports[0]
    raise ValueError("Could not select target output for rewrite")


def _partition_removable_ops(
    func_ir: FunctionIR,
    candidate_ops: set[str],
    target_output: str,
) -> tuple[set[str], dict[str, str]]:
    preserved_outputs: list[str] = []
    preserved_bindings: dict[str, str] = {}
    for op_id in candidate_ops:
        op = func_ir.ops[op_id]
        for output in op.outputs:
            outside_uses = [use for use in func_ir.values[output].use_ops if use not in candidate_ops]
            if output != target_output and outside_uses:
                preserved_outputs.append(output)
                preserved_bindings[output] = op_id

    preserved_ops = set(backward_slice_by_values(func_ir, preserved_outputs)) & candidate_ops
    removable = set(candidate_ops) - preserved_ops
    return removable, preserved_bindings


def _insert_bp_summary_ops(
    func_ir: FunctionIR,
    *,
    insertion_block: str,
    target_output: str,
    frontier_value: str,
    costs_value: str,
    candidate_value: str,
    donor_callable,
    damping: float,
) -> tuple[list[str], dict[str, str]]:
    op_ids: list[str] = []
    donor_const = _new_value(func_ir, "bp_callable", "builtin_function_or_method", {"literal": donor_callable})
    op_ids.append(_new_op(func_ir, "const", [], [donor_const], insertion_block, {"literal": donor_callable, "name": donor_callable.__name__}))

    damping_const = _new_value(func_ir, "damping", "float", {"literal": damping})
    op_ids.append(_new_op(func_ir, "const", [], [damping_const], insertion_block, {"literal": damping}))

    summary_value = _new_value(func_ir, "bp_summary", "float", {})
    op_ids.append(_new_op(func_ir, "call", [donor_const, frontier_value, costs_value, damping_const], [summary_value], insertion_block, {"n_args": 3, "grafted": True}))

    metric_key = _new_value(func_ir, "metric_key", "str", {"literal": "metric"})
    op_ids.append(_new_op(func_ir, "const", [], [metric_key], insertion_block, {"literal": "metric"}))

    candidate_metric = _new_value(func_ir, "candidate_metric", "float", {})
    op_ids.append(_new_op(func_ir, "get_item", [candidate_value, metric_key], [candidate_metric], insertion_block, {"grafted": True}))

    score_temp = _new_value(func_ir, "bp_score_temp", "float", {})
    op_ids.append(_new_op(func_ir, "binary", [candidate_metric, summary_value], [score_temp], insertion_block, {"operator": "Add", "grafted": True}))

    op_ids.append(_new_op(func_ir, "assign", [score_temp], [target_output], insertion_block, {"target": "score", "grafted": True}))
    return op_ids, {target_output: "score"}


def _find_named_value(func_ir: FunctionIR, region: RewriteRegion, name: str) -> str | None:
    candidate_ids = region.entry_values + region.exit_values + region.state_carriers
    for value_id in candidate_ids:
        value = func_ir.values[value_id]
        if value.attrs.get("var_name") == name:
            return value_id
    for value_id, value in func_ir.values.items():
        if value.attrs.get("var_name") == name:
            return value_id
    return None


def _remove_ops(func_ir: FunctionIR, removed_ops: set[str]) -> None:
    for op_id in removed_ops:
        func_ir.ops.pop(op_id, None)
    for block in func_ir.blocks.values():
        block.op_ids = [op_id for op_id in block.op_ids if op_id not in removed_ops]


def _splice_block_ops(
    func_ir: FunctionIR,
    block_id: str,
    target_output: str,
    removed_ops: set[str],
    new_op_ids: list[str],
) -> None:
    block = func_ir.blocks[block_id]
    consumer_ops = [
        op_id
        for op_id in block.op_ids
        if target_output in func_ir.ops[op_id].inputs and op_id not in removed_ops
    ]
    if consumer_ops:
        insert_index = block.op_ids.index(consumer_ops[0])
    else:
        insert_index = len(block.op_ids)
    block.op_ids[insert_index:insert_index] = new_op_ids


def _recompute_use_def(func_ir: FunctionIR) -> None:
    for value in func_ir.values.values():
        value.def_op = None
        value.use_ops = []
    for op_id, op in func_ir.ops.items():
        for value_id in op.inputs:
            func_ir.values[value_id].use_ops.append(op_id)
        for value_id in op.outputs:
            func_ir.values[value_id].def_op = op_id


def _new_value(func_ir: FunctionIR, name_hint: str, type_hint: str, attrs: dict[str, Any]) -> str:
    existing = [int(value_id.split("_")[1]) for value_id in func_ir.values if value_id.startswith("v_")]
    next_id = max(existing, default=-1) + 1
    value_id = f"v_{next_id}"
    func_ir.values[value_id] = Value(
        id=value_id,
        name_hint=name_hint,
        type_hint=type_hint,
        source_span=None,
        attrs=dict(attrs),
    )
    return value_id


def _new_op(
    func_ir: FunctionIR,
    opcode: str,
    inputs: list[str],
    outputs: list[str],
    block_id: str,
    attrs: dict[str, Any],
) -> str:
    existing = [int(op_id.split("_")[1]) for op_id in func_ir.ops if op_id.startswith("op_")]
    next_id = max(existing, default=-1) + 1
    op_id = f"op_{next_id}"
    func_ir.ops[op_id] = Op(
        id=op_id,
        opcode=opcode,
        inputs=list(inputs),
        outputs=list(outputs),
        block_id=block_id,
        source_span=None,
        attrs=dict(attrs),
    )
    return op_id
