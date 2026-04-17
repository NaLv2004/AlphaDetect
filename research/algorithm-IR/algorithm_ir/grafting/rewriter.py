from __future__ import annotations

from typing import Any

from algorithm_ir.grafting.matcher import match_skeleton
from algorithm_ir.grafting.skeletons import OverridePlan, Skeleton
from algorithm_ir.ir import render_function_ir, validate_function_ir
from algorithm_ir.ir.model import FunctionIR, Op, Value
from algorithm_ir.ir.xdsl_bridge import create_xdsl_op_from_payload
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
    if skeleton.name == "bp_summary_update":
        return _graft_bp_summary(func_ir, region, contract, skeleton, projection)
    if skeleton.name == "bp_tree_runtime_update":
        return _graft_bp_tree_runtime_update(func_ir, region, contract, skeleton, projection)
    raise NotImplementedError(f"Unsupported skeleton {skeleton.name}")


def _graft_bp_summary(
    func_ir: FunctionIR,
    region: RewriteRegion,
    contract: BoundaryContract,
    skeleton: Skeleton,
    projection,
) -> AlgorithmArtifact:
    new_ir = func_ir.clone()
    target_output = _select_target_output(new_ir, contract)
    score_slice = set(backward_slice_by_values(new_ir, [target_output])) & set(region.op_ids)
    removed_ops, preserved_bindings = _partition_removable_ops(new_ir, score_slice, target_output)
    insertion_block = new_ir.ops[new_ir.values[target_output].def_op].block_id

    frontier_value = _find_named_value(new_ir, region, "frontier")
    costs_value = _find_named_value(new_ir, region, "costs")
    candidate_value = _find_named_value(new_ir, region, "candidate")
    if not (frontier_value and costs_value and candidate_value):
        raise ValueError("Could not resolve frontier/costs/candidate bindings for BP graft")

    new_op_ids, new_xdsl_ops, reconnect_map = _build_bp_summary_ops(
        new_ir,
        insertion_block=insertion_block,
        target_output=target_output,
        frontier_value=frontier_value,
        costs_value=costs_value,
        candidate_value=candidate_value,
        donor_callable=skeleton.donor_callable,
        damping=float(skeleton.lowering_template["damping"]),
    )

    _insert_ops_before_consumer_or_terminator(
        new_ir,
        block_id=insertion_block,
        target_output=target_output,
        removed_ops=removed_ops,
        new_xdsl_ops=new_xdsl_ops,
    )
    _erase_ops_from_xdsl(new_ir, removed_ops)
    rewritten_ir = _rebuild_from_xdsl(new_ir)
    _ensure_valid(rewritten_ir)

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
        ir=rewritten_ir,
        source_code=render_function_ir(rewritten_ir),
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


def _graft_bp_tree_runtime_update(
    func_ir: FunctionIR,
    region: RewriteRegion,
    contract: BoundaryContract,
    skeleton: Skeleton,
    projection,
) -> AlgorithmArtifact:
    new_ir = func_ir.clone()
    insertion_block = _select_insertion_block(new_ir, region)

    explored_value = _find_named_value(new_ir, region, "explored")
    frontier_value = _find_named_value(new_ir, region, "frontier")
    costs_value = _find_named_value(new_ir, region, "costs")
    audit_value = _find_named_value(new_ir, region, "audit")
    if not (explored_value and frontier_value and costs_value and audit_value):
        raise ValueError("Could not resolve explored/frontier/costs/audit bindings for runtime BP graft")

    new_op_ids, new_xdsl_ops, reconnect_map = _build_bp_runtime_ops(
        new_ir,
        insertion_block=insertion_block,
        explored_value=explored_value,
        frontier_value=frontier_value,
        costs_value=costs_value,
        audit_value=audit_value,
        donor_callable=skeleton.donor_callable,
        damping=float(skeleton.lowering_template["damping"]),
    )

    _insert_ops_before_terminator(new_ir, insertion_block, new_xdsl_ops)
    rewritten_ir = _rebuild_from_xdsl(new_ir)
    _ensure_valid(rewritten_ir)

    override_plan = OverridePlan(
        plan_id=f"override_{region.region_id}",
        target_region_id=region.region_id,
        removed_op_ids=[],
        preserved_bindings={},
        new_state_defs=[],
        schedule_insertions=[{"block_id": insertion_block, "new_op_ids": new_op_ids}],
        reconnect_map=reconnect_map,
        projection_id=getattr(projection, "proj_id", None),
    )

    return AlgorithmArtifact(
        ir=rewritten_ir,
        source_code=render_function_ir(rewritten_ir),
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


def _build_bp_summary_ops(
    func_ir: FunctionIR,
    *,
    insertion_block: str,
    target_output: str,
    frontier_value: str,
    costs_value: str,
    candidate_value: str,
    donor_callable,
    damping: float,
) -> tuple[list[str], list[Any], dict[str, str]]:
    op_ids: list[str] = []
    xdsl_ops: list[Any] = []

    donor_const = _new_value(
        func_ir,
        "bp_callable",
        "builtin_function_or_method",
        {"literal": donor_callable},
    )
    op_ids.append(
        _append_xdsl_op(
            func_ir,
            xdsl_ops,
            opcode="const",
            inputs=[],
            outputs=[donor_const],
            block_id=insertion_block,
            attrs={"literal": donor_callable, "name": donor_callable.__name__},
        )
    )

    damping_const = _new_value(func_ir, "damping", "float", {"literal": damping})
    op_ids.append(
        _append_xdsl_op(
            func_ir,
            xdsl_ops,
            opcode="const",
            inputs=[],
            outputs=[damping_const],
            block_id=insertion_block,
            attrs={"literal": damping},
        )
    )

    summary_value = _new_value(func_ir, "bp_summary", "float", {})
    op_ids.append(
        _append_xdsl_op(
            func_ir,
            xdsl_ops,
            opcode="call",
            inputs=[donor_const, frontier_value, costs_value, damping_const],
            outputs=[summary_value],
            block_id=insertion_block,
            attrs={"n_args": 3, "grafted": True},
        )
    )

    metric_key = _new_value(func_ir, "metric_key", "str", {"literal": "metric"})
    op_ids.append(
        _append_xdsl_op(
            func_ir,
            xdsl_ops,
            opcode="const",
            inputs=[],
            outputs=[metric_key],
            block_id=insertion_block,
            attrs={"literal": "metric"},
        )
    )

    candidate_metric = _new_value(func_ir, "candidate_metric", "float", {})
    op_ids.append(
        _append_xdsl_op(
            func_ir,
            xdsl_ops,
            opcode="get_item",
            inputs=[candidate_value, metric_key],
            outputs=[candidate_metric],
            block_id=insertion_block,
            attrs={"grafted": True},
        )
    )

    score_temp = _new_value(func_ir, "bp_score_temp", "float", {})
    op_ids.append(
        _append_xdsl_op(
            func_ir,
            xdsl_ops,
            opcode="binary",
            inputs=[candidate_metric, summary_value],
            outputs=[score_temp],
            block_id=insertion_block,
            attrs={"operator": "Add", "grafted": True},
        )
    )

    op_ids.append(
        _append_xdsl_op(
            func_ir,
            xdsl_ops,
            opcode="assign",
            inputs=[score_temp],
            outputs=[target_output],
            block_id=insertion_block,
            attrs={"target": "score", "grafted": True},
        )
    )

    return op_ids, xdsl_ops, {target_output: "score"}


def _build_bp_runtime_ops(
    func_ir: FunctionIR,
    *,
    insertion_block: str,
    explored_value: str,
    frontier_value: str,
    costs_value: str,
    audit_value: str,
    donor_callable,
    damping: float,
) -> tuple[list[str], list[Any], dict[str, str]]:
    op_ids: list[str] = []
    xdsl_ops: list[Any] = []

    donor_const = _new_value(
        func_ir,
        "bp_runtime_callable",
        "builtin_function_or_method",
        {"literal": donor_callable},
    )
    op_ids.append(
        _append_xdsl_op(
            func_ir,
            xdsl_ops,
            opcode="const",
            inputs=[],
            outputs=[donor_const],
            block_id=insertion_block,
            attrs={"literal": donor_callable, "name": donor_callable.__name__},
        )
    )

    damping_const = _new_value(func_ir, "runtime_damping", "float", {"literal": damping})
    op_ids.append(
        _append_xdsl_op(
            func_ir,
            xdsl_ops,
            opcode="const",
            inputs=[],
            outputs=[damping_const],
            block_id=insertion_block,
            attrs={"literal": damping},
        )
    )

    bp_result = _new_value(func_ir, "bp_runtime_result", "float", {})
    op_ids.append(
        _append_xdsl_op(
            func_ir,
            xdsl_ops,
            opcode="call",
            inputs=[donor_const, explored_value, frontier_value, costs_value, audit_value, damping_const],
            outputs=[bp_result],
            block_id=insertion_block,
            attrs={"n_args": 5, "grafted": True, "runtime_bp_pass": True},
        )
    )

    return op_ids, xdsl_ops, {"runtime_bp_result": bp_result}


def _append_xdsl_op(
    func_ir: FunctionIR,
    xdsl_ops: list[Any],
    *,
    opcode: str,
    inputs: list[str],
    outputs: list[str],
    block_id: str,
    attrs: dict[str, Any],
) -> str:
    op_id = _next_prefixed_id(func_ir.ops, "op_")
    temp_op = Op(
        id=op_id,
        opcode=opcode,
        inputs=list(inputs),
        outputs=list(outputs),
        block_id=block_id,
        source_span=None,
        attrs=dict(attrs),
    )
    func_ir.ops[op_id] = temp_op
    payload = {
        "id": op_id,
        "opcode": opcode,
        "block_id": block_id,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "source_span": None,
        "attrs": dict(attrs),
        "output_meta": [_value_payload(func_ir.values[value_id]) for value_id in outputs],
    }
    xdsl_op = create_xdsl_op_from_payload(
        opcode=opcode,
        payload=payload,
        result_type_hints=[func_ir.values[value_id].type_hint for value_id in outputs],
    )
    func_ir.xdsl_op_map[op_id] = xdsl_op
    xdsl_ops.append(xdsl_op)
    return op_id


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


def _select_insertion_block(func_ir: FunctionIR, region: RewriteRegion) -> str:
    candidate_blocks = []
    for block_id in region.block_ids:
        block = func_ir.blocks[block_id]
        if any(func_ir.ops[op_id].attrs.get("target") == "left" for op_id in block.op_ids):
            candidate_blocks.append(block_id)
    if candidate_blocks:
        return candidate_blocks[0]
    return region.block_ids[-1]


def _insert_ops_before_consumer_or_terminator(
    func_ir: FunctionIR,
    *,
    block_id: str,
    target_output: str,
    removed_ops: set[str],
    new_xdsl_ops: list[Any],
) -> None:
    block = func_ir.blocks[block_id]
    consumer_op_id = next(
        (
            op_id
            for op_id in block.op_ids
            if op_id not in removed_ops and target_output in func_ir.ops[op_id].inputs
        ),
        None,
    )
    xdsl_block = func_ir.xdsl_block_map[block_id]
    if consumer_op_id is not None:
        xdsl_block.insert_ops_before(new_xdsl_ops, func_ir.xdsl_op_map[consumer_op_id])
        return
    _insert_ops_before_terminator(func_ir, block_id, new_xdsl_ops)


def _insert_ops_before_terminator(func_ir: FunctionIR, block_id: str, new_xdsl_ops: list[Any]) -> None:
    block = func_ir.blocks[block_id]
    xdsl_block = func_ir.xdsl_block_map[block_id]
    terminator_op_id = next(
        (
            op_id
            for op_id in block.op_ids
            if func_ir.ops[op_id].opcode in {"jump", "branch", "return"}
        ),
        None,
    )
    if terminator_op_id is not None:
        xdsl_block.insert_ops_before(new_xdsl_ops, func_ir.xdsl_op_map[terminator_op_id])
    else:
        xdsl_block.add_ops(new_xdsl_ops)


def _erase_ops_from_xdsl(func_ir: FunctionIR, removed_ops: set[str]) -> None:
    for op_id in removed_ops:
        xdsl_op = func_ir.xdsl_op_map.get(op_id)
        if xdsl_op is None:
            continue
        parent_block = xdsl_op.parent_block()
        if parent_block is not None:
            parent_block.erase_op(xdsl_op, safe_erase=False)


def _rebuild_from_xdsl(func_ir: FunctionIR) -> FunctionIR:
    return FunctionIR.from_xdsl(func_ir.xdsl_module)


def _ensure_valid(func_ir: FunctionIR) -> None:
    errors = validate_function_ir(func_ir)
    if errors:
        raise ValueError("Rewritten IR is invalid:\n" + "\n".join(errors))


def _new_value(func_ir: FunctionIR, name_hint: str, type_hint: str, attrs: dict[str, Any]) -> str:
    value_id = _next_prefixed_id(func_ir.values, "v_")
    func_ir.values[value_id] = Value(
        id=value_id,
        name_hint=name_hint,
        type_hint=type_hint,
        source_span=None,
        attrs=dict(attrs),
    )
    return value_id


def _next_prefixed_id(existing: dict[str, Any], prefix: str) -> str:
    next_id = max(
        (int(item_id.split("_", maxsplit=1)[1]) for item_id in existing if item_id.startswith(prefix)),
        default=-1,
    ) + 1
    return f"{prefix}{next_id}"


def _value_payload(value: Value) -> dict[str, Any]:
    return {
        "id": value.id,
        "name_hint": value.name_hint,
        "type_hint": value.type_hint,
        "source_span": value.source_span,
        "attrs": dict(value.attrs),
    }
