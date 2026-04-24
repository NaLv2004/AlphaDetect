from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from algorithm_ir.analysis.dynamic_analysis import runtime_values_for_static
from algorithm_ir.region.selector import RewriteRegion
from algorithm_ir.runtime.tracer import RuntimeEvent, RuntimeValue
from algorithm_ir.ir.model import FunctionIR


@dataclass
class BoundaryContract:
    contract_id: str
    region_id: str
    input_ports: list[str]
    output_ports: list[str]
    normalized_input_ports: list[str]
    normalized_output_ports: list[str]
    port_signature: dict[str, Any]
    port_order_evidence: dict[str, Any]
    readable_slots: list[str]
    writable_slots: list[str]
    new_state_policy: dict[str, Any]
    reconnect_points: dict[str, list[str]]
    invariants: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)
    dependency_overrides: list[Any] = field(default_factory=list)


def infer_boundary_contract(
    func_ir: FunctionIR,
    region: RewriteRegion,
    runtime_trace: list[RuntimeEvent] | None = None,
    runtime_values: dict[str, RuntimeValue] | None = None,
) -> BoundaryContract:
    reconnect_points: dict[str, list[str]] = {}
    region_op_set = set(region.op_ids)
    for value_id in region.exit_values:
        reconnect_points[value_id] = [
            use_op for use_op in func_ir.values[value_id].use_ops if use_op not in region_op_set
        ]

    normalized_inputs = sorted(
        region.entry_values,
        key=lambda vid: _input_port_key(func_ir, region, vid),
    )
    normalized_outputs = sorted(
        region.exit_values,
        key=lambda vid: _output_port_key(func_ir, reconnect_points, vid),
    )

    scalar_like = True
    observed_types: set[str] = set()
    if runtime_values is not None:
        for value_id in region.exit_values:
            for runtime_value in runtime_values_for_static(runtime_values, value_id):
                observed_types.add(runtime_value.metadata.get("type_name", "object"))
        if observed_types:
            scalar_like = observed_types <= {"int", "float", "bool"}

    invariants = {
        "output_count": len(region.exit_values),
        "scalar_outputs": scalar_like,
        "preserve_comparability": any(
            func_ir.ops[use_op].opcode == "compare"
            for value_id in region.exit_values
            for use_op in func_ir.values[value_id].use_ops
            if use_op not in region_op_set
        ),
    }

    evidence = {
        "region_entry_values": list(region.entry_values),
        "region_exit_values": list(region.exit_values),
        "observed_types": sorted(observed_types),
        "runtime_trace_events": len(runtime_trace or []),
    }
    port_signature = {
        "inputs": [_port_signature_entry(func_ir, vid, "input") for vid in normalized_inputs],
        "outputs": [_port_signature_entry(func_ir, vid, "output") for vid in normalized_outputs],
    }
    port_order_evidence = {
        "inputs": {
            vid: _input_port_key(func_ir, region, vid)
            for vid in normalized_inputs
        },
        "outputs": {
            vid: _output_port_key(func_ir, reconnect_points, vid)
            for vid in normalized_outputs
        },
    }

    return BoundaryContract(
        contract_id=f"contract_{region.region_id}",
        region_id=region.region_id,
        input_ports=list(region.entry_values),
        output_ports=list(region.exit_values),
        normalized_input_ports=normalized_inputs,
        normalized_output_ports=normalized_outputs,
        port_signature=port_signature,
        port_order_evidence=port_order_evidence,
        readable_slots=list(region.read_set),
        writable_slots=list(region.write_set),
        new_state_policy={
            "allowed": region.allows_new_state,
            "state_carriers": list(region.state_carriers),
        },
        reconnect_points=reconnect_points,
        invariants=invariants,
        evidence=evidence,
    )


def _input_port_key(func_ir: FunctionIR, region: RewriteRegion, value_id: str) -> tuple:
    consumer_positions = []
    for use_op in func_ir.values[value_id].use_ops:
        if use_op in set(region.op_ids):
            consumer_positions.append(_op_position(func_ir, use_op))
    first_consumer = min(consumer_positions) if consumer_positions else (10**9, 10**9)
    value = func_ir.values[value_id]
    return (
        first_consumer[0],
        first_consumer[1],
        value.type_hint or "",
        len(value.use_ops),
        value_id,
    )


def _output_port_key(
    func_ir: FunctionIR,
    reconnect_points: dict[str, list[str]],
    value_id: str,
) -> tuple:
    use_positions = [
        _op_position(func_ir, use_op)
        for use_op in reconnect_points.get(value_id, [])
        if use_op in func_ir.ops
    ]
    first_external_use = min(use_positions) if use_positions else (10**9, 10**9)
    value = func_ir.values[value_id]
    def_pos = _op_position(func_ir, value.def_op) if value.def_op in func_ir.ops else (10**9, 10**9)
    return (
        first_external_use[0],
        first_external_use[1],
        def_pos[0],
        def_pos[1],
        value.type_hint or "",
        value_id,
    )


def _port_signature_entry(func_ir: FunctionIR, value_id: str, direction: str) -> dict[str, Any]:
    value = func_ir.values[value_id]
    external_uses = len(value.use_ops)
    return {
        "direction": direction,
        "type_hint": value.type_hint,
        "n_uses": external_uses,
        "is_control_related": any(
            func_ir.ops[use_op].opcode == "branch"
            for use_op in value.use_ops
            if use_op in func_ir.ops
        ),
        "is_effect_related": any(
            func_ir.ops[use_op].opcode in {"set_attr", "set_item", "append", "pop"}
            or (
                func_ir.ops[use_op].opcode == "call"
                and (
                    func_ir.ops[use_op].attrs.get("effectful")
                    or func_ir.ops[use_op].attrs.get("has_side_effect")
                    or func_ir.ops[use_op].attrs.get("escapes")
                )
            )
            for use_op in value.use_ops
            if use_op in func_ir.ops
        ),
    }


def _op_position(func_ir: FunctionIR, op_id: str) -> tuple[int, int]:
    op = func_ir.ops[op_id]
    block_order = list(func_ir.blocks.keys()).index(op.block_id)
    op_order = func_ir.blocks[op.block_id].op_ids.index(op_id)
    return block_order, op_order
