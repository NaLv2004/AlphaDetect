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

    return BoundaryContract(
        contract_id=f"contract_{region.region_id}",
        region_id=region.region_id,
        input_ports=list(region.entry_values),
        output_ports=list(region.exit_values),
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

