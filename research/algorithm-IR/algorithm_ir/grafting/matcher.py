from __future__ import annotations

from algorithm_ir.grafting.skeletons import Skeleton
from algorithm_ir.region.contract import BoundaryContract
from algorithm_ir.region.selector import RewriteRegion
from algorithm_ir.ir.model import FunctionIR


def match_skeleton(
    func_ir: FunctionIR,
    region: RewriteRegion,
    contract: BoundaryContract,
    skeleton: Skeleton,
) -> bool:
    if skeleton.required_contract.get("requires_scalar_output"):
        has_scalar_target = False
        for value_id in contract.output_ports:
            value = func_ir.values[value_id]
            if value.attrs.get("var_name") == "score":
                has_scalar_target = True
                break
            if value.type_hint in {"int", "float", "bool"}:
                has_scalar_target = True
                break
        if not has_scalar_target:
            return False
    required_names = set(skeleton.required_contract.get("needs_inputs", []))
    # Search region boundary values, values referenced by region ops,
    # co-block values, and function parameters (always accessible).
    region_value_ids = set(region.entry_values + region.exit_values + region.state_carriers)
    region_block_ids = set()
    for op_id in region.op_ids:
        if op_id in func_ir.ops:
            op = func_ir.ops[op_id]
            region_value_ids.update(op.inputs)
            region_value_ids.update(op.outputs)
            region_block_ids.add(op.block_id)
    # Include all values from blocks that contain region ops
    for op in func_ir.ops.values():
        if op.block_id in region_block_ids:
            region_value_ids.update(op.inputs)
            region_value_ids.update(op.outputs)
    # Include function parameters (def_op is None) — always live
    for value_id, value in func_ir.values.items():
        if value.def_op is None:
            region_value_ids.add(value_id)
    available_names = {
        func_ir.values[value_id].attrs.get("var_name")
        for value_id in region_value_ids
        if value_id in func_ir.values
    }
    available_names.discard(None)
    return required_names <= available_names
