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
    available_names = {
        func_ir.values[value_id].attrs.get("var_name")
        for value_id in region.entry_values + region.exit_values + region.state_carriers
    }
    available_names.update(
        value.attrs.get("var_name")
        for value in func_ir.values.values()
        if value.attrs.get("var_name") is not None
    )
    return required_names <= available_names
