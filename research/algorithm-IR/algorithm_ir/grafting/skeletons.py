from __future__ import annotations

from dataclasses import dataclass, field
from types import FunctionType
from typing import Any

from algorithm_ir.frontend import compile_function_to_ir


@dataclass
class Skeleton:
    skel_id: str
    name: str
    required_contract: dict[str, Any]
    transform_rules: list[dict[str, Any]]
    lowering_template: dict[str, Any]
    optional_projection_hints: dict[str, Any] = field(default_factory=dict)
    donor_callable: FunctionType | None = None
    donor_ir: object | None = None


@dataclass
class OverridePlan:
    plan_id: str
    target_region_id: str
    removed_op_ids: list[str]
    preserved_bindings: dict[str, str]
    new_state_defs: list[dict[str, Any]]
    schedule_insertions: list[dict[str, Any]]
    reconnect_map: dict[str, str]
    projection_id: str | None


def make_bp_summary_skeleton(bp_fn: FunctionType, damping: float = 0.1) -> Skeleton:
    return Skeleton(
        skel_id=f"skel_{bp_fn.__name__}",
        name="bp_summary_update",
        required_contract={
            "needs_inputs": ["frontier", "costs", "candidate"],
            "requires_scalar_output": True,
        },
        transform_rules=[{"kind": "replace_score_with_bp_summary"}],
        lowering_template={"damping": damping},
        optional_projection_hints={"preferred_family": "local_interaction"},
        donor_callable=bp_fn,
        donor_ir=compile_function_to_ir(bp_fn),
    )


def make_bp_tree_runtime_skeleton(bp_fn: FunctionType, damping: float = 0.1) -> Skeleton:
    return Skeleton(
        skel_id=f"skel_{bp_fn.__name__}",
        name="bp_tree_runtime_update",
        required_contract={
            "needs_inputs": ["explored", "frontier", "costs"],
            "requires_scalar_output": False,
        },
        transform_rules=[{"kind": "run_bp_after_expansion"}],
        lowering_template={"damping": damping},
        optional_projection_hints={"preferred_family": "scheduling"},
        donor_callable=bp_fn,
        donor_ir=compile_function_to_ir(bp_fn),
    )
