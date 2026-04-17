from __future__ import annotations

from algorithm_ir.projection.base import Projection
from algorithm_ir.region.selector import RewriteRegion


def detect_local_interaction_projection(region: RewriteRegion) -> Projection | None:
    if not region.entry_values or not region.exit_values:
        return None
    return Projection(
        proj_id=f"proj_local_{region.region_id}",
        region_id=region.region_id,
        family="local_interaction",
        node_set=list(region.entry_values) + list(region.exit_values),
        edge_set=[],
        evidence={"entry_values": list(region.entry_values), "exit_values": list(region.exit_values)},
        interface={
            "has_local_neighbors": len(region.state_carriers) > 0,
            "supports_iterative_local_update": len(region.schedule_anchors.get("loop_blocks", [])) > 0,
            "supports_summary_extraction": len(region.exit_values) > 0,
        },
        score=0.7,
    )

