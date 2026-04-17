from __future__ import annotations

from algorithm_ir.projection.base import Projection
from algorithm_ir.region.selector import RewriteRegion


def detect_scheduling_projection(region: RewriteRegion) -> Projection | None:
    has_candidate_pool = any("container:" in item for item in region.write_set) or len(region.state_carriers) > 0
    supports_selection = any("item:" in item for item in region.read_set)
    if not has_candidate_pool:
        return None
    return Projection(
        proj_id=f"proj_sched_{region.region_id}",
        region_id=region.region_id,
        family="scheduling",
        node_set=list(region.state_carriers),
        edge_set=[],
        evidence={"read_set": list(region.read_set), "write_set": list(region.write_set)},
        interface={
            "has_candidate_pool": has_candidate_pool,
            "supports_selection": supports_selection,
            "supports_reinsertion": any("container:" in item for item in region.write_set),
            "supports_local_score": any("item:" in item for item in region.read_set),
        },
        score=0.6,
    )

