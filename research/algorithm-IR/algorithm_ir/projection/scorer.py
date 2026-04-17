from __future__ import annotations

from algorithm_ir.factgraph.model import FactGraph
from algorithm_ir.projection.base import Projection
from algorithm_ir.projection.local_interaction import detect_local_interaction_projection
from algorithm_ir.projection.scheduling import detect_scheduling_projection
from algorithm_ir.region.selector import RewriteRegion


def annotate_region(
    region: RewriteRegion,
    fg: FactGraph | None = None,
) -> list[Projection]:
    projections: list[Projection] = []
    for detector in (detect_scheduling_projection, detect_local_interaction_projection):
        projection = detector(region)
        if projection is not None:
            projections.append(projection)
    if fg is not None:
        for projection in projections:
            projection.evidence["factgraph_function"] = fg.metadata.get("function_name")
    return projections

