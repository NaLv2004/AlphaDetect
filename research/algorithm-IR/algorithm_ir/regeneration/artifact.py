from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from algorithm_ir.projection.base import Projection
from algorithm_ir.region.selector import RewriteRegion
from algorithm_ir.ir.model import FunctionIR


@dataclass
class AlgorithmArtifact:
    ir: FunctionIR
    source_code: str
    rewritten_regions: list[RewriteRegion]
    projections: list[Projection]
    provenance: dict[str, Any] = field(default_factory=dict)

