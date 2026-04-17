from __future__ import annotations

from algorithm_ir.ir import render_function_ir
from algorithm_ir.regeneration.artifact import AlgorithmArtifact


def emit_artifact_source(artifact: AlgorithmArtifact) -> str:
    return render_function_ir(artifact.ir)

