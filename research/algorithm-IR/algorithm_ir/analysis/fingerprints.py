from __future__ import annotations

from algorithm_ir.runtime.tracer import RuntimeValue


def fingerprint_runtime_value(runtime_value: RuntimeValue) -> tuple:
    metadata = runtime_value.metadata
    return (
        metadata.get("type_name"),
        tuple(sorted(metadata.get("keys", []))),
        metadata.get("shape"),
        metadata.get("container_size"),
    )

