from __future__ import annotations

from algorithm_ir.runtime.tracer import RuntimeValue


def runtime_values_for_static(
    runtime_values: dict[str, RuntimeValue],
    static_value_id: str,
) -> list[RuntimeValue]:
    return [
        value
        for value in runtime_values.values()
        if value.static_value_id == static_value_id
    ]

