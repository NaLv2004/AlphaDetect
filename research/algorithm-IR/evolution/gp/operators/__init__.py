"""Typed GP operators (sub-package)."""
from __future__ import annotations

from evolution.gp.operators.base import (
    GPContext,
    Operator,
    OperatorResult,
    OperatorStats,
    OPERATOR_REGISTRY,
    register_operator,
    run_operator_with_gates,
    measure_complexity,
)

# Importing typed_mutations registers operators in OPERATOR_REGISTRY.
from evolution.gp.operators import typed_mutations as _typed_mutations  # noqa: F401

__all__ = [
    "GPContext",
    "Operator",
    "OperatorResult",
    "OperatorStats",
    "OPERATOR_REGISTRY",
    "register_operator",
    "run_operator_with_gates",
    "measure_complexity",
]
