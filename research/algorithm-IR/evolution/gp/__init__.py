"""Phase H+4 — typed Genetic Programming framework for slot evolution.

This package implements the IR-only typed GP framework described in
``code_review/typed_gp_remediation_plan.md``. All operations work
exclusively on FunctionIR; Python source is only emitted at the
evaluation boundary (evaluator.evaluate_source_quick).

Public surface:

* ``canonical_ir_hash(ir)`` — structural hash of FunctionIR
* ``resolve_slot_region(genome, slot_key)`` — three-tier region resolver
* ``SlotContract`` / ``TypedPort`` — contract-level type description
* ``SlotIndividual`` / ``MutationRecord`` — population members
* ``OperatorResult`` / ``Operator`` / ``run_operator_with_gates`` —
  uniform algorithm interface for typed mutations
* ``OPERATOR_REGISTRY`` — registered typed operators
* ``MicroPopulation`` — μ+λ evolutionary loop on one slot

Wiring: ``evolution.slot_evolution.step_slot_population`` delegates to
``MicroPopulation.evolve()`` once Phase S3.1 lands.
"""
from __future__ import annotations

from evolution.gp.canonical_hash import canonical_ir_hash, ir_structural_signature
from evolution.gp.region_resolver import (
    SlotRegionInfo,
    resolve_slot_region,
)
from evolution.gp.contract import SlotContract, TypedPort
from evolution.gp.individual import SlotIndividual
from evolution.gp.lineage import MutationRecord
from evolution.gp.operators.base import (
    Operator,
    OperatorResult,
    OperatorStats,
    OPERATOR_REGISTRY,
    register_operator,
    run_operator_with_gates,
)

__all__ = [
    "canonical_ir_hash",
    "ir_structural_signature",
    "SlotRegionInfo",
    "resolve_slot_region",
    "SlotContract",
    "TypedPort",
    "SlotIndividual",
    "MutationRecord",
    "Operator",
    "OperatorResult",
    "OperatorStats",
    "OPERATOR_REGISTRY",
    "register_operator",
    "run_operator_with_gates",
]
