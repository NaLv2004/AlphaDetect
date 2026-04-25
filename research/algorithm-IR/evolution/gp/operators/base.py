"""Operator base — uniform contract for typed GP operators.

Phase H+4 §6 / §8.3.

Every typed GP operator implements the ``Operator`` protocol::

    class Operator(Protocol):
        name: str
        weight: float
        def propose(ctx, parent, parent2=None) -> OperatorResult: ...

Operators are registered via the ``@register_operator`` decorator and
collected in ``OPERATOR_REGISTRY``. The 8-gate runner
``run_operator_with_gates`` enforces the principle:

  1. child_ir is not None
  2. validate_function_ir(child_ir) == []
  3. ir_hash(child) != ir_hash(parent)         (no-op detection, IR-level)
  4. probe runs (delegated to evaluator)        — handled by caller
  5. probe output type matches contract         — handled by caller
  6. behavior_hash != parent.behavior_hash      — handled by caller
  7. complexity <= contract.complexity_cap

  (gate 4-6 are evaluator-side because they require running the IR;
   this module enforces 1, 2, 3, 7 — the structural / IR-only gates.)

No Python source is touched. Every step is on FunctionIR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.ir.validator import validate_function_ir

from evolution.gp.canonical_hash import canonical_ir_hash
from evolution.gp.contract import SlotContract


# ---------------------------------------------------------------------------
# Result + stats
# ---------------------------------------------------------------------------

@dataclass
class OperatorResult:
    """Output of one ``Operator.propose`` call."""
    child_ir: FunctionIR | None
    diff_summary: str = ""
    rejection_reason: str | None = None
    # Fields populated by run_operator_with_gates:
    child_hash: str = ""
    complexity: int = 0
    accepted_structurally: bool = False        # passed gates 1/2/3/7


@dataclass
class OperatorStats:
    """Per-operator counters within one MicroPopulation step."""
    name: str = ""
    n_attempted: int = 0
    n_proposed_none: int = 0          # gate 1
    n_validate_rejected: int = 0      # gate 2
    n_noop_ir: int = 0                # gate 3
    n_complexity_rejected: int = 0    # gate 7
    n_accepted_structurally: int = 0  # passed all 4 IR-side gates
    # Filled in by the population caller:
    n_probe_rejected: int = 0
    n_noop_behavior: int = 0
    n_evaluated: int = 0
    n_improved: int = 0


# ---------------------------------------------------------------------------
# Context object passed to operators
# ---------------------------------------------------------------------------

@dataclass
class GPContext:
    """Read-only context exposed to operators while proposing.

    Operators may inspect:
      - the parent IR (and parent2 for crossover)
      - the slot contract
      - the slot region op_ids (so they can scope their edits to ops
        actually inside the slot, not outside it)
      - the rng (for reproducibility)
    """
    contract: SlotContract
    region_op_ids: frozenset[str]
    rng: np.random.Generator
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol + registry
# ---------------------------------------------------------------------------

class Operator(Protocol):
    name: str
    weight: float

    def propose(
        self,
        ctx: GPContext,
        parent_ir: FunctionIR,
        parent2_ir: FunctionIR | None = None,
    ) -> OperatorResult: ...


# Maps operator name -> (factory, default weight, allowed_for_crossover)
OPERATOR_REGISTRY: dict[str, tuple[Callable[[], Operator], float, bool]] = {}


def register_operator(
    name: str,
    *,
    weight: float = 0.10,
    crossover: bool = False,
) -> Callable[[Callable[[], Operator]], Callable[[], Operator]]:
    """Decorator: register an Operator factory.

    Usage::

        @register_operator("mut_const", weight=0.20)
        def make_mut_const() -> Operator: ...
    """
    def deco(factory: Callable[[], Operator]) -> Callable[[], Operator]:
        OPERATOR_REGISTRY[name] = (factory, float(weight), bool(crossover))
        return factory
    return deco


# ---------------------------------------------------------------------------
# Complexity helper (op-count, used for the complexity gate)
# ---------------------------------------------------------------------------

def measure_complexity(ir: FunctionIR) -> int:
    """Trivial complexity metric — number of ops."""
    return len(ir.ops)


# ---------------------------------------------------------------------------
# 8-gate runner — IR-side only (gates 1/2/3/7)
# ---------------------------------------------------------------------------

def run_operator_with_gates(
    op: Operator,
    ctx: GPContext,
    parent_ir: FunctionIR,
    parent_hash: str,
    *,
    parent2_ir: FunctionIR | None = None,
    stats: OperatorStats | None = None,
) -> OperatorResult:
    """Invoke ``op.propose`` and apply IR-side gates 1/2/3/7.

    Behavior gates (4-6) require evaluator execution and are applied by
    the caller (MicroPopulation.step). Returns the OperatorResult with
    ``accepted_structurally`` and ``rejection_reason`` set.
    """
    if stats is not None:
        stats.n_attempted += 1

    try:
        result = op.propose(ctx, parent_ir, parent2_ir)
    except Exception as exc:
        if stats is not None:
            stats.n_proposed_none += 1
        return OperatorResult(
            child_ir=None,
            diff_summary="",
            rejection_reason=f"operator_raised:{type(exc).__name__}",
        )

    # Gate 1: child_ir is not None
    if result is None or result.child_ir is None:
        if stats is not None:
            stats.n_proposed_none += 1
        if result is None:
            return OperatorResult(
                child_ir=None,
                rejection_reason="operator_returned_none",
            )
        if result.rejection_reason is None:
            result.rejection_reason = "child_ir_none"
        return result

    child_ir = result.child_ir

    # Gate 2: validate_function_ir
    try:
        errs = validate_function_ir(child_ir)
    except Exception as exc:
        errs = [f"validate_raised:{type(exc).__name__}"]
    if errs:
        if stats is not None:
            stats.n_validate_rejected += 1
        result.child_ir = None
        result.rejection_reason = f"validate_failed:{len(errs)}_errs"
        return result

    # Gate 3: IR-level no-op detection
    child_hash = canonical_ir_hash(child_ir)
    result.child_hash = child_hash
    if child_hash == parent_hash:
        if stats is not None:
            stats.n_noop_ir += 1
        result.child_ir = None
        result.rejection_reason = "noop_ir_hash"
        return result

    # Gate 7: complexity cap
    complexity = measure_complexity(child_ir)
    result.complexity = complexity
    if complexity > ctx.contract.complexity_cap:
        if stats is not None:
            stats.n_complexity_rejected += 1
        result.child_ir = None
        result.rejection_reason = (
            f"complexity_cap:{complexity}>{ctx.contract.complexity_cap}"
        )
        return result

    if stats is not None:
        stats.n_accepted_structurally += 1
    result.accepted_structurally = True
    return result
