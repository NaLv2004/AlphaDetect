"""SlotIndividual — one member of a typed-GP MicroPopulation.

Phase H+4 §4.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import inf

from algorithm_ir.ir.model import FunctionIR

from evolution.gp.contract import SlotContract
from evolution.gp.lineage import MutationRecord


@dataclass
class SlotIndividual:
    ir: FunctionIR
    contract: SlotContract
    fitness_train: float = inf
    fitness_val: float = inf
    novelty: float = 0.0
    complexity: int = 0
    ir_hash: str = ""
    behavior_hash: str = ""
    lineage: list[MutationRecord] = field(default_factory=list)
    operator_origin: str = "seed"
    parents: tuple[str, ...] = ()

    def is_better_than(self, other: "SlotIndividual", *, eps: float = 0.0) -> bool:
        """Strict-improvement comparator (lower fitness is better)."""
        return self.fitness_val + eps < other.fitness_val
