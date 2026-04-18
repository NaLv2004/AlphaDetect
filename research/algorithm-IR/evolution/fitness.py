"""Abstract fitness evaluation interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evolution.genome import IRGenome


@dataclass
class FitnessResult:
    """Generic fitness result with arbitrary named metrics.

    No domain-specific fields — metrics are stored as a dict.
    Composite score is a weighted sum of metrics (lower is better).
    """

    metrics: dict[str, float] = field(default_factory=dict)
    is_valid: bool = True
    weights: dict[str, float] = field(default_factory=dict)

    def composite_score(self) -> float:
        """Weighted sum of metrics (lower is better)."""
        if not self.is_valid:
            return float("inf")
        score = 0.0
        for name, value in self.metrics.items():
            w = self.weights.get(name, 1.0)
            score += w * value
        return score

    def __lt__(self, other: FitnessResult) -> bool:
        return self.composite_score() < other.composite_score()

    def __le__(self, other: FitnessResult) -> bool:
        return self.composite_score() <= other.composite_score()

    def __repr__(self) -> str:
        return (
            f"FitnessResult(score={self.composite_score():.6f}, "
            f"valid={self.is_valid}, metrics={self.metrics})"
        )


class FitnessEvaluator(ABC):
    """Abstract base for domain-specific fitness evaluation."""

    @abstractmethod
    def evaluate(self, genome: IRGenome) -> FitnessResult:
        """Evaluate a single genome."""
        ...

    def evaluate_batch(self, genomes: list[IRGenome]) -> list[FitnessResult]:
        """Evaluate a batch of genomes. Default: sequential."""
        return [self.evaluate(g) for g in genomes]
