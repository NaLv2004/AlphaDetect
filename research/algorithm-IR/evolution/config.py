"""Evolution framework configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class EvolutionConfig:
    """All hyperparameters for the evolution engine.

    This is a generic configuration — no domain-specific fields.
    """

    # Population
    population_size: int = 100
    n_generations: int = 500
    seed: int = 42

    # Selection
    tournament_size: int = 5
    elite_count: int = 3

    # Variation
    mutation_rate: float = 0.8
    crossover_rate: float = 0.3
    constant_mutate_sigma: float = 0.1

    # Stagnation
    stagnation_threshold: int = 20
    hard_restart_after: int = 50

    # Hall of fame / diversity
    hall_of_fame_size: int = 10
    niche_radius: int = 3

    # Program structure
    program_roles: list[str] = field(default_factory=list)
    n_constants: int = 4
    constant_range: tuple[float, float] = (-3.0, 3.0)

    # Evaluation backend
    use_cpp: bool = True

    # Metric weights for composite fitness (metric_name → weight)
    metric_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict (JSON-safe)."""
        d = asdict(self)
        # tuple → list for JSON
        d["constant_range"] = list(d["constant_range"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvolutionConfig:
        """Deserialize from plain dict."""
        d = dict(d)  # copy
        if "constant_range" in d:
            d["constant_range"] = tuple(d["constant_range"])
        return cls(**d)
