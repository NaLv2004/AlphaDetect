"""Main evolution engine.

Generic evolutionary algorithm with tournament selection, elitism,
niching, stagnation detection, hard restarts, and hall of fame.
All application-specific logic lives in the FitnessEvaluator.
"""

from __future__ import annotations

import time
import logging
from typing import Any

import numpy as np

from evolution.config import EvolutionConfig
from evolution.fitness import FitnessResult, FitnessEvaluator
from evolution.genome import IRGenome
from evolution.operators import mutate_genome, crossover_genome
from evolution.skeleton_registry import SkeletonRegistry
from evolution.random_program import random_ir_program

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """Generic evolution engine operating on IRGenome populations."""

    def __init__(
        self,
        config: EvolutionConfig,
        evaluator: FitnessEvaluator,
        registry: SkeletonRegistry,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.registry = registry
        self.rng = rng or np.random.default_rng(config.seed)

        # State
        self.population: list[IRGenome] = []
        self.fitness: list[FitnessResult] = []
        self.generation = 0
        self.best_ever: IRGenome | None = None
        self.best_fitness: FitnessResult | None = None
        self.hall_of_fame: list[tuple[IRGenome, FitnessResult]] = []
        self.stagnation_count = 0
        self._history: list[dict[str, Any]] = []

    def init_population(self, seed_genomes: list[IRGenome] | None = None) -> None:
        """Create initial random population, optionally seeded with known-good genomes."""
        specs = {role: self.registry.get_program_spec(role)
                 for role in self.config.program_roles}

        # Add seed genomes first
        if seed_genomes:
            for g in seed_genomes:
                self.population.append(g.clone())

        # Fill remaining slots with random genomes
        while len(self.population) < self.config.population_size:
            programs = {}
            for role, spec in specs.items():
                if spec is not None:
                    programs[role] = random_ir_program(
                        spec, self.rng,
                        max_depth=3,
                    )
            genome = IRGenome(
                programs=programs,
                constants=self.rng.uniform(
                    self.config.constant_range[0],
                    self.config.constant_range[1],
                    size=self.config.n_constants,
                ),
            )
            self.population.append(genome)

        # Evaluate initial population
        self.fitness = self.evaluator.evaluate_batch(self.population)
        self._update_best()
        logger.info(
            "Population initialized: %d individuals, best=%.6f",
            len(self.population),
            self.best_fitness.composite_score() if self.best_fitness else float("inf"),
        )

    def run(self, n_generations: int | None = None, callback=None,
            seed_genomes: list[IRGenome] | None = None) -> IRGenome:
        """Run the evolutionary loop.

        Args:
            n_generations: Override for config.n_generations.
            callback: Optional callback(gen, best_fitness, population)
                called after each generation.
            seed_genomes: Optional list of known-good genomes to seed
                the initial population.

        Returns the best genome found.
        """
        if not self.population:
            self.init_population(seed_genomes=seed_genomes)

        n_gens = n_generations or self.config.n_generations

        for gen in range(n_gens):
            self.generation += 1
            t0 = time.time()

            offspring = self._breed_next_generation()

            # Evaluate offspring
            off_fitness = self.evaluator.evaluate_batch(offspring)

            # Combine parents + offspring, select survivors
            combined = list(zip(self.population, self.fitness)) + list(zip(offspring, off_fitness))

            # Sort by fitness (lower composite_score is better)
            combined.sort(key=lambda x: x[1].composite_score())

            # Niching: remove duplicates within niche_radius
            if self.config.niche_radius > 0:
                combined = self._apply_niching(combined)

            # Select top population_size
            combined = combined[:self.config.population_size]
            self.population = [g for g, f in combined]
            self.fitness = [f for g, f in combined]

            prev_best = self.best_fitness.composite_score() if self.best_fitness else float("inf")
            self._update_best()
            cur_best = self.best_fitness.composite_score() if self.best_fitness else float("inf")

            # Track stagnation
            if cur_best >= prev_best - 1e-8:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0

            # Hard restart if stagnation is too long
            if self.config.hard_restart_after > 0 and self.stagnation_count >= self.config.hard_restart_after:
                logger.info("Hard restart at generation %d", self.generation)
                self._hard_restart()

            elapsed = time.time() - t0
            self._history.append({
                "generation": self.generation,
                "best_score": cur_best,
                "avg_score": np.mean([f.composite_score() for f in self.fitness]),
                "elapsed": elapsed,
                "stagnation": self.stagnation_count,
            })

            if gen % 10 == 0 or gen == n_gens - 1:
                logger.info(
                    "Gen %d: best=%.6f avg=%.6f stag=%d (%.2fs)",
                    self.generation,
                    cur_best,
                    self._history[-1]["avg_score"],
                    self.stagnation_count,
                    elapsed,
                )

            if callback is not None:
                callback(self.generation, self.best_fitness, self.population)

        return self.best_ever

    def _breed_next_generation(self) -> list[IRGenome]:
        """Create offspring via tournament selection + crossover + mutation."""
        offspring: list[IRGenome] = []
        n_offspring = self.config.population_size - self.config.elite_count

        for _ in range(n_offspring):
            parent1 = self._tournament_select()
            child: IRGenome

            if self.rng.random() < self.config.crossover_rate:
                parent2 = self._tournament_select()
                child = crossover_genome(
                    parent1, parent2, self.config, self.rng
                )
            else:
                child = parent1.clone()

            if self.rng.random() < self.config.mutation_rate:
                mutate_genome(child, self.config, self.rng)

            child.generation = self.generation
            offspring.append(child)

        return offspring

    def _tournament_select(self) -> IRGenome:
        """Tournament selection."""
        indices = self.rng.choice(
            len(self.population),
            size=min(self.config.tournament_size, len(self.population)),
            replace=False,
        )
        best_idx = min(indices, key=lambda i: self.fitness[i].composite_score())
        return self.population[best_idx]

    def _apply_niching(
        self,
        combined: list[tuple[IRGenome, FitnessResult]],
    ) -> list[tuple[IRGenome, FitnessResult]]:
        """Remove individuals too close to better ones (by structural hash)."""
        kept: list[tuple[IRGenome, FitnessResult]] = []
        seen_hashes: dict[str, int] = {}

        for genome, fit in combined:
            h = genome.structural_hash()
            count = seen_hashes.get(h, 0)
            if count < self.config.niche_radius:
                kept.append((genome, fit))
                seen_hashes[h] = count + 1

        return kept

    def _update_best(self) -> None:
        """Update best_ever and hall of fame."""
        if not self.fitness:
            return

        best_idx = min(
            range(len(self.fitness)),
            key=lambda i: self.fitness[i].composite_score()
        )
        best_genome = self.population[best_idx]
        best_fit = self.fitness[best_idx]

        if self.best_fitness is None or best_fit < self.best_fitness:
            self.best_ever = best_genome.clone()
            self.best_fitness = best_fit

        # Update hall of fame
        self.hall_of_fame.append((best_genome.clone(), best_fit))
        # Keep only top entries
        self.hall_of_fame.sort(key=lambda x: x[1].composite_score())
        self.hall_of_fame = self.hall_of_fame[:self.config.hall_of_fame_size]

    def _hard_restart(self) -> None:
        """Replace most of population with random individuals, keep elites."""
        # Keep hall-of-fame individuals
        elite_genomes = [g.clone() for g, f in self.hall_of_fame]

        # Re-init rest
        specs = {role: self.registry.get_program_spec(role)
                 for role in self.config.program_roles}

        new_pop: list[IRGenome] = list(elite_genomes)
        while len(new_pop) < self.config.population_size:
            programs = {}
            for role, spec in specs.items():
                if spec is not None:
                    programs[role] = random_ir_program(
                        spec, self.rng, max_depth=3,
                    )
            genome = IRGenome(
                programs=programs,
                constants=self.rng.uniform(
                    self.config.constant_range[0],
                    self.config.constant_range[1],
                    size=self.config.n_constants,
                ),
            )
            new_pop.append(genome)

        self.population = new_pop
        self.fitness = self.evaluator.evaluate_batch(self.population)
        self.stagnation_count = 0
        self._update_best()

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._history
