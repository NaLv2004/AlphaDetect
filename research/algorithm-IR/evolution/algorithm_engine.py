"""Two-level algorithm evolution engine.

Macro level: evolves the *population* of AlgorithmGenomes (different
detector skeletons and their slot assignments).

Micro level: within each genome, evolves the slot populations
(IR programs filling each AlgSlot) using fast, lightweight mutations.

Grafting layer: cross-pollinates slot implementations between genomes
and across hierarchy levels.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any, Callable

import numpy as np

from evolution.fitness import FitnessResult
from evolution.pool_types import (
    AlgorithmGenome,
    AlgorithmEvolutionConfig,
    AlgorithmFitnessEvaluator,
    GraftRecord,
    PatternMatcherFn,
    SlotPopulation,
)
from evolution.ir_pool import build_ir_pool, find_algslot_ops
from evolution.operators import mutate_ir, crossover_ir
from evolution.random_program import random_ir_program
from evolution.materialize import materialize_with_override

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Two-Level Algorithm Evolution Engine
# ═══════════════════════════════════════════════════════════════════════════

class AlgorithmEvolutionEngine:
    """Hierarchical evolution: macro (skeleton selection) + micro (slot IR)."""

    def __init__(
        self,
        evaluator: AlgorithmFitnessEvaluator,
        config: AlgorithmEvolutionConfig | None = None,
        rng: np.random.Generator | None = None,
        pattern_matcher: PatternMatcherFn | None = None,
        test_inputs_fn: Callable[[], list] | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or AlgorithmEvolutionConfig()
        self.rng = rng or np.random.default_rng(self.config.seed)
        self.pattern_matcher = pattern_matcher
        self.test_inputs_fn = test_inputs_fn

        # State
        self.population: list[AlgorithmGenome] = []
        self.fitness: list[FitnessResult] = []
        self.generation = 0
        self.best_genome: AlgorithmGenome | None = None
        self.best_fitness: FitnessResult | None = None
        self.hall_of_fame: list[tuple[AlgorithmGenome, FitnessResult]] = []
        self.stagnation_count = 0
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_population(
        self,
        seed_genomes: list[AlgorithmGenome] | None = None,
    ) -> None:
        """Build initial population from IR pool + optional seeds."""
        pool = build_ir_pool(self.rng)
        self.population = []

        if seed_genomes:
            for g in seed_genomes:
                self.population.append(g.clone())

        for genome in pool:
            if len(self.population) < self.config.pool_size:
                self.population.append(genome)

        # If pool is smaller than pool_size, duplicate with variations
        while len(self.population) < self.config.pool_size:
            src = self.population[self.rng.integers(0, len(self.population))]
            clone = src.clone()
            clone.generation = 0
            # Mutate one slot in the clone
            self._mutate_random_slot(clone)
            self.population.append(clone)

        # Initial evaluation
        self.fitness = self.evaluator.evaluate_batch(self.population)
        self._update_best()
        logger.info(
            "Population initialized: %d genomes, best=%.6f",
            len(self.population),
            self.best_fitness.composite_score() if self.best_fitness else float("inf"),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        n_generations: int | None = None,
        callback: Callable | None = None,
    ) -> AlgorithmGenome:
        """Run the two-level evolution loop.

        Parameters
        ----------
        n_generations : int, optional
            Override config.n_generations.
        callback : callable, optional
            Called after each generation: callback(gen, best_fitness, population).

        Returns
        -------
        AlgorithmGenome
            The best genome found.
        """
        if not self.population:
            self.init_population()

        n_gens = n_generations or self.config.n_generations
        cfg = self.config

        for gen in range(n_gens):
            self.generation += 1
            t0 = time.perf_counter()

            # ---- Micro level: evolve slots within each genome ----
            for i, genome in enumerate(self.population):
                self._micro_evolve(genome)

            # ---- Re-evaluate after micro evolution ----
            self.fitness = self.evaluator.evaluate_batch(self.population)
            self._update_best()

            # ---- Macro level: breed next generation ----
            offspring = self._breed_macro()

            # ---- Track A: structural grafting via PatternMatcher ----
            graft_offspring: list[AlgorithmGenome] = []
            if self.pattern_matcher is not None:
                entries = [
                    g.to_entry(f)
                    for g, f in zip(self.population, self.fitness)
                ]
                proposals = self.pattern_matcher(entries, self.generation)
                for proposal in proposals:
                    try:
                        child = self._execute_graft(proposal)
                        if child is not None:
                            child.generation = self.generation
                            graft_offspring.append(child)
                    except Exception as exc:
                        logger.debug("Graft failed: %s", exc)

            # Evaluate offspring
            all_offspring = offspring + graft_offspring
            off_fitness = self.evaluator.evaluate_batch(all_offspring)

            # Selection: keep best from parents + offspring
            combined = list(zip(self.population, self.fitness)) + \
                       list(zip(all_offspring, off_fitness))
            combined.sort(key=lambda x: x[1].composite_score())

            # Elitism + truncation selection
            survivors = combined[:cfg.pool_size]
            self.population = [g for g, _ in survivors]
            self.fitness = [f for _, f in survivors]

            self._update_best()
            elapsed = time.perf_counter() - t0

            # ---- Trace collection for PatternMatcher ----
            if self.pattern_matcher is not None and self.generation % 3 == 0:
                self._collect_traces(top_k=min(3, len(self.population)))

            # ---- Grafting: cross-pollinate slots ----
            if self.generation % 5 == 0:
                self._graft_pass()

            # ---- Stagnation detection ----
            self._check_stagnation()

            # ---- Record history ----
            record = {
                "generation": self.generation,
                "best_score": self.best_fitness.composite_score() if self.best_fitness else float("inf"),
                "best_algo": self.best_genome.algo_id if self.best_genome else None,
                "elapsed": elapsed,
                "pop_algos": [g.algo_id for g in self.population],
            }
            self._history.append(record)
            logger.info(
                "Gen %d: best=%.6f (%s) elapsed=%.1fs stagnation=%d",
                self.generation,
                record["best_score"],
                record["best_algo"],
                elapsed,
                self.stagnation_count,
            )

            if callback:
                callback(self.generation, self.best_fitness, self.population)

        return self.best_genome

    # ------------------------------------------------------------------
    # Micro-level evolution (slot populations)
    # ------------------------------------------------------------------

    def _micro_evolve(self, genome: AlgorithmGenome) -> None:
        """Evolve slot populations within one genome for a few micro-generations."""
        cfg = self.config
        for _ in range(cfg.micro_generations):
            for slot_id, pop in genome.slot_populations.items():
                if len(pop.variants) < 2:
                    continue
                self._micro_step(genome, slot_id, pop)

    def _micro_step(
        self, genome: AlgorithmGenome, slot_id: str, pop: SlotPopulation,
    ) -> None:
        """One micro-generation step on a single slot population.

        Generates candidate variants via mutation/crossover, evaluates each
        using ``materialize_with_override`` (fixing all other slots at their
        best variant), updates fitness, and trims the sub-population.
        """
        cfg = self.config
        rng = self.rng

        n = len(pop.variants)
        if n == 0:
            return

        new_variants: list = []

        # Mutation: pick a random variant, mutate it
        if rng.random() < cfg.micro_mutation_rate and n < cfg.micro_pop_size * 2:
            idx = rng.integers(0, n)
            parent = pop.variants[idx]
            try:
                child = mutate_ir(parent, rng)
                new_variants.append(child)
            except Exception:
                pass

        # Crossover: pick two, cross
        if n >= 2 and rng.random() < cfg.micro_mutation_rate * 0.5:
            i1, i2 = rng.choice(n, 2, replace=False)
            try:
                child = crossover_ir(pop.variants[i1], pop.variants[i2], rng)
                new_variants.append(child)
            except Exception:
                pass

        # Evaluate each new variant
        for variant_ir in new_variants:
            fitness_val = self._evaluate_slot_variant(genome, slot_id, variant_ir)
            pop.variants.append(variant_ir)
            pop.fitness.append(fitness_val)

        # Update best_idx based on fitness
        if pop.fitness:
            best_i = 0
            best_f = pop.fitness[0]
            for i, f in enumerate(pop.fitness):
                if f < best_f:
                    best_f = f
                    best_i = i
            pop.best_idx = best_i

        # Trim to keep micro pop manageable
        max_size = cfg.micro_pop_size * 3
        if len(pop.variants) > max_size:
            # Sort by fitness, keep best
            indexed = list(range(len(pop.variants)))
            indexed.sort(key=lambda i: pop.fitness[i])
            keep = indexed[:max_size]
            keep_set = set(keep)
            new_variants_list = [pop.variants[i] for i in sorted(keep)]
            new_fitness_list = [pop.fitness[i] for i in sorted(keep)]
            # Find new best_idx
            old_best_variant = pop.variants[pop.best_idx] if pop.best_idx < len(pop.variants) else None
            pop.variants = new_variants_list
            pop.fitness = new_fitness_list
            # Re-find best
            pop.best_idx = 0
            for i, f in enumerate(pop.fitness):
                if f < pop.fitness[pop.best_idx]:
                    pop.best_idx = i

    def _evaluate_slot_variant(
        self,
        genome: AlgorithmGenome,
        slot_id: str,
        variant_ir,
    ) -> float:
        """Evaluate a single slot variant by materializing the genome with
        this variant overriding the specified slot, then running a quick
        fitness evaluation.

        Returns scalar fitness (lower is better). Returns inf on failure.
        """
        try:
            fn = materialize_with_override(genome, {slot_id: variant_ir})
        except Exception:
            return float("inf")

        # Use evaluator's quick single-result path with a few samples
        rng = np.random.default_rng(self.config.seed + self.generation)
        from evolution.mimo_evaluator import generate_mimo_sample, qam16_constellation, qpsk_constellation

        # Determine constellation from evaluator
        constellation = getattr(self.evaluator, "constellation", qam16_constellation())
        cfg = getattr(self.evaluator, "config", None)
        Nr = cfg.Nr if cfg else 4
        Nt = cfg.Nt if cfg else 4
        snr_db = 15.0
        if cfg and cfg.snr_db_list:
            snr_db = cfg.snr_db_list[len(cfg.snr_db_list) // 2]

        # Run a few quick trials
        n_quick = 5
        total_err = 0
        total_sym = 0
        for trial in range(n_quick):
            try:
                H, x_true, y, sigma2 = generate_mimo_sample(
                    Nr, Nt, constellation, snr_db, rng,
                )
                x_hat = fn(H, y, sigma2, constellation)
                if x_hat is None or len(x_hat) != Nt:
                    total_err += Nt
                else:
                    from evolution.mimo_evaluator import _nearest_symbols
                    x_hat = _nearest_symbols(x_hat, constellation)
                    total_err += int(np.sum(np.abs(x_true - x_hat) > 1e-6))
                total_sym += Nt
            except Exception:
                total_err += Nt
                total_sym += Nt

        return total_err / max(total_sym, 1)

    # ------------------------------------------------------------------
    # Macro-level breeding
    # ------------------------------------------------------------------

    def _breed_macro(self) -> list[AlgorithmGenome]:
        """Create offspring via tournament selection + slot-level crossover."""
        cfg = self.config
        offspring = []
        n_offspring = cfg.pool_size // 2

        for _ in range(n_offspring):
            p1 = self._tournament_select(3)
            p2 = self._tournament_select(3)
            child = self._crossover_genomes(p1, p2)
            self._mutate_random_slot(child)
            child.generation = self.generation
            offspring.append(child)

        return offspring

    def _tournament_select(self, k: int = 3) -> AlgorithmGenome:
        """Tournament selection: pick k random, return best."""
        n = len(self.population)
        indices = self.rng.choice(n, size=min(k, n), replace=False)
        best_idx = min(indices, key=lambda i: self.fitness[i].composite_score())
        return self.population[best_idx]

    def _crossover_genomes(
        self, p1: AlgorithmGenome, p2: AlgorithmGenome,
    ) -> AlgorithmGenome:
        """Cross two genomes by mixing slot populations.

        If same skeleton: cross slot populations pair-wise.
        If different: take p1's skeleton, graft compatible slots from p2.
        """
        child = p1.clone()
        child.parent_ids = [p1.algo_id, p2.algo_id]

        if p1.algo_id == p2.algo_id:
            # Same skeleton: randomly swap slot populations
            for slot_id in child.slot_populations:
                if slot_id in p2.slot_populations and self.rng.random() < 0.5:
                    child.slot_populations[slot_id] = deepcopy(
                        p2.slot_populations[slot_id]
                    )
        else:
            # Different skeletons: graft compatible slots
            for slot_id, pop in p2.slot_populations.items():
                if slot_id in child.slot_populations:
                    # Compatible slot found — donate best variant
                    if pop.variants and pop.best_idx < len(pop.variants):
                        donor_ir = deepcopy(pop.variants[pop.best_idx])
                        child.slot_populations[slot_id].variants.append(donor_ir)
                        child.slot_populations[slot_id].fitness.append(float("inf"))

        return child

    def _mutate_random_slot(self, genome: AlgorithmGenome) -> None:
        """Mutate a random slot's best variant."""
        if not genome.slot_populations:
            return
        slot_ids = list(genome.slot_populations.keys())
        slot_id = slot_ids[self.rng.integers(0, len(slot_ids))]
        pop = genome.slot_populations[slot_id]
        if not pop.variants:
            return
        idx = pop.best_idx if pop.best_idx < len(pop.variants) else 0
        try:
            mutated = mutate_ir(pop.variants[idx], self.rng)
            pop.variants.append(mutated)
            pop.fitness.append(float("inf"))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Grafting
    # ------------------------------------------------------------------

    def _graft_pass(self) -> None:
        """Cross-pollinate slot implementations across genomes."""
        if len(self.population) < 2:
            return

        # Find donor = best genome, recipients = bottom half
        ranked = sorted(
            range(len(self.population)),
            key=lambda i: self.fitness[i].composite_score(),
        )
        donor_idx = ranked[0]
        donor = self.population[donor_idx]

        n_recipients = max(1, len(ranked) // 4)
        for r_idx in ranked[-n_recipients:]:
            recipient = self.population[r_idx]
            # Find compatible slots
            for slot_id, d_pop in donor.slot_populations.items():
                if slot_id not in recipient.slot_populations:
                    continue
                if not d_pop.variants or d_pop.best_idx >= len(d_pop.variants):
                    continue
                # Donate best variant
                donor_ir = deepcopy(d_pop.variants[d_pop.best_idx])
                r_pop = recipient.slot_populations[slot_id]
                r_pop.variants.append(donor_ir)
                r_pop.fitness.append(float("inf"))

    def _execute_graft(self, proposal) -> AlgorithmGenome | None:
        """Execute a GraftProposal via graft_general() and produce a new genome.

        Steps:
          1. Call graft_general() for IR-level op surgery
          2. Discover new slots introduced by the donor
          3. Initialize SlotPopulations for new slots
          4. Build and return a new AlgorithmGenome
        """
        from algorithm_ir.grafting.graft_general import graft_general
        from evolution.ir_pool import find_algslot_ops

        # Find the host genome
        host_genome: AlgorithmGenome | None = None
        for g in self.population:
            if g.algo_id == proposal.host_algo_id:
                host_genome = g
                break
        if host_genome is None:
            logger.warning("Host genome %s not found", proposal.host_algo_id)
            return None

        # Execute the graft
        artifact = graft_general(host_genome.structural_ir, proposal)

        # Build child genome
        child = AlgorithmGenome(
            algo_id=AlgorithmGenome._make_id(),
            structural_ir=artifact.ir,
            slot_populations=deepcopy(host_genome.slot_populations),
            constants=host_genome.constants.copy(),
            generation=self.generation,
            parent_ids=[host_genome.algo_id, proposal.donor_algo_id or ""],
            graft_history=list(host_genome.graft_history) + [
                GraftRecord(
                    generation=self.generation,
                    host_algo_id=host_genome.algo_id,
                    donor_algo_id=proposal.donor_algo_id,
                    proposal_id=proposal.proposal_id,
                    region_summary=str(proposal.region.region_id),
                    new_slots_created=artifact.new_slot_ids,
                ),
            ],
            tags=set(host_genome.tags) | {"grafted"},
        )

        # Initialize SlotPopulations for new slots introduced by donor
        for slot_id in artifact.new_slot_ids:
            if slot_id not in child.slot_populations:
                from evolution.skeleton_registry import ProgramSpec
                spec = ProgramSpec(
                    name=slot_id,
                    param_names=[],
                    param_types=[],
                    return_type="object",
                )
                # Attempt to extract the donor's slot implementation
                donor_variants = []
                if proposal.donor_ir:
                    for op in proposal.donor_ir.ops.values():
                        if op.attrs.get("slot_id") == slot_id:
                            # The donor itself references this slot
                            break
                child.slot_populations[slot_id] = SlotPopulation(
                    slot_id=slot_id,
                    spec=spec,
                    variants=donor_variants,
                    fitness=[float("inf")] * len(donor_variants),
                    best_idx=0,
                )

        return child

    # ------------------------------------------------------------------
    # Runtime trace collection
    # ------------------------------------------------------------------

    def _collect_traces(self, top_k: int = 3) -> None:
        """Collect runtime traces for top-K individuals via interpreter path.

        Traces are stored in genome.metadata and later exposed via to_entry().
        Uses a small number of sample inputs (not full Monte Carlo).
        """
        from algorithm_ir.runtime.interpreter import execute_ir
        from algorithm_ir.factgraph.builder import build_factgraph
        from evolution.materialize import materialize

        ranked_indices = sorted(
            range(len(self.population)),
            key=lambda i: self.fitness[i].composite_score(),
        )
        top_indices = ranked_indices[:top_k]

        for idx in top_indices:
            genome = self.population[idx]
            try:
                # Materialize the genome to get a FunctionIR with slots filled
                mat_ir = materialize(genome)
                if mat_ir is None:
                    continue

                # Generate a single sample input for trace collection
                cfg = getattr(self.evaluator, 'config', None)
                if cfg and hasattr(cfg, 'Nr'):
                    nr, nt = cfg.Nr, cfg.Nt
                else:
                    nr, nt = 4, 4

                from evolution.mimo_evaluator import generate_mimo_sample, qam16_constellation
                constellation = qam16_constellation()
                H, x_true, y, sigma2 = generate_mimo_sample(
                    nr, nt, constellation, 15.0, self.rng,
                )

                # Execute via interpreter path
                result, trace, runtime_values = execute_ir(
                    mat_ir, [H, y, sigma2, constellation],
                )

                # Build FactGraph
                factgraph = build_factgraph(mat_ir, trace, runtime_values)

                # Cache in genome metadata
                genome.metadata["_cached_trace"] = trace
                genome.metadata["_cached_runtime_values"] = runtime_values
                genome.metadata["_cached_factgraph"] = factgraph

            except Exception as exc:
                logger.debug("Trace collection failed for %s: %s", genome.algo_id, exc)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _update_best(self) -> None:
        """Update best_genome and hall_of_fame."""
        if not self.fitness:
            return

        best_idx = min(range(len(self.fitness)),
                       key=lambda i: self.fitness[i].composite_score())
        cur_best = self.fitness[best_idx]

        if self.best_fitness is None or cur_best < self.best_fitness:
            self.best_genome = self.population[best_idx].clone()
            self.best_fitness = cur_best

            # Update hall of fame
            self.hall_of_fame.append((self.best_genome.clone(), cur_best))
            # Keep top 10
            self.hall_of_fame.sort(key=lambda x: x[1].composite_score())
            self.hall_of_fame = self.hall_of_fame[:10]

    def _check_stagnation(self) -> None:
        """Detect and handle stagnation."""
        if len(self._history) < 2:
            self.stagnation_count = 0
            return

        prev_score = self._history[-2]["best_score"] if len(self._history) >= 2 else float("inf")
        cur_score = self._history[-1]["best_score"]

        if abs(prev_score - cur_score) < 1e-8:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0

        # Hard restart bottom quartile if stagnant for too long
        if self.stagnation_count >= 10:
            logger.info("Stagnation detected (%d gens), restarting bottom quartile",
                        self.stagnation_count)
            self._hard_restart_bottom()
            self.stagnation_count = 0

    def _hard_restart_bottom(self) -> None:
        """Replace bottom 25% of population with fresh random genomes."""
        ranked = sorted(
            range(len(self.population)),
            key=lambda i: self.fitness[i].composite_score(),
        )
        n_restart = max(1, len(ranked) // 4)
        fresh = build_ir_pool(self.rng)

        for i, r_idx in enumerate(ranked[-n_restart:]):
            if i < len(fresh):
                self.population[r_idx] = fresh[i]
                # Mutate for diversity
                self._mutate_random_slot(self.population[r_idx])
                self._mutate_random_slot(self.population[r_idx])
            else:
                # Clone and mutate existing
                src = self.population[ranked[0]]
                clone = src.clone()
                self._mutate_random_slot(clone)
                self._mutate_random_slot(clone)
                self.population[r_idx] = clone

    @property
    def history(self) -> list[dict[str, Any]]:
        """Generation-by-generation evolution history."""
        return list(self._history)
