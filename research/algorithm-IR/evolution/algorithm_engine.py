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


# ══════════════════════════════════════════════════════════════════════════�?
# Two-Level Algorithm Evolution Engine
# ══════════════════════════════════════════════════════════════════════════�?

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
        # Gene bank: original pool genomes kept permanently for donor lookup
        self.gene_bank: list[AlgorithmGenome] = []

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
        # Populate gene bank with original pool genomes (permanent donor source)
        self.gene_bank = list(pool)

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
            graft_proposal_ids: list[str] = []  # track proposal IDs for RL feedback
            if self.pattern_matcher is not None:
                entries = [
                    g.to_entry(f)
                    for g, f in zip(self.population, self.fitness)
                ]
                # Always include gene bank entries so expert rules can find
                # original algorithm donors even if they've been eliminated
                # from the current population.
                pop_ids = {g.algo_id for g in self.population}
                bank_entries = [
                    g.to_entry(None)
                    for g in self.gene_bank
                    if g.algo_id not in pop_ids
                ]
                all_entries = entries + bank_entries
                proposals = self.pattern_matcher(all_entries, self.generation)
                n_graft_ok = 0
                for proposal in proposals:
                    try:
                        child = self._execute_graft(proposal)
                        if child is not None:
                            child.generation = self.generation
                            graft_offspring.append(child)
                            graft_proposal_ids.append(proposal.proposal_id)
                            n_graft_ok += 1
                    except Exception as exc:
                        logger.warning("Graft failed: %s", exc)
                if proposals:
                    logger.info(
                        "  Structural grafts: %d/%d succeeded",
                        n_graft_ok, len(proposals),
                    )

            # Evaluate offspring
            all_offspring = offspring + graft_offspring
            off_fitness = self.evaluator.evaluate_batch(all_offspring)

            # ---- RL feedback for GNN pattern matcher ----
            if graft_offspring and graft_proposal_ids:
                # Normalized reward:
                #   +1  if graft_score = 0    (perfect detector)
                #    0  if graft_score = 0.5  (half random guessing)
                #   -1  if graft_score = 1.0  (random guessing)
                # This gives the GNN a strong gradient for any *working* graft
                # (SER < 0.5) vs broken grafts (SER ~1.0), independent of
                # whether the graft beats the already-excellent best parent.
                graft_start = len(offspring)
                for gi, pid in enumerate(graft_proposal_ids):
                    graft_idx = graft_start + gi
                    if graft_idx < len(off_fitness):
                        graft_score = off_fitness[graft_idx].composite_score()
                        reward = 1.0 - 2.0 * min(graft_score, 1.0)
                        self._record_graft_reward(pid, reward)

            # Selection with niching: preserve structural diversity
            combined = list(zip(self.population, self.fitness)) + \
                       list(zip(all_offspring, off_fitness))
            survivors = self._select_with_niching(combined, cfg.pool_size)
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
                    # Bootstrap: generate a mutation so micro-evolution can start
                    if len(pop.variants) == 1:
                        try:
                            child = mutate_ir(pop.variants[0], self.rng)
                            pop.variants.append(child)
                            pop.fitness.append(float("inf"))
                        except Exception:
                            continue
                    else:
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

        # Run a few quick trials with per-trial timeout
        import concurrent.futures as _cf
        n_quick = 5
        total_err = 0
        total_sym = 0
        _timeout = getattr(cfg, "timeout_sec", 5.0) / max(n_quick, 1) * 2 if cfg else 2.0

        _ex = _cf.ThreadPoolExecutor(max_workers=1)
        _ex_dirty = False

        for trial in range(n_quick):
            try:
                H, x_true, y, sigma2 = generate_mimo_sample(
                    Nr, Nt, constellation, snr_db, rng,
                )
                if _ex_dirty:
                    _ex.shutdown(wait=False)
                    _ex = _cf.ThreadPoolExecutor(max_workers=1)
                    _ex_dirty = False
                try:
                    fut = _ex.submit(fn, H, y, sigma2, constellation)
                    x_hat = fut.result(timeout=_timeout)
                    if x_hat is None or len(x_hat) != Nt:
                        total_err += Nt
                    else:
                        from evolution.mimo_evaluator import _nearest_symbols
                        x_hat = _nearest_symbols(x_hat, constellation)
                        total_err += int(np.sum(np.abs(x_true - x_hat) > 1e-6))
                except _cf.TimeoutError:
                    fut.cancel()
                    total_err += Nt
                    _ex_dirty = True
                except Exception:
                    total_err += Nt
                total_sym += Nt
            except Exception:
                total_err += Nt
                total_sym += Nt

        _ex.shutdown(wait=False)

        return total_err / max(total_sym, 1)

    # ------------------------------------------------------------------
    # Macro-level breeding
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Niching selection
    # ------------------------------------------------------------------

    def _select_with_niching(
        self,
        candidates: list[tuple[AlgorithmGenome, FitnessResult]],
        target_size: int,
    ) -> list[tuple[AlgorithmGenome, FitnessResult]]:
        """Select survivors with structural diversity protection.

        Strategy: fill elite slots first (top 50%), then fill remaining
        slots by picking the best genome from each under-represented
        algo_id niche. This prevents the population collapsing to one
        algorithm family.
        """
        candidates.sort(key=lambda x: x[1].composite_score())

        # Phase 1: elite — top 50% by pure fitness
        n_elite = max(1, target_size // 2)
        survivors = list(candidates[:n_elite])
        selected_ids = set()
        for g, _ in survivors:
            selected_ids.add(id(g))

        # Phase 1.5: protect best grafted individuals (min 2 reserved slots)
        grafted_in_survivors = sum(1 for g, _ in survivors if "grafted" in g.tags)
        if grafted_in_survivors < 2:
            grafted_cands = [
                (g, f) for g, f in candidates
                if "grafted" in g.tags and id(g) not in selected_ids
            ]
            grafted_cands.sort(key=lambda x: x[1].composite_score())
            for g, f in grafted_cands[:2 - grafted_in_survivors]:
                survivors.append((g, f))
                selected_ids.add(id(g))

        # Phase 2: diversity �?fill remaining slots with best unseen algo_ids
        remaining = [c for c in candidates if id(c[0]) not in selected_ids]
        # Group by algo_id
        niche_best: dict[str, tuple[AlgorithmGenome, FitnessResult]] = {}
        for g, f in remaining:
            aid = g.algo_id
            if aid not in niche_best or f.composite_score() < niche_best[aid][1].composite_score():
                niche_best[aid] = (g, f)

        # Already-covered algo_ids from elite
        elite_aids = {g.algo_id for g, _ in survivors}
        # Prioritise niches not yet in survivors
        uncovered = [(g, f) for aid, (g, f) in niche_best.items() if aid not in elite_aids]
        uncovered.sort(key=lambda x: x[1].composite_score())
        for g, f in uncovered:
            if len(survivors) >= target_size:
                break
            survivors.append((g, f))
            selected_ids.add(id(g))

        # Phase 3: if still under target, fill from remaining by fitness
        if len(survivors) < target_size:
            leftovers = [c for c in candidates if id(c[0]) not in selected_ids]
            leftovers.sort(key=lambda x: x[1].composite_score())
            for c in leftovers:
                if len(survivors) >= target_size:
                    break
                survivors.append(c)

        return survivors[:target_size]

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
                    # Compatible slot found �?donate best variant
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
          0. Trim donor IR to avoid inlining the entire donor
          1. Call graft_general() for IR-level op surgery
          2. Discover new slots introduced by the donor
          3. Initialize SlotPopulations for new slots
          4. Build and return a new AlgorithmGenome
        """
        from algorithm_ir.grafting.graft_general import graft_general
        from evolution.ir_pool import find_algslot_ops

        # --- Step 0: Trim donor IR ---
        # GNN proposals already carry a GNN-selected trimmed donor (skip).
        # For other matchers (expert rules, static), apply heuristic trimming.
        if not proposal.proposal_id.startswith("gnn_graft"):
            proposal = self._trim_donor_for_proposal(proposal)

        # Find the host genome (current population first, then gene bank)
        host_genome: AlgorithmGenome | None = None
        for g in self.population:
            if g.algo_id == proposal.host_algo_id:
                host_genome = g
                break
        if host_genome is None:
            for g in self.gene_bank:
                if g.algo_id == proposal.host_algo_id:
                    host_genome = g
                    break
        if host_genome is None:
            logger.warning("Host genome %s not found", proposal.host_algo_id)
            return None

        # Find donor genome (current population first, then gene bank)
        donor_genome: AlgorithmGenome | None = None
        for g in self.population:
            if g.algo_id == proposal.donor_algo_id:
                donor_genome = g
                break
        if donor_genome is None:
            for g in self.gene_bank:
                if g.algo_id == proposal.donor_algo_id:
                    donor_genome = g
                    break

        # Execute the graft (inline �?donor ops are cloned into host IR)
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
            metadata={},
        )

        # Initialize SlotPopulations for new slots introduced by donor.
        # When a donor genome is available, copy its slot populations so
        # that inlined slot ops have real implementations (not empty
        # fallbacks).  Without this, grafted children produce dead code
        # because materialize() falls back to a pass-through stub.
        for slot_id in artifact.new_slot_ids:
            if slot_id in child.slot_populations:
                continue
            # Try to inherit from donor genome's slot populations
            donor_pop = self._find_donor_slot_population(
                donor_genome, slot_id,
            )
            if donor_pop is not None:
                child.slot_populations[slot_id] = SlotPopulation(
                    slot_id=slot_id,
                    spec=donor_pop.spec,
                    variants=[deepcopy(v) for v in donor_pop.variants],
                    fitness=list(donor_pop.fitness),
                    best_idx=donor_pop.best_idx,
                    source_variants=list(donor_pop.source_variants)
                    if donor_pop.source_variants
                    else [],
                )
            else:
                # Fallback: create an empty population (will use stub)
                from evolution.skeleton_registry import ProgramSpec
                spec = ProgramSpec(
                    name=slot_id,
                    param_names=[],
                    param_types=[],
                    return_type="object",
                )
                child.slot_populations[slot_id] = SlotPopulation(
                    slot_id=slot_id,
                    spec=spec,
                    variants=[],
                    fitness=[],
                    best_idx=0,
                )

        return child

    def _trim_donor_for_proposal(self, proposal) -> object:
        """Trim the donor IR to a region matching the host region size.

        Without trimming, graft_general inlines the ENTIRE donor IR
        (typically 10-20 ops) to replace a small 1-3 op host region,
        creating bloated hybrids that don't function. This method creates
        a focused donor IR containing only a similar number of ops.
        """
        from copy import deepcopy
        from algorithm_ir.ir.model import FunctionIR, Op, Value, Block

        donor_ir = proposal.donor_ir
        if donor_ir is None:
            return proposal

        host_region_size = len(proposal.region.op_ids)

        # Count non-terminator ops in donor
        terminators = {"return", "branch", "jump"}
        donor_non_term = [
            oid for oid in self._ordered_donor_ops(donor_ir)
            if donor_ir.ops.get(oid) and donor_ir.ops[oid].opcode not in terminators
        ]

        # If donor is already small enough, no trimming needed
        max_allowed = max(host_region_size * 3, 5)  # at most 3x host region
        if len(donor_non_term) <= max_allowed:
            return proposal

        # Select the most interesting region from the donor:
        # Prefer regions with "call" ops (slot calls are the functional units)
        best_start, best_score = 0, -1
        for start in range(len(donor_non_term) - max_allowed + 1):
            region = donor_non_term[start:start + max_allowed]
            score = sum(
                2 if donor_ir.ops[oid].opcode == "call" else
                1 if donor_ir.ops[oid].opcode in ("binary", "unary") else 0
                for oid in region
            )
            if score > best_score:
                best_score = score
                best_start = start

        selected_ops = set(donor_non_term[best_start:best_start + max_allowed])

        # Build a trimmed donor IR containing only the selected ops
        trimmed = self._build_trimmed_donor(donor_ir, selected_ops)
        if trimmed is None:
            return proposal

        # Create a modified proposal with the trimmed donor
        new_proposal = deepcopy(proposal)
        new_proposal.donor_ir = trimmed
        return new_proposal

    def _ordered_donor_ops(self, ir) -> list[str]:
        """Return donor ops in block order (entry block first)."""
        result = []
        entry = ir.blocks.get(ir.entry_block)
        if entry:
            result.extend(entry.op_ids)
        for bid, block in ir.blocks.items():
            if bid != ir.entry_block:
                result.extend(block.op_ids)
        return result

    def _build_trimmed_donor(self, donor_ir, selected_op_ids: set[str]):
        """Build a minimal FunctionIR from selected donor ops."""
        from algorithm_ir.ir.model import FunctionIR, Op, Value, Block

        # Collect values defined and used by selected ops
        defined_vals: set[str] = set()
        used_vals: set[str] = set()
        for oid in selected_op_ids:
            op = donor_ir.ops.get(oid)
            if op is None:
                continue
            defined_vals.update(op.outputs)
            used_vals.update(op.inputs)

        # Entry values = used but not defined in region �?become function args
        entry_vals = sorted(used_vals - defined_vals)
        # Exit values = defined in region but used outside (or all defined)
        exit_vals = sorted(defined_vals)

        # Map entry values to donor's arg values where possible
        arg_map = {}
        donor_args = set(donor_ir.arg_values)
        for v in entry_vals:
            if v in donor_args:
                arg_map[v] = v

        # Build single-block IR
        block_id = "entry"
        ops = {}
        for oid in self._ordered_donor_ops(donor_ir):
            if oid in selected_op_ids:
                op = donor_ir.ops[oid]
                ops[oid] = Op(
                    id=op.id,
                    opcode=op.opcode,
                    inputs=list(op.inputs),
                    outputs=list(op.outputs),
                    block_id=block_id,
                    source_span=op.source_span,
                    attrs=dict(op.attrs),
                )

        # Add a return op that outputs the last defined value
        ret_vals = exit_vals[-1:] if exit_vals else []
        ret_id = "ret_trim"
        ops[ret_id] = Op(
            id=ret_id, opcode="return",
            inputs=ret_vals, outputs=[],
            block_id=block_id, attrs={},
        )

        # Collect all referenced values
        all_vals = {}
        for vid in set(entry_vals) | set(defined_vals):
            v = donor_ir.values.get(vid)
            if v:
                all_vals[vid] = Value(
                    id=v.id, name_hint=v.name_hint,
                    type_hint=v.type_hint, source_span=v.source_span,
                    def_op=v.def_op if v.def_op in ops else "",
                    use_ops=[u for u in v.use_ops if u in ops],
                    attrs=dict(v.attrs),
                )

        block = Block(
            id=block_id,
            op_ids=list(ops.keys()),
            preds=[], succs=[], attrs={},
        )

        # Use donor's arg values that are needed, plus extra entry values
        func_args = [v for v in donor_ir.arg_values if v in entry_vals]
        for v in entry_vals:
            if v not in func_args:
                func_args.append(v)

        return FunctionIR(
            id=donor_ir.id + "_trim",
            name=donor_ir.name + "_trim",
            arg_values=func_args,
            return_values=ret_vals,
            values=all_vals,
            ops=ops,
            blocks={block_id: block},
            entry_block=block_id,
            attrs=dict(donor_ir.attrs) if donor_ir.attrs else {},
        )

    def _find_donor_slot_population(
        self,
        donor_genome: AlgorithmGenome | None,
        slot_id: str,
    ) -> SlotPopulation | None:
        """Find a matching SlotPopulation from the donor genome.

        Tries exact key match, then suffix match (e.g. 'bp_sweep' matches
        population keyed as 'bp.sweep'), then slot_id substring.
        """
        if donor_genome is None or not donor_genome.slot_populations:
            return None

        # 1. Exact key match
        if slot_id in donor_genome.slot_populations:
            pop = donor_genome.slot_populations[slot_id]
            if pop.variants:
                return pop

        # 2. Match by slot_id suffix (e.g. donor key "bp.sweep" for slot_id "bp_sweep")
        for key, pop in donor_genome.slot_populations.items():
            if not pop.variants:
                continue
            # Check if the key's short name matches
            short = key.split(".")[-1]
            if short == slot_id or slot_id.endswith(short) or slot_id.endswith(f"_{short}"):
                return pop
            # Check if slot_id matches the population's slot_id
            if pop.slot_id == slot_id:
                return pop

        return None

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
                # Materialize the genome �?returns Python source (str),
                # not FunctionIR.  The interpreter path requires FunctionIR,
                # so skip trace collection when materialization yields source.
                mat_result = materialize(genome)
                if mat_result is None:
                    continue
                if isinstance(mat_result, str):
                    # Source-level materialization �?cannot use interpreter path
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

    def _record_graft_reward(self, proposal_id: str, reward: float) -> None:
        """Forward graft outcome to GNN pattern matcher if applicable."""
        if self.pattern_matcher is None:
            return
        # Walk through composite matchers to find GNN matchers
        matchers = [self.pattern_matcher]
        if hasattr(self.pattern_matcher, "matchers"):
            matchers = self.pattern_matcher.matchers
        for m in matchers:
            if hasattr(m, "record_outcome"):
                m.record_outcome(proposal_id, reward)

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
