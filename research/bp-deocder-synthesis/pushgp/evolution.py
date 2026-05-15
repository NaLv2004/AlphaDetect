"""Generic Push-GP evolution loop (fitness-agnostic).

Plug-in `fitness_fn(genome) -> float` (smaller = better).  We never
import LDPC code here — that hookup happens in PR6/PR7.

Hard rules enforced here:
  * Every offspring is run through `validate_genome` before entering
    the next population.  Up to `max_retries` attempts per slot; if
    none survive, the parent is cloned instead.
  * Elitism keeps the top `elitism` individuals untouched.
  * Initial population is built from caller-supplied seeds (random
    program generation alone almost never satisfies permutation
    invariance — see PR3 design notes).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .crossover import crossover_genome
from .genome import Genome
from .mutation import mutate_genome
from .random_program import RandomProgramGenerator
from .selection import tournament_select
from .serialize import genome_fingerprint
from .validators import validate_genome


FitnessFn = Callable[[Genome], float]


@dataclass
class EvolutionConfig:
    pop_size: int = 50
    generations: int = 200
    elitism: int = 3
    tournament_k: int = 5
    p_crossover: float = 0.5
    n_mutations: int = 2
    p_const_tweak: float = 0.2
    max_retries: int = 30
    validator_deg: int = 8
    seed: int = 0
    # ---- from-scratch / diversity-first knobs ----
    # Maximum candidate generations per slot when filling a population
    # (initial random fill OR per-generation offspring fill).  No
    # parent.copy() fallback — if exhausted, raise.
    max_attempts_per_slot: int = 500
    # Random program length range for from-scratch initialization.
    rand_min_size: int = 4
    rand_max_size: int = 30
    # Worst-fitness individuals get more aggressive mutation; this is the
    # mutation count applied to the rank-worst individual.  Best
    # individual still gets `n_mutations`.  Linear interpolation by rank.
    n_mutations_max: int = 6
    # Reject duplicates by structural fingerprint.
    dedup: bool = True


@dataclass
class GenLog:
    gen: int
    best_fit: float
    mean_fit: float
    median_fit: float
    n_valid: int
    elapsed_s: float


@dataclass
class EvolutionResult:
    best_genome: Genome
    best_fitness: float
    history: List[GenLog] = field(default_factory=list)
    final_population: List[Genome] = field(default_factory=list)
    final_fitnesses: List[float] = field(default_factory=list)


def _grow_population_from_seeds(
    seeds: Sequence[Genome],
    target: int,
    rng: np.random.Generator,
    rpg: RandomProgramGenerator,
    cfg: EvolutionConfig,
) -> List[Genome]:
    """Pad seed list to `target` via repeated mutation; reject invalids."""
    pop: List[Genome] = [s.copy() for s in seeds]
    if not pop:
        raise ValueError("at least one seed genome is required")
    i = 0
    while len(pop) < target:
        parent = pop[i % len(seeds)]
        for _ in range(cfg.max_retries):
            child = mutate_genome(parent, rng, rpg, n_mutations=cfg.n_mutations,
                                  p_const_tweak=cfg.p_const_tweak)
            ok, _ = validate_genome(child, rng=np.random.default_rng(int(rng.integers(0, 2**31))),
                                    deg=cfg.validator_deg)
            if ok:
                pop.append(child)
                break
        else:
            pop.append(parent.copy())
        i += 1
    return pop


def evolve(
    fitness_fn: FitnessFn,
    seeds: Sequence[Genome],
    cfg: EvolutionConfig,
    on_generation: Optional[Callable[[GenLog, List[Genome], List[float]], None]] = None,
) -> EvolutionResult:
    rng = np.random.default_rng(cfg.seed)
    rpg = RandomProgramGenerator(rng=rng)

    pop = _grow_population_from_seeds(seeds, cfg.pop_size, rng, rpg, cfg)
    fits = [float(fitness_fn(g)) for g in pop]

    history: List[GenLog] = []
    for gen_idx in range(cfg.generations):
        t0 = time.time()
        order = np.argsort(np.where(np.isfinite(fits), fits, np.inf))
        elites = [pop[int(i)].copy() for i in order[: cfg.elitism]]
        elite_fits = [fits[int(i)] for i in order[: cfg.elitism]]

        # Build next population.
        new_pop: List[Genome] = list(elites)
        new_fits: List[float] = list(elite_fits)
        n_valid = 0

        while len(new_pop) < cfg.pop_size:
            parent_a = tournament_select(pop, fits, cfg.tournament_k, rng)
            if rng.random() < cfg.p_crossover:
                parent_b = tournament_select(pop, fits, cfg.tournament_k, rng)
                child0 = crossover_genome(parent_a, parent_b, rng)
            else:
                child0 = parent_a.copy()

            child = None
            for _ in range(cfg.max_retries):
                cand = mutate_genome(child0, rng, rpg, n_mutations=cfg.n_mutations,
                                     p_const_tweak=cfg.p_const_tweak)
                ok, _ = validate_genome(
                    cand,
                    rng=np.random.default_rng(int(rng.integers(0, 2**31))),
                    deg=cfg.validator_deg,
                )
                if ok:
                    child = cand
                    n_valid += 1
                    break
            if child is None:
                child = parent_a.copy()  # fallback
            new_pop.append(child)
            new_fits.append(float(fitness_fn(child)))

        pop, fits = new_pop, new_fits

        f_arr = np.asarray(fits, dtype=np.float64)
        f_arr = np.where(np.isfinite(f_arr), f_arr, np.inf)
        log = GenLog(
            gen=gen_idx,
            best_fit=float(f_arr.min()),
            mean_fit=float(np.mean(f_arr[np.isfinite(f_arr)])) if np.isfinite(f_arr).any() else float("inf"),
            median_fit=float(np.median(f_arr)),
            n_valid=n_valid,
            elapsed_s=time.time() - t0,
        )
        history.append(log)
        if on_generation is not None:
            on_generation(log, pop, fits)

    best_idx = int(np.argmin(np.where(np.isfinite(fits), fits, np.inf)))
    return EvolutionResult(
        best_genome=pop[best_idx].copy(),
        best_fitness=float(fits[best_idx]),
        history=history,
        final_population=pop,
        final_fitnesses=fits,
    )


__all__ = [
    "EvolutionConfig", "EvolutionResult", "GenLog",
    "TwoPopGenLog", "TwoPopResult",
    "evolve", "evolve_from_scratch",
]


# ===================================================================
# From-scratch, diversity-first evolution (TWO-POP design)
# ===================================================================
#
# Hard rules (per `/memories/repo/gp-search-principles.md`):
#   1. validate_v2c / validate_c2v gate ENTRY.  No parent-copy fallback.
#   2. NO seed.  Initial population is pure random valid programs.
#   3. Elitism off by default; configurable via cfg.elitism (per side).
#   4. Dedup by structural fingerprint, per side.
#   5. Worse individuals (by rank) get more aggressive mutation.
#
# Architecture:
#   * Two independent populations — `pop_v` (V2C programs) and
#     `pop_c` (C2V programs) — each of size `cfg.pop_size`.
#   * Plus a small "constants pool" of size `cfg.pop_size`, each entry
#     being one independent log_constants vector evolved by tweak.
#   * Pairing for fitness: positional, with a per-generation random
#     permutation.  Each generation evaluates `pop_size` triples
#     (v[π(i)], c[i], k[i]).  Per-side fitness = the fitness of the
#     pair the program participated in this gen.
#
# This means programs do NOT carry private constants and the V/C/K
# searches do not directly co-adapt — exactly the principle "no
# structural prior".

from .crossover import crossover_program
from .mutation import mutate_log_constants, mutate_program
from .parallel_init import (
    DEFAULT_WORKERS,
    parallel_fill_random,
    parallel_validate_programs,
)
from .random_program import C2V_INSTR, V2C_INSTR
from .serialize import program_to_dict
from .validators import validate_c2v, validate_v2c

# Per-program fingerprint (lighter than full genome_fingerprint —
# excludes constants since constants live in a separate pool).
def _prog_fingerprint(prog) -> str:
    import json as _json
    return _json.dumps(program_to_dict(prog), sort_keys=True, separators=(",", ":"))


def _const_fingerprint(k: np.ndarray, *, quant: int = 2) -> str:
    return ",".join(f"{round(float(x), quant)}" for x in k)


def _rank_to_n_mutations(rank: int, n_total: int, cfg: "EvolutionConfig") -> int:
    if n_total <= 1 or cfg.n_mutations_max <= cfg.n_mutations:
        return cfg.n_mutations
    frac = rank / float(n_total - 1)
    return int(round(cfg.n_mutations + frac * (cfg.n_mutations_max - cfg.n_mutations)))


def _validate_one(side: str, prog, *, deg: int, seed: int) -> bool:
    val = validate_v2c if side == "v2c" else validate_c2v
    ok, _ = val(prog, rng=np.random.default_rng(seed & 0xFFFFFFFF), deg=deg)
    return bool(ok)


def _evolve_side_offspring(
    *,
    side: str,
    pop: List,
    fits: List[float],
    cfg: "EvolutionConfig",
    rng: np.random.Generator,
    rpg: RandomProgramGenerator,
    instr_set: Sequence[str],
    pool=None,
    n_workers: int = 1,
) -> Tuple[List, int, int]:
    """Build the next-generation pop for one side (V2C or C2V).

    Returns `(new_pop, n_attempts, n_invalid_rejected)`.
    Strategy:
      1. Optional elitism: keep top cfg.elitism untouched.
      2. Round-robin: generate `pop_size - elites` candidates from
         tournament parents (mut/crossover, rank-scaled mutation).
      3. Batch-validate via multiprocessing pool.
      4. Accept valid + distinct.  Repeat until pop full.
    """
    order = np.argsort(np.where(np.isfinite(fits), fits, np.inf))
    rank_of: Dict[int, int] = {int(idx): r for r, idx in enumerate(order)}

    new_pop: List = []
    seen: set = set()
    if cfg.elitism > 0:
        for i in order[: cfg.elitism]:
            g = pop[int(i)]
            fp = _prog_fingerprint(g)
            if cfg.dedup and fp in seen:
                continue
            seen.add(fp)
            from .program import deep_copy_program
            new_pop.append(deep_copy_program(g))

    n_attempts = 0
    n_invalid = 0
    target = cfg.pop_size
    max_attempts = cfg.max_attempts_per_slot * target

    from .program import deep_copy_program
    while len(new_pop) < target:
        if n_attempts >= max_attempts:
            raise RuntimeError(
                f"evolve gen offspring fill exhausted ({side}): "
                f"{len(new_pop)}/{target} after {n_attempts} attempts"
            )
        # Build a batch of candidate offspring.
        batch_size = max(1, min(n_workers * 8, target - len(new_pop) + 16))
        cands = []
        for _ in range(batch_size):
            parent_a = tournament_select(pop, fits, cfg.tournament_k, rng)
            n_mut = _rank_to_n_mutations(rank_of.get(_id_in(pop, parent_a), 0),
                                         len(pop), cfg)
            if rng.random() < cfg.p_crossover:
                parent_b = tournament_select(pop, fits, cfg.tournament_k, rng)
                child0 = crossover_program(parent_a, parent_b, rng,
                                           instr_set=instr_set)
            else:
                child0 = deep_copy_program(parent_a)
            cand = mutate_program(child0, rng, rpg, instr_set,
                                  n_mutations=max(1, n_mut))
            cands.append(cand)
        # Batch-validate.
        if n_workers > 1 and pool is not None:
            base_seed = int(rng.integers(0, 2**31))
            ok_list = parallel_validate_programs(
                side, cands, workers=n_workers,
                deg=cfg.validator_deg, base_seed=base_seed, pool=pool,
            )
        else:
            ok_list = [
                _validate_one(side, c, deg=cfg.validator_deg,
                              seed=int(rng.integers(0, 2**31)))
                for c in cands
            ]
        n_attempts += len(cands)
        for ok, cand in zip(ok_list, cands):
            if not ok:
                n_invalid += 1
                continue
            if cfg.dedup:
                fp = _prog_fingerprint(cand)
                if fp in seen:
                    continue
                seen.add(fp)
            new_pop.append(cand)
            if len(new_pop) >= target:
                break
    return new_pop, n_attempts, n_invalid


def _id_in(lst, item) -> int:
    """Identity-based lookup (tournament returns actual elements)."""
    for i, x in enumerate(lst):
        if x is item:
            return i
    return 0


# ----------------------------------------------------------- log type for two-pop


@dataclass
class TwoPopGenLog:
    gen: int
    best_fit: float
    mean_fit: float
    median_fit: float
    elapsed_s: float
    # New: per-side offspring stats
    v_attempts: int = 0
    v_invalid: int = 0
    c_attempts: int = 0
    c_invalid: int = 0


@dataclass
class TwoPopResult:
    best_genome: Genome
    best_fitness: float
    pop_v: List
    pop_c: List
    pop_k: List
    pair_perm: List[int]  # final-gen pairing
    fits: List[float]     # per-pair fitness, indexed by C-pop position
    history: List[TwoPopGenLog] = field(default_factory=list)


# ----------------------------------------------------------- main entry


def evolve_from_scratch(
    fitness_pair_fn: Callable[[Genome], float],
    cfg: "EvolutionConfig",
    *,
    workers: int = DEFAULT_WORKERS,
    batch_eval_fn: Optional[Callable[[List, List, List, List[int]], List[float]]] = None,
    on_generation: Optional[Callable[
        [TwoPopGenLog, List, List, List, List[int], List[float]], None
    ]] = None,
) -> TwoPopResult:
    """Two-pop from-scratch evolution.

    `fitness_pair_fn(Genome) -> float` evaluates a single (V2C, C2V,
    log_consts) triple, smaller = better.

    Workflow:
      Init: parallel-fill `pop_v`, `pop_c` (size pop_size each); random
            `pop_k` of log_constants.
      For each gen:
        a. Random pairing permutation π over [0, pop_size).
        b. Build genomes (v[π(i)], c[i], k[i]) and call fitness_pair_fn.
        c. Per-side fitness = pair fitness.
        d. Per-side offspring fill via `_evolve_side_offspring`
           (validation gates entry, dedup on, rank-scaled mutation).
        e. Constants pool: pure tweak with prob `p_const_tweak`,
           dedup by quantized fingerprint.
    """
    if cfg.pop_size <= 0:
        raise ValueError("pop_size must be > 0")

    rng = np.random.default_rng(cfg.seed)
    rpg = RandomProgramGenerator(rng=rng)

    # Persistent pool reused across gens for both init and offspring
    # validation (avoids spawning overhead).
    from multiprocessing import Pool
    pool = Pool(processes=workers) if workers > 1 else None
    try:
        # ---- 1. Initial pops via parallel brute-force fill -------------
        t_init = time.time()
        pop_v, v_attempts = parallel_fill_random(
            "v2c", cfg.pop_size,
            max_attempts=cfg.max_attempts_per_slot * cfg.pop_size * 1000,
            workers=workers,
            chunk_attempts=max(1000, cfg.max_attempts_per_slot),
            min_size=cfg.rand_min_size, max_size=cfg.rand_max_size,
            deg=cfg.validator_deg, base_seed=cfg.seed * 17 + 1,
            pool=pool,
        )
        pop_c, c_attempts = parallel_fill_random(
            "c2v", cfg.pop_size,
            max_attempts=cfg.max_attempts_per_slot * cfg.pop_size * 1000,
            workers=workers,
            chunk_attempts=max(1000, cfg.max_attempts_per_slot),
            min_size=cfg.rand_min_size, max_size=cfg.rand_max_size,
            deg=cfg.validator_deg, base_seed=cfg.seed * 17 + 2,
            pool=pool,
        )
        pop_k = [rpg.random_log_constants() for _ in range(cfg.pop_size)]

        # Dedup the initial pop_v / pop_c (parallel_fill_random doesn't
        # dedup, so fingerprint-collide entries get re-randomized via
        # extra fills — usually redundant given the entropy).
        if cfg.dedup:
            seen_v: set = set()
            uniq_v = []
            for p in pop_v:
                fp = _prog_fingerprint(p)
                if fp in seen_v:
                    continue
                seen_v.add(fp)
                uniq_v.append(p)
            while len(uniq_v) < cfg.pop_size:
                extra, n_extra = parallel_fill_random(
                    "v2c", cfg.pop_size - len(uniq_v),
                    max_attempts=cfg.max_attempts_per_slot * cfg.pop_size * 100,
                    workers=workers, chunk_attempts=2000,
                    min_size=cfg.rand_min_size, max_size=cfg.rand_max_size,
                    deg=cfg.validator_deg,
                    base_seed=cfg.seed * 17 + 1001 + len(uniq_v),
                    pool=pool,
                )
                v_attempts += n_extra
                for p in extra:
                    fp = _prog_fingerprint(p)
                    if fp in seen_v:
                        continue
                    seen_v.add(fp)
                    uniq_v.append(p)
                    if len(uniq_v) >= cfg.pop_size:
                        break
            pop_v = uniq_v[: cfg.pop_size]

            seen_c: set = set()
            uniq_c = []
            for p in pop_c:
                fp = _prog_fingerprint(p)
                if fp in seen_c:
                    continue
                seen_c.add(fp)
                uniq_c.append(p)
            while len(uniq_c) < cfg.pop_size:
                extra, n_extra = parallel_fill_random(
                    "c2v", cfg.pop_size - len(uniq_c),
                    max_attempts=cfg.max_attempts_per_slot * cfg.pop_size * 100,
                    workers=workers, chunk_attempts=2000,
                    min_size=cfg.rand_min_size, max_size=cfg.rand_max_size,
                    deg=cfg.validator_deg,
                    base_seed=cfg.seed * 17 + 2001 + len(uniq_c),
                    pool=pool,
                )
                c_attempts += n_extra
                for p in extra:
                    fp = _prog_fingerprint(p)
                    if fp in seen_c:
                        continue
                    seen_c.add(fp)
                    uniq_c.append(p)
                    if len(uniq_c) >= cfg.pop_size:
                        break
            pop_c = uniq_c[: cfg.pop_size]

        print(f"[init] V pop filled: {len(pop_v)} valid in {v_attempts} attempts "
              f"({len(pop_v)/v_attempts:.4%})", flush=True)
        print(f"[init] C pop filled: {len(pop_c)} valid in {c_attempts} attempts "
              f"({len(pop_c)/c_attempts:.4%})", flush=True)
        print(f"[init] elapsed: {time.time()-t_init:.1f}s", flush=True)

        # ---- 2. Initial fitness evaluation via positional pairing -----
        perm = list(rng.permutation(cfg.pop_size))
        if batch_eval_fn is not None:
            t_e = time.time()
            fits = batch_eval_fn(pop_v, pop_c, pop_k, perm)
            print(f"[init-eval] {len(fits)} pairs in {time.time()-t_e:.1f}s "
                  f"(parallel)  best={min(fits):+.4f} med={float(np.median(fits)):+.4f}",
                  flush=True)
        else:
            fits = _eval_pairs(pop_v, pop_c, pop_k, perm, fitness_pair_fn,
                               prefix="[init-eval]")

        history: List[TwoPopGenLog] = []

        # ---- 3. Generation loop ---------------------------------------
        for gen_idx in range(cfg.generations):
            t0 = time.time()
            # Per-side fitness: each entry got one pair fitness this gen.
            # fits[i] is the fitness of pair (v[perm[i]], c[i], k[i]).
            fits_c = list(fits)
            fits_v = [0.0] * cfg.pop_size
            fits_k = [0.0] * cfg.pop_size
            for i, vi in enumerate(perm):
                fits_v[vi] = fits[i]
                fits_k[i] = fits[i]

            new_pop_v, va, vinv = _evolve_side_offspring(
                side="v2c", pop=pop_v, fits=fits_v, cfg=cfg, rng=rng,
                rpg=rpg, instr_set=V2C_INSTR, pool=pool, n_workers=workers,
            )
            new_pop_c, ca, cinv = _evolve_side_offspring(
                side="c2v", pop=pop_c, fits=fits_c, cfg=cfg, rng=rng,
                rpg=rpg, instr_set=C2V_INSTR, pool=pool, n_workers=workers,
            )
            new_pop_k = _evolve_constants(
                pop_k, fits_k, cfg, rng,
            )

            pop_v, pop_c, pop_k = new_pop_v, new_pop_c, new_pop_k

            perm = list(rng.permutation(cfg.pop_size))
            if batch_eval_fn is not None:
                t_e = time.time()
                fits = batch_eval_fn(pop_v, pop_c, pop_k, perm)
                print(f"[gen {gen_idx}] eval {len(fits)} pairs in "
                      f"{time.time()-t_e:.1f}s (parallel)  "
                      f"best={min(fits):+.4f} med={float(np.median(fits)):+.4f}",
                      flush=True)
            else:
                fits = _eval_pairs(
                    pop_v, pop_c, pop_k, perm, fitness_pair_fn,
                    prefix=f"[gen {gen_idx}]",
                )

            f_arr = np.asarray(fits, dtype=np.float64)
            f_arr = np.where(np.isfinite(f_arr), f_arr, np.inf)
            log = TwoPopGenLog(
                gen=gen_idx,
                best_fit=float(f_arr.min()),
                mean_fit=(float(np.mean(f_arr[np.isfinite(f_arr)]))
                          if np.isfinite(f_arr).any() else float("inf")),
                median_fit=float(np.median(f_arr)),
                elapsed_s=time.time() - t0,
                v_attempts=va, v_invalid=vinv,
                c_attempts=ca, c_invalid=cinv,
            )
            history.append(log)
            if on_generation is not None:
                on_generation(log, pop_v, pop_c, pop_k, perm, fits)

        # ---- 4. Build best genome from final pairing ------------------
        best_idx = int(np.argmin(np.where(np.isfinite(fits), fits, np.inf)))
        best_v = pop_v[perm[best_idx]]
        best_c = pop_c[best_idx]
        best_k = pop_k[best_idx]
        from .program import deep_copy_program
        best_genome = Genome(
            prog_v2c=deep_copy_program(best_v),
            prog_c2v=deep_copy_program(best_c),
            log_constants=best_k.copy(),
        )
        return TwoPopResult(
            best_genome=best_genome,
            best_fitness=float(fits[best_idx]),
            pop_v=pop_v, pop_c=pop_c, pop_k=pop_k,
            pair_perm=perm, fits=fits,
            history=history,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def _eval_pairs(
    pop_v, pop_c, pop_k, perm, fitness_pair_fn, *, prefix=""
) -> List[float]:
    """Evaluate `len(pop_c)` triples (v[perm[i]], c[i], k[i])."""
    from .program import deep_copy_program
    n = len(pop_c)
    fits: List[float] = []
    for i in range(n):
        g = Genome(
            prog_v2c=deep_copy_program(pop_v[perm[i]]),
            prog_c2v=deep_copy_program(pop_c[i]),
            log_constants=pop_k[i].copy(),
        )
        f = float(fitness_pair_fn(g))
        fits.append(f)
        if prefix:
            print(f"{prefix} pair {i:>3d}/{n}  fit={f:+.4f}", flush=True)
    return fits


def _evolve_constants(
    pop_k: List[np.ndarray],
    fits_k: List[float],
    cfg: "EvolutionConfig",
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Constants pool: tweak top survivors, dedup by quantized hash."""
    n = cfg.pop_size
    order = np.argsort(np.where(np.isfinite(fits_k), fits_k, np.inf))
    new_pop: List[np.ndarray] = []
    seen: set = set()
    if cfg.elitism > 0:
        for i in order[: cfg.elitism]:
            k = pop_k[int(i)].copy()
            fp = _const_fingerprint(k)
            if cfg.dedup and fp in seen:
                continue
            seen.add(fp)
            new_pop.append(k)
    while len(new_pop) < n:
        parent_idx = int(tournament_select_idx(fits_k, cfg.tournament_k, rng))
        k = pop_k[parent_idx].copy()
        # Always tweak to ensure diversity (offspring should differ).
        k = mutate_log_constants(k, rng,
                                 n_tweaks=max(1, int(rng.integers(1, 3))))
        fp = _const_fingerprint(k)
        if cfg.dedup and fp in seen:
            # Tweak harder
            k = mutate_log_constants(k, rng, n_tweaks=3, sigma=0.6)
            fp = _const_fingerprint(k)
            if fp in seen:
                continue
        seen.add(fp)
        new_pop.append(k)
    return new_pop


def tournament_select_idx(
    fits: List[float],
    k: int,
    rng: np.random.Generator,
) -> int:
    """Index-returning tournament select (smaller fitness wins)."""
    n = len(fits)
    if n == 0:
        raise ValueError("empty pop")
    candidates = rng.integers(0, n, size=min(k, n))
    best_i = int(candidates[0])
    best_f = fits[best_i]
    for ci in candidates[1:]:
        cf = fits[int(ci)]
        if (np.isfinite(cf) and (not np.isfinite(best_f) or cf < best_f)):
            best_i = int(ci)
            best_f = cf
    return best_i

