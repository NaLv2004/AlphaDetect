"""Test: every generation must produce pop_size BEHAVIORALLY-distinct individuals.

This validates the fix for the dead-code-equivalence problem: previously, ~70%
of the population had identical V2C output vectors on a fixed input panel even
though every program had a unique structural fingerprint.  After switching the
dedup key from structural to behavioral, EVERY individual in EVERY generation
must yield a distinct behavioral fingerprint on the same fixed panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from pushgp.evolution import (
    EvolutionConfig,
    _behav_fingerprint,
    evolve_from_scratch,
)
from pushgp.genome import Genome
from pushgp.random_program import RandomProgramGenerator


def _trivial_fitness(g: Genome) -> float:
    """Random-but-deterministic fitness so evolve_from_scratch can run.

    We hash a piece of the genome to make pairs distinguishable enough that
    selection has something to do.  The actual values are irrelevant for this
    test — we only check behavioral diversity of the populations.
    """
    h = hash(repr(g.prog_v2c)[:200] + repr(g.prog_c2v)[:200]
             + str(tuple(round(x, 3) for x in g.log_constants)))
    return float((h % 1000) / 100.0 - 5.0)


def test_per_gen_behavioral_uniqueness():
    cfg = EvolutionConfig(
        pop_size=8,
        generations=1,
        elitism=1,
        tournament_k=2,
        p_crossover=0.5,
        n_mutations=2,
        seed=12345,
        rand_min_size=4,
        rand_max_size=12,
        max_attempts_per_slot=200,
        dedup=True,
    )
    rpg = RandomProgramGenerator(rng=np.random.default_rng(cfg.seed))

    captured: list = []

    def cb(_log, pop_v, pop_c, pop_k, _perm, fits):
        v_fps = [_behav_fingerprint("v2c", p) for p in pop_v]
        c_fps = [_behav_fingerprint("c2v", p) for p in pop_c]
        captured.append({
            "n_v_distinct": len(set(v_fps)),
            "n_c_distinct": len(set(c_fps)),
            "pop_size": cfg.pop_size,
            "v_fp_sample": v_fps[0],
        })

    res = evolve_from_scratch(
        _trivial_fitness,
        cfg,
        on_generation=cb,
        workers=4,
    )
    assert res is not None

    # Verify per-generation populations
    print(f"\ncaptured {len(captured)} generations")
    for i, snap in enumerate(captured):
        print(f"  gen {i}: V distinct={snap['n_v_distinct']}/{snap['pop_size']}  "
              f"C distinct={snap['n_c_distinct']}/{snap['pop_size']}")
        assert snap["n_v_distinct"] == snap["pop_size"], (
            f"gen {i} V2C: only {snap['n_v_distinct']}/{snap['pop_size']} "
            f"behaviorally distinct individuals"
        )
        assert snap["n_c_distinct"] == snap["pop_size"], (
            f"gen {i} C2V: only {snap['n_c_distinct']}/{snap['pop_size']} "
            f"behaviorally distinct individuals"
        )


if __name__ == "__main__":
    test_per_gen_behavioral_uniqueness()
    print("\nOK: every generation produced pop_size behaviorally-distinct individuals.")
