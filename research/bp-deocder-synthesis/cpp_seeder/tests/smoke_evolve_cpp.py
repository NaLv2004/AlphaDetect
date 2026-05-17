"""Tiny smoke test for the cpp_seeder integration in evolve_from_scratch.
Uses a stub fitness function to skip the LDPC decoder."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # cpp_seeder
sys.path.insert(0, str(ROOT.parent))           # bp-deocder-synthesis (for pushgp)

import numpy as np
from pushgp.evolution import EvolutionConfig, evolve_from_scratch
from pushgp.genome import Genome


def stub_fit(g: Genome) -> float:
    return float(np.random.default_rng(hash(id(g)) & 0xFFFFFFFF).normal())


def main():
    cfg = EvolutionConfig(
        pop_size=12,
        generations=1,
        elitism=2,
        tournament_k=3,
        n_mutations=2,
        p_const_tweak=0.25,
        seed=999,
        max_attempts_per_slot=2000,
        rand_min_size=4,
        rand_max_size=20,
        dedup=True,
    )
    res = evolve_from_scratch(stub_fit, cfg, workers=4)
    print(f"OK: best_fit={res.best_fitness:.4f}, n_history={len(res.history)}")


if __name__ == "__main__":
    main()
