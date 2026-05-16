"""Smoke test for the `bind_pairs` mode of `evolve_from_scratch`.

Checks:

  1. With `bind_pairs=True`, the per-generation `perm` reported via the
     callback is always identity `[0, 1, ..., N-1]`.
  2. The elite triple (rank-0 by fitness) is preserved across generations
     at slot 0 — same V2C / C2V / K *bytes* — proving the binding is
     atomic.
  3. With `bind_pairs=False`, the perm changes generation-to-generation
     (legacy CCEA behavior) — proves the toggle works.

Run:
    conda run -n AutoGenOld python research/bp-deocder-synthesis/code_review/smoke_bind_pairs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np

from pushgp.evolution import EvolutionConfig, evolve_from_scratch
from pushgp.genome import Genome
from pushgp.serialize import program_to_dict


def _stub_fitness(g: Genome) -> float:
    """Cheap deterministic fitness: program length + K-norm.  No real BP."""
    return float(len(g.prog_v2c) + len(g.prog_c2v)
                 + 0.01 * float(np.linalg.norm(g.log_constants)))


def _prog_bytes(prog) -> str:
    import json
    return json.dumps(program_to_dict(prog), sort_keys=True,
                       separators=(",", ":"))


def run(bind: bool) -> dict:
    seen_perms: list[list[int]] = []
    elite_v: list[str] = []
    elite_c: list[str] = []
    elite_k: list[tuple[float, ...]] = []

    def on_gen(log, pop_v, pop_c, pop_k, perm, fits):
        seen_perms.append(list(perm))
        # Index-0 elite for that gen.
        elite_v.append(_prog_bytes(pop_v[perm[0]]))
        elite_c.append(_prog_bytes(pop_c[0]))
        elite_k.append(tuple(float(x) for x in pop_k[0]))

    cfg = EvolutionConfig(
        pop_size=8, generations=3, elitism=2, tournament_k=3,
        p_crossover=0.7, n_mutations=2, p_const_tweak=0.25,
        seed=42, max_attempts_per_slot=2000,
        rand_min_size=6, rand_max_size=20,
        cpp_seeder=True, bind_pairs=bind,
    )
    evolve_from_scratch(_stub_fitness, cfg, workers=4,
                          on_generation=on_gen)
    return {"perms": seen_perms, "elite_v": elite_v,
            "elite_c": elite_c, "elite_k": elite_k}


def main() -> int:
    print("=== bind_pairs=True ===")
    r_bind = run(True)
    for i, p in enumerate(r_bind["perms"]):
        ok = p == list(range(len(p)))
        print(f"  gen{i} perm identity: {ok}  perm[:5]={p[:5]}")
        assert ok, f"bind_pairs=True must use identity perm; got {p}"

    print("=== bind_pairs=False (legacy) ===")
    r_free = run(False)
    distinct_perms = len({tuple(p) for p in r_free["perms"]})
    print(f"  distinct perms across {len(r_free['perms'])} gens: {distinct_perms}")
    assert distinct_perms >= 2, (
        f"bind_pairs=False should produce changing perms; got "
        f"{distinct_perms} distinct across {len(r_free['perms'])} gens")

    print("\nSMOKE PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
