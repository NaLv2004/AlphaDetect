"""End-to-end smoke test of slot micro-evolution.

Spins up a SubprocessMIMOEvaluator, runs ``step_slot_population`` for
3 micro-gens on 4 representative detector genomes, and reports whether
slot evolution produces measurable SER improvements.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np

from evolution.ir_pool import build_ir_pool
from evolution.mimo_evaluator import MIMOEvalConfig
from evolution.subprocess_evaluator import SubprocessMIMOEvaluator
from evolution.slot_evolution import step_slot_population


def main() -> int:
    pool = build_ir_pool(np.random.default_rng(42))
    by_id = {g.algo_id: g for g in pool}

    cfg = MIMOEvalConfig(
        Nr=4, Nt=4, mod_order=16, snr_db_list=[16.0],
        n_trials=8, timeout_sec=1.0, batch_workers=1,
    )
    evaluator = SubprocessMIMOEvaluator(cfg)

    rng = np.random.default_rng(2026)

    targets = ["lmmse", "zf", "osic", "stack"]

    for algo_id in targets:
        g = by_id.get(algo_id)
        if g is None or not g.slot_populations:
            print(f"[skip] {algo_id}: no slot populations")
            continue
        for pop_key, pop in list(g.slot_populations.items()):
            if not pop.variants:
                continue
            print(f"\n=== {algo_id}.{pop_key.split('.')[-1]} === "
                  f"(initial pop size: {len(pop.variants)})")
            best_history = []
            for gen in range(3):
                stats = step_slot_population(
                    g, pop_key, pop,
                    evaluator=evaluator, rng=rng,
                    n_children=4, n_trials=12, timeout_sec=1.5,
                    snr_db=16.0, max_pop_size=12, perturb_scale=0.2,
                )
                best_history.append(stats.best_after)
                d = stats.as_dict()
                print(f"  gen{gen}: attempted={d['n_attempted']:2d} "
                      f"validated={d['n_validated']:2d} "
                      f"evaluated={d['n_evaluated']:2d} "
                      f"improved={d['n_improved']:2d} "
                      f"best_before={d['best_before']:.4f} "
                      f"best_after={d['best_after']:.4f} "
                      f"delta={d['best_delta']:+.4f}")
            print(f"  best history: {[f'{x:.4f}' for x in best_history]}")

    evaluator.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
