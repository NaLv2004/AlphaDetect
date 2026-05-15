"""Top-level driver for evolving an LDPC BP decoder via Push-GP.

Usage:

    python -m pushgp_ldpc.main_evolve \
        --snr-list 1.0 2.0 3.0 \
        --pop-size 12 --generations 5 \
        --max-iter 8 --frames 4 \
        --bgn 2 --set 1 --zc 2 \
        --out-dir results/evolve_smoke

All parameters are CLI-tunable.  No hard-coded SNR list (per project
convention).  The default seed is the hand-coded OMS genome saved at
`pushgp_ldpc/seeds/oms.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_5g import build_parity
from pushgp.evolution import EvolutionConfig, evolve
from pushgp.genome import Genome
from pushgp_ldpc.adapter import load_oms_seed, save_oms_seed
from pushgp_ldpc.eval import FitnessConfig, evaluate_genome


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Push-GP evolution of LDPC BP decoder.")
    p.add_argument("--snr-list", type=float, nargs="+", required=True,
                   help="SNR points (dB) to evaluate fitness on.")
    p.add_argument("--pop-size", type=int, default=12)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--elitism", type=int, default=2)
    p.add_argument("--tournament-k", type=int, default=3)
    p.add_argument("--frames", type=int, default=4,
                   help="Decoded frames per SNR per genome.")
    p.add_argument("--max-iter", type=int, default=8)
    p.add_argument("--bgn", type=int, default=2)
    p.add_argument("--set", dest="set_idx", type=int, default=1)
    p.add_argument("--zc", type=int, default=2)
    p.add_argument("--code-rate", type=float, default=0.5)
    p.add_argument("--n-mutations", type=int, default=2)
    p.add_argument("--p-const-tweak", type=float, default=0.2)
    p.add_argument("--p-crossover", type=float, default=0.5)
    p.add_argument("--max-retries", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-genome", type=str, default=None,
                   help="Path to a Genome JSON to use as evolution seed (default: OMS).")
    p.add_argument("--out-dir", type=str, default="results/evolve_smoke")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Make sure the canonical OMS seed file exists.
    save_oms_seed()

    if args.seed_genome:
        seed = Genome.load(Path(args.seed_genome))
    else:
        seed = load_oms_seed()

    par = build_parity(args.bgn, args.set_idx, args.zc)
    fit_cfg = FitnessConfig(
        par=par,
        snr_list=tuple(args.snr_list),
        n_frames_per_snr=args.frames,
        max_iter=args.max_iter,
        code_rate=args.code_rate,
    )

    def fitness(g: Genome) -> float:
        return evaluate_genome(g, fit_cfg)

    cfg = EvolutionConfig(
        pop_size=args.pop_size,
        generations=args.generations,
        elitism=args.elitism,
        tournament_k=args.tournament_k,
        n_mutations=args.n_mutations,
        p_const_tweak=args.p_const_tweak,
        p_crossover=args.p_crossover,
        max_retries=args.max_retries,
        seed=args.seed,
    )

    history_log = []

    def on_gen(log, pop, fits):
        history_log.append({
            "gen": log.gen,
            "best_fit": log.best_fit,
            "mean_fit": log.mean_fit,
            "median_fit": log.median_fit,
            "n_valid": log.n_valid,
            "elapsed_s": log.elapsed_s,
        })
        print(f"[gen {log.gen}] best={log.best_fit:.4f} mean={log.mean_fit:.4f} "
              f"median={log.median_fit:.4f} valid={log.n_valid} "
              f"t={log.elapsed_s:.2f}s")

    print(f"[init] code BG{args.bgn} set{args.set_idx} Zc={args.zc} "
          f"(N={par.cols}, M={par.rows}); SNR={list(args.snr_list)} dB; "
          f"frames/SNR={args.frames}; max_iter={args.max_iter}")
    seed_fit = fitness(seed)
    print(f"[init] OMS seed fitness = {seed_fit:.4f}")

    res = evolve(fitness, [seed], cfg, on_generation=on_gen)
    print(f"[done] best_fitness = {res.best_fitness:.4f}")

    res.best_genome.save(out / "champion.json")
    with open(out / "history.json", "w") as f:
        json.dump({
            "args": vars(args),
            "seed_fit": seed_fit,
            "best_fit": res.best_fitness,
            "history": history_log,
        }, f, indent=2)
    return res


if __name__ == "__main__":
    main()
