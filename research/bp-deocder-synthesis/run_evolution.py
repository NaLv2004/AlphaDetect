"""PR8 driver: run a real evolution then benchmark champion vs OMS."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_5g import build_parity
from pushgp.evolution import EvolutionConfig, evolve
from pushgp.genome import Genome
from pushgp_ldpc.adapter import load_oms_seed, save_oms_seed
from pushgp_ldpc.benchmark import plot_results, run_benchmark, write_csv
from pushgp_ldpc.eval import FitnessConfig, evaluate_genome


def main():
    out_dir = Path("results/evolve_run1")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_oms_seed()
    seed = load_oms_seed()

    # Modest evolution settings — small code, few SNRs/frames.
    par = build_parity(bgn=2, set_idx=1, zc=2)  # 84 x 104

    fit_cfg = FitnessConfig(
        par=par,
        snr_list=(-3.0, -2.0),
        n_frames_per_snr=4,
        max_iter=6,
        # code_rate left as None => derived from `par` (physical_code_rate).
    )

    cfg = EvolutionConfig(
        pop_size=10,
        generations=15,
        elitism=2,
        tournament_k=3,
        n_mutations=2,
        p_const_tweak=0.2,
        p_crossover=0.5,
        max_retries=15,
        seed=2025,
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
        print(f"[gen {log.gen:02d}] best={log.best_fit:.4f} "
              f"mean={log.mean_fit:.4f} median={log.median_fit:.4f} "
              f"valid={log.n_valid} t={log.elapsed_s:.2f}s",
              flush=True)

    print(f"[init] code BG2 set1 Zc=2 (N={par.cols}, M={par.rows})",
          flush=True)
    seed_fit = evaluate_genome(seed, fit_cfg)
    print(f"[init] OMS seed fitness = {seed_fit:.4f}", flush=True)
    print(f"[init] running evolution: pop={cfg.pop_size} gens={cfg.generations}",
          flush=True)

    t0 = time.time()
    res = evolve(lambda g: evaluate_genome(g, fit_cfg), [seed], cfg,
                 on_generation=on_gen)
    t_evolve = time.time() - t0
    print(f"[done] evolution took {t_evolve:.1f}s; best={res.best_fitness:.4f}",
          flush=True)

    res.best_genome.save(out_dir / "champion.json")

    # Benchmark champion vs OMS at a small waterfall.
    print("[bench] running benchmark…", flush=True)
    t0 = time.time()
    from pushgp_ldpc.eval import physical_code_rate
    bench = run_benchmark(
        res.best_genome, par,
        snr_list=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        n_frames=40, max_iter=8, code_rate=physical_code_rate(par), seed=99,
    )
    print(f"[bench] took {time.time() - t0:.1f}s", flush=True)
    write_csv(bench, out_dir / "bench.csv")
    plot_ok = plot_results(bench, out_dir / "bench.png")
    print(f"[bench] plot saved: {plot_ok}", flush=True)
    for i, snr in enumerate(bench.snr_db):
        print(f"  SNR={snr:.1f} dB  BER OMS={bench.ber_oms[i]:.3e}  "
              f"BER Champ={bench.ber_champ[i]:.3e}  "
              f"FER OMS={bench.fer_oms[i]:.3e}  FER Champ={bench.fer_champ[i]:.3e}",
              flush=True)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "seed_fit": seed_fit,
            "best_fit": res.best_fitness,
            "history": history_log,
            "bench": {
                "snr_db": bench.snr_db,
                "ber_oms": bench.ber_oms,
                "ber_champ": bench.ber_champ,
                "fer_oms": bench.fer_oms,
                "fer_champ": bench.fer_champ,
            },
            "evolve_seconds": t_evolve,
        }, f, indent=2)


if __name__ == "__main__":
    main()
