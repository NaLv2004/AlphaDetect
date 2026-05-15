"""Run a large-scale Push-GP evolution and dump every individual of every
generation to a JSONL log along with its full syntax tree, fitness, and
per-SNR BER/FER.  Designed to be polled live by `poll_log.py`.

Output layout:
  results/logged_evolution/<run_name>/
      meta.json          — run config, code params, SNR list, seed, started_at
      individuals.jsonl  — one JSON record per individual per generation
      gen_summary.jsonl  — one JSON record per generation (best/mean/median)
      champion.json      — best genome at end of run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow `python experiments/run_logged_evolution.py` from project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _PARENT)
sys.path.insert(0, os.path.dirname(_PARENT))

from ldpc_5g import build_parity
from pushgp.evolution import (
    EvolutionConfig, TwoPopGenLog, evolve, evolve_from_scratch,
)
from pushgp.genome import Genome
from pushgp.parallel_init import DEFAULT_WORKERS
from pushgp.program import deep_copy_program
from pushgp.serialize import genome_to_dict, tree_max_depth, tree_size
from pushgp_ldpc.adapter import load_oms_seed, save_oms_seed
from pushgp_ldpc.eval import FitnessConfig
from pushgp_ldpc.eval_logged import GenomeMetrics, evaluate_genome_with_ber


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None,
                   help="Sub-folder under results/logged_evolution/")
    p.add_argument("--pop-size", type=int, default=40)
    p.add_argument("--gens", type=int, default=20)
    p.add_argument("--elitism", type=int, default=3)
    p.add_argument("--tournament-k", type=int, default=4)
    p.add_argument("--p-crossover", type=float, default=0.7)
    p.add_argument("--n-mutations", type=int, default=2)
    p.add_argument("--p-const-tweak", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=2025)
    # Code + fitness eval
    p.add_argument("--bgn", type=int, default=2)
    p.add_argument("--set-idx", type=int, default=1)
    p.add_argument("--zc", type=int, default=2)
    p.add_argument("--snr-list", type=str, default="-3,-2,-1",
                   help="Comma-sep SNRs in dB.")
    p.add_argument("--n-frames", type=int, default=6)
    p.add_argument("--max-iter", type=int, default=8)
    # ---- From-scratch / diversity-first knobs ----
    p.add_argument("--from-scratch", action="store_true", default=True,
                   help="Initialize from random valid genomes (no OMS seed). "
                        "Default ON.")
    p.add_argument("--use-oms-seed", dest="from_scratch", action="store_false",
                   help="Override --from-scratch and seed with OMS instead.")
    p.add_argument("--n-mutations-max", type=int, default=6,
                   help="Mutation count for the rank-worst individual; best "
                        "still gets --n-mutations.")
    p.add_argument("--no-dedup", dest="dedup", action="store_false",
                   default=True,
                   help="Disable structural-fingerprint deduplication.")
    p.add_argument("--max-attempts-per-slot", type=int, default=500,
                   help="Max validation attempts per population slot before "
                        "raising.")
    p.add_argument("--rand-min-size", type=int, default=4)
    p.add_argument("--rand-max-size", type=int, default=30)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="Multiprocessing pool size for parallel random "
                        "init + offspring validation.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class GenerationLogger:
    """Logger supporting both legacy single-pop `evolve()` and the
    new two-pop `evolve_from_scratch()` callbacks.

    Two-pop callback signature:
        on_generation(gen_log: TwoPopGenLog,
                      pop_v, pop_c, pop_k,
                      perm: List[int], fits: List[float])
    where pair i = (v[perm[i]], c[i], k[i]).

    Legacy single-pop signature:
        on_generation(gen_log: GenLog, pop: List[Genome], fits: List[float])
    """
    def __init__(self, out_dir: Path, fit_cfg: FitnessConfig):
        self.out_dir = out_dir
        self.ind_path = out_dir / "individuals.jsonl"
        self.sum_path = out_dir / "gen_summary.jsonl"
        self.fit_cfg = fit_cfg
        self._cache: Dict[int, GenomeMetrics] = {}

    def fitness_fn(self, g: Genome) -> float:
        m = evaluate_genome_with_ber(g, self.fit_cfg)
        self._cache[id(g)] = m
        return m.fitness

    def _metrics_for(self, g: Genome) -> GenomeMetrics:
        m = self._cache.get(id(g))
        if m is None:
            m = evaluate_genome_with_ber(g, self.fit_cfg)
            self._cache[id(g)] = m
        return m

    # Legacy single-pop callback (kept for `evolve()` mode).
    def on_generation_single(self, gen_log, pop: List[Genome],
                             fits: List[float]) -> None:
        with self.ind_path.open("a", encoding="utf-8") as fh:
            for i, (g, f) in enumerate(zip(pop, fits)):
                m = self._metrics_for(g)
                gd = genome_to_dict(g)
                rec = {
                    "gen": int(gen_log.gen),
                    "idx": int(i),
                    "fitness": float(f),
                    "ber_per_snr": [float(b) for b in m.ber_per_snr],
                    "fer_per_snr": [float(b) for b in m.fer_per_snr],
                    "valid": bool(m.valid),
                    "v2c_size": tree_size(g.prog_v2c),
                    "c2v_size": tree_size(g.prog_c2v),
                    "v2c_max_depth": tree_max_depth(g.prog_v2c),
                    "c2v_max_depth": tree_max_depth(g.prog_c2v),
                    "log_constants": gd["log_constants"],
                    "v2c": gd["prog_v2c"],
                    "c2v": gd["prog_c2v"],
                    "error": m.error,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with self.sum_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "gen": int(gen_log.gen),
                "best_fit": float(gen_log.best_fit),
                "mean_fit": float(gen_log.mean_fit),
                "median_fit": float(gen_log.median_fit),
                "n_valid_offspring": int(gen_log.n_valid),
                "elapsed_s": float(gen_log.elapsed_s),
            }) + "\n")
        self._cache.clear()
        print(
            f"[gen {gen_log.gen:03d}] best={gen_log.best_fit:+.4f} "
            f"mean={gen_log.mean_fit:+.4f} med={gen_log.median_fit:+.4f} "
            f"valid={gen_log.n_valid} t={gen_log.elapsed_s:.1f}s",
            flush=True,
        )

    # New two-pop callback.
    def on_generation_two_pop(
        self,
        gen_log: TwoPopGenLog,
        pop_v: List, pop_c: List, pop_k: List,
        perm: List[int], fits: List[float],
    ) -> None:
        # Log every paired triple this generation as one "individual"
        # record, plus also dump the raw v / c / k pools so a downstream
        # analyzer can re-pair however it likes.
        with self.ind_path.open("a", encoding="utf-8") as fh:
            for i, f in enumerate(fits):
                v_prog = pop_v[perm[i]]
                c_prog = pop_c[i]
                k = pop_k[i]
                g = Genome(prog_v2c=deep_copy_program(v_prog),
                           prog_c2v=deep_copy_program(c_prog),
                           log_constants=k.copy())
                gd = genome_to_dict(g)
                # Quick re-eval to recover BER/FER (cached in evolve loop
                # only inside the worker; here we fall through to the
                # full eval — same cost as the search itself).
                m = evaluate_genome_with_ber(g, self.fit_cfg)
                rec = {
                    "gen": int(gen_log.gen),
                    "idx": int(i),
                    "v_idx": int(perm[i]),
                    "c_idx": int(i),
                    "k_idx": int(i),
                    "fitness": float(f),
                    "ber_per_snr": [float(b) for b in m.ber_per_snr],
                    "fer_per_snr": [float(b) for b in m.fer_per_snr],
                    "valid": bool(m.valid),
                    "v2c_size": tree_size(g.prog_v2c),
                    "c2v_size": tree_size(g.prog_c2v),
                    "v2c_max_depth": tree_max_depth(g.prog_v2c),
                    "c2v_max_depth": tree_max_depth(g.prog_c2v),
                    "log_constants": gd["log_constants"],
                    "v2c": gd["prog_v2c"],
                    "c2v": gd["prog_c2v"],
                    "error": m.error,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with self.sum_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "gen": int(gen_log.gen),
                "best_fit": float(gen_log.best_fit),
                "mean_fit": float(gen_log.mean_fit),
                "median_fit": float(gen_log.median_fit),
                "v_offspring_attempts": int(gen_log.v_attempts),
                "v_offspring_invalid": int(gen_log.v_invalid),
                "c_offspring_attempts": int(gen_log.c_attempts),
                "c_offspring_invalid": int(gen_log.c_invalid),
                "elapsed_s": float(gen_log.elapsed_s),
            }) + "\n")
        self._cache.clear()
        v_rate = (1.0 - gen_log.v_invalid / gen_log.v_attempts) if gen_log.v_attempts > 0 else 0.0
        c_rate = (1.0 - gen_log.c_invalid / gen_log.c_attempts) if gen_log.c_attempts > 0 else 0.0
        print(
            f"[gen {gen_log.gen:03d}] best={gen_log.best_fit:+.4f} "
            f"mean={gen_log.mean_fit:+.4f} med={gen_log.median_fit:+.4f} "
            f"v_acc={v_rate:.2%} ({gen_log.v_attempts}att) "
            f"c_acc={c_rate:.2%} ({gen_log.c_attempts}att) "
            f"t={gen_log.elapsed_s:.1f}s",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    snr_list: Tuple[float, ...] = tuple(float(x) for x in args.snr_list.split(","))

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path("results") / "logged_evolution" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Truncate (fresh run).
    (out_dir / "individuals.jsonl").write_text("", encoding="utf-8")
    (out_dir / "gen_summary.jsonl").write_text("", encoding="utf-8")

    # Build code + fitness cfg.
    par = build_parity(bgn=args.bgn, set_idx=args.set_idx, zc=args.zc)
    fit_cfg = FitnessConfig(
        par=par,
        snr_list=snr_list,
        n_frames_per_snr=args.n_frames,
        max_iter=args.max_iter,
        code_rate=0.5,
    )

    # Seed (only used if --use-oms-seed).
    if not args.from_scratch:
        save_oms_seed()
        seed = load_oms_seed()
    else:
        seed = None

    # Ev cfg.  Default elitism=0 in from-scratch mode (weak-elite/no-elite
    # principle from /memories/repo/gp-search-principles.md).  Caller can
    # override via --elitism on the CLI.
    elitism = args.elitism
    if args.from_scratch and "--elitism" not in sys.argv:
        elitism = 0
    ev_cfg = EvolutionConfig(
        pop_size=args.pop_size,
        generations=args.gens,
        elitism=elitism,
        tournament_k=args.tournament_k,
        n_mutations=args.n_mutations,
        n_mutations_max=args.n_mutations_max,
        p_const_tweak=args.p_const_tweak,
        p_crossover=args.p_crossover,
        max_retries=20,
        seed=args.seed,
        max_attempts_per_slot=args.max_attempts_per_slot,
        rand_min_size=args.rand_min_size,
        rand_max_size=args.rand_max_size,
        dedup=args.dedup,
    )

    meta = {
        "run_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "from_scratch": bool(args.from_scratch),
        "code": {"bgn": args.bgn, "set_idx": args.set_idx, "zc": args.zc,
                 "N": int(par.cols), "M": int(par.rows)},
        "snr_list_db": list(snr_list),
        "n_frames_per_snr": args.n_frames,
        "max_iter": args.max_iter,
        "evolution": {
            "pop_size": ev_cfg.pop_size,
            "generations": ev_cfg.generations,
            "elitism": ev_cfg.elitism,
            "tournament_k": ev_cfg.tournament_k,
            "p_crossover": ev_cfg.p_crossover,
            "n_mutations": ev_cfg.n_mutations,
            "n_mutations_max": ev_cfg.n_mutations_max,
            "p_const_tweak": ev_cfg.p_const_tweak,
            "seed": ev_cfg.seed,
            "dedup": ev_cfg.dedup,
            "max_attempts_per_slot": ev_cfg.max_attempts_per_slot,
            "rand_min_size": ev_cfg.rand_min_size,
            "rand_max_size": ev_cfg.rand_max_size,
        },
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[init] run dir: {out_dir}", flush=True)
    print(f"[init] code BG{args.bgn} set{args.set_idx} Zc={args.zc} "
          f"N={par.cols} M={par.rows}", flush=True)
    print(f"[init] SNRs={snr_list}  n_frames={args.n_frames}  "
          f"max_iter={args.max_iter}", flush=True)
    print(f"[init] pop_size={ev_cfg.pop_size} gens={ev_cfg.generations} "
          f"elitism={ev_cfg.elitism} dedup={ev_cfg.dedup}", flush=True)
    print(f"[init] mode={'FROM-SCRATCH (random)' if args.from_scratch else 'OMS-SEEDED'}",
          flush=True)

    seed_m: Optional[GenomeMetrics] = None
    if seed is not None:
        seed_m = evaluate_genome_with_ber(seed, fit_cfg)
        print(f"[init] OMS seed: fit={seed_m.fitness:+.4f} "
              f"BER={seed_m.ber_per_snr}  FER={seed_m.fer_per_snr}",
              flush=True)

    # ---- ALWAYS evaluate OMS baseline through the SAME evaluator the
    # evolution uses, regardless of --from-scratch.  This is step 0 of
    # every run: a hand-coded reference under the EXACT same code,
    # SNR list, n_frames, and max_iter as evolution.  Required so the
    # baseline is fully aligned with the evolution scoring.
    from pushgp_ldpc.adapter import oms_seed_genome  # local import to avoid surprise on legacy path
    oms_baseline = oms_seed_genome()
    baseline_m = evaluate_genome_with_ber(oms_baseline, fit_cfg)
    print(f"[baseline] OMS@evo-cfg: fit={baseline_m.fitness:+.4f} "
          f"BER={baseline_m.ber_per_snr}  FER={baseline_m.fer_per_snr}  "
          f"valid={baseline_m.valid}", flush=True)
    baseline_record = {
        "kind": "oms_baseline",
        "fit_cfg": {
            "snr_list_db": list(snr_list),
            "n_frames_per_snr": args.n_frames,
            "max_iter": args.max_iter,
            "code_rate": 0.5,
            "code": {"bgn": args.bgn, "set_idx": args.set_idx, "zc": args.zc,
                     "N": int(par.cols), "M": int(par.rows)},
        },
        "fitness": baseline_m.fitness,
        "ber_per_snr": list(baseline_m.ber_per_snr),
        "fer_per_snr": list(baseline_m.fer_per_snr),
        "n_frames_per_snr": baseline_m.n_frames_per_snr,
        "valid": baseline_m.valid,
        "error": baseline_m.error,
    }
    (out_dir / "baseline.json").write_text(
        json.dumps(baseline_record, indent=2), encoding="utf-8")

    logger = GenerationLogger(out_dir, fit_cfg)

    t0 = time.time()
    if args.from_scratch:
        # Two-pop evolve_from_scratch: fitness fn evaluates a Genome triple.
        res = evolve_from_scratch(
            logger.fitness_fn, ev_cfg,
            workers=args.workers,
            on_generation=logger.on_generation_two_pop,
        )
    else:
        # Legacy single-pop evolve with init-eval progress.
        n_seen = [0]

        def fitness_fn_progress(g):
            n_seen[0] += 1
            f = logger.fitness_fn(g)
            if n_seen[0] <= ev_cfg.pop_size:
                print(f"[init-eval {n_seen[0]:>3}/{ev_cfg.pop_size}] fit={f:+.4f}",
                      flush=True)
            return f

        res = evolve(
            fitness_fn_progress, [seed], ev_cfg,
            on_generation=logger.on_generation_single,
        )
    t_total = time.time() - t0

    res.best_genome.save(out_dir / "champion.json")
    champ_m = evaluate_genome_with_ber(res.best_genome, fit_cfg)
    summary = {
        "elapsed_s": t_total,
        "from_scratch": bool(args.from_scratch),
        "seed_fitness": (seed_m.fitness if seed_m is not None else None),
        "seed_ber_per_snr": (list(seed_m.ber_per_snr) if seed_m is not None else None),
        "seed_fer_per_snr": (list(seed_m.fer_per_snr) if seed_m is not None else None),
        "champion_fitness": float(res.best_fitness),
        "champion_ber_per_snr": list(champ_m.ber_per_snr),
        "champion_fer_per_snr": list(champ_m.fer_per_snr),
        "champion_v2c_size": tree_size(res.best_genome.prog_v2c),
        "champion_c2v_size": tree_size(res.best_genome.prog_c2v),
        "champion_v2c_max_depth": tree_max_depth(res.best_genome.prog_v2c),
        "champion_c2v_max_depth": tree_max_depth(res.best_genome.prog_c2v),
    }
    (out_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] {t_total:.1f}s  champion_fit={res.best_fitness:+.4f}  "
          f"BER={champ_m.ber_per_snr}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
