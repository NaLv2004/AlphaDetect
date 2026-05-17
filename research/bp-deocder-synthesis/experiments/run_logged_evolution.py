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
from pushgp.op_filter import load_op_filter
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
    # Code + fitness eval — user gives only (A, E); everything else is
    # derived per the NR LDPC standard via ldpc_5g.derive_params.
    p.add_argument("--info-len-A", type=int, default=176,
                   help="K-info bits per CB (TS 38.212 A).")
    p.add_argument("--code-length-E", type=int, default=352,
                   help="Rate-matched output length per CB (TS 38.212 E).")
    p.add_argument("--snr-list", type=str, default="2,3,4",
                   help="Comma-sep SNRs in dB.")
    p.add_argument("--n-frames", type=int, default=6)
    p.add_argument("--max-iter", type=int, default=8)
    # DCE oracle code (independent of training code; default = small
    # BG2-set1-Zc=2 N=104 lifted code for fast DCE BP probes).
    p.add_argument("--dce-bgn", type=int, default=2,
                   help="DCE oracle base graph number (1 or 2).")
    p.add_argument("--dce-set-idx", type=int, default=1,
                   help="DCE oracle Zc-set index (1..8).")
    p.add_argument("--dce-zc", type=int, default=2,
                   help="DCE oracle lifting size (default 2 -> N=104).")
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
    p.add_argument("--cpp-seeder", dest="cpp_seeder", action="store_true",
                   default=False,
                   help="Use C++ pybind11 seeder (pushgp_cpp_seeder) for "
                        "initial random fill of pop_v / pop_c. Bit-identical "
                        "VM/validator semantics; ~7x faster than Python.")
    # ---- BP-equivalence DCE (post-seed + per-gen post-offspring) ----
    p.add_argument("--dce-bp", dest="dce_bp", action="store_true",
                   default=False,
                   help="Enable BP-equivalence DCE on populations after "
                        "initial seeding and after each generation's "
                        "offspring fill. Uses cpp acceleration by default.")
    p.add_argument("--dce-bp-max-iter", type=int, default=8,
                   help="BP max_iter used by DCE equivalence check.")
    p.add_argument("--dce-bp-decimals", type=int, default=6,
                   help="Rounding decimals for post-LLR equivalence.")
    p.add_argument("--dce-bp-max-passes", type=int, default=800,
                   help="Hard cap on DCE inner reduction passes.")
    p.add_argument("--dce-bp-max-decode-evals", type=int, default=-1,
                   help="Hard cap on BP decode invocations per program "
                        "(<0 = unlimited).")
    p.add_argument("--dce-bp-threads", type=int, default=0,
                   help="Worker threads for cpp DCE batch (0 = use --workers).")
    p.add_argument("--dce-bp-no-cpp", dest="dce_bp_use_cpp",
                   action="store_false", default=True,
                   help="Force pure-Python DCE (default cpp ON).")
    p.add_argument("--dce-bp-snr-db", type=float, default=None,
                   help="SNR (dB) used to build the DCE oracle's rx_llrs. "
                        "Defaults to the median of --snr-list.")
    p.add_argument("--dce-bp-n-frames", type=int, default=1,
                   help="Number of channel frames per oracle (default 1).")
    p.add_argument("--dce-bp-oracle-seed", type=int, default=20000,
                   help="RNG seed for the DCE oracle channel realization.")
    # ---- Pair-binding (co-adaptation) ----
    p.add_argument("--bind-pairs", dest="bind_pairs", action="store_true",
                   default=True,
                   help="Bind (V2C, C2V, K) triples from gen 0: identity "
                        "pairing, single triple tournament, atomic triple "
                        "crossover. Default ON. Solves the random-pairing "
                        "co-adaptation problem of the legacy two-pop CCEA.")
    p.add_argument("--no-bind-pairs", dest="bind_pairs", action="store_false",
                   help="Restore legacy random-permutation pairing every "
                        "generation (per-side independent offspring).")
    # ---- C++-accelerated fitness evaluation ----
    p.add_argument("--cpp-fitness", dest="cpp_fitness", action="store_true",
                   default=True,
                   help="Route every BP decode in fitness eval through the "
                        "C++ kernel `pushgp_cpp_dce.decode_bp` (default ON, "
                        "10-50x speedup; byte-locked to the Python path to "
                        "6 decimals by cpp_dce/tests/test_bp_equivalence.py).")
    p.add_argument("--no-cpp-fitness", dest="cpp_fitness", action="store_false",
                   help="Use the legacy pure-Python fitness eval loop "
                        "(useful for A/B equivalence testing).")
    # ---- Operator filter (opcode whitelist) -----------------------
    p.add_argument("--op-config", type=str, default=None,
                   help="Path to a JSON op-filter config restricting the "
                        "opcodes evolution may sample/mutate to. See "
                        "pushgp/op_filter.py docstring and "
                        "configs/op_filter_oms.json for format. When "
                        "set, the C++ seeder is forced off (the cpp "
                        "seeder does not yet support opcode filtering).")
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
        # Optional ParallelPairEvaluator: if set, on_generation_two_pop
        # will pull metrics from `parallel_eval.last_metrics` instead of
        # re-running evaluate_genome_with_ber for each individual.
        self.parallel_eval = None

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
                # Prefer pre-computed metrics from the parallel evaluator.
                pe = self.parallel_eval
                if pe is not None and i < len(pe.last_metrics):
                    m = pe.last_metrics[i]
                else:
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

    # Build code + fitness cfg.  Everything (bgn, set_idx, Zc, K, N,
    # K_cb_bit, ...) is derived from (A, E) per the NR LDPC standard.
    fit_cfg = FitnessConfig(
        info_len_A=args.info_len_A,
        code_length_E=args.code_length_E,
        snr_list=snr_list,
        n_frames_per_snr=args.n_frames,
        max_iter=args.max_iter,
        use_cpp_fitness=bool(args.cpp_fitness),
    )
    par = fit_cfg.par
    p_der = fit_cfg.derived

    # Seed (only used if --use-oms-seed).
    if not args.from_scratch:
        save_oms_seed()
        seed = load_oms_seed()
    else:
        seed = None

    # Operator filter (opcode whitelist).  Loaded once here so we can
    # (a) print a summary up front and (b) force-disable cpp_seeder
    # when active (cpp seeder ignores opcode filtering).
    op_filter = load_op_filter(args.op_config)
    if op_filter.applies():
        print(op_filter.describe(), flush=True)
        if args.cpp_seeder:
            print("[op-filter] WARNING: --cpp-seeder is incompatible with "
                  "--op-config; forcing cpp_seeder=OFF (Python multiprocessing "
                  "seeder will be used and respects the filter).", flush=True)
            args.cpp_seeder = False

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
        cpp_seeder=args.cpp_seeder,
        dce_bp_enabled=args.dce_bp,
        dce_bp_max_iter=args.dce_bp_max_iter,
        dce_bp_decimals=args.dce_bp_decimals,
        dce_bp_max_passes=args.dce_bp_max_passes,
        dce_bp_max_decode_evals=args.dce_bp_max_decode_evals,
        dce_bp_threads=args.dce_bp_threads,
        dce_bp_use_cpp=args.dce_bp_use_cpp,
        bind_pairs=args.bind_pairs,
    )

    meta = {
        "run_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "from_scratch": bool(args.from_scratch),
        "code": {"info_len_A": args.info_len_A,
                 "code_length_E": args.code_length_E,
                 "bgn": p_der.bgn, "set_idx": p_der.set_idx,
                 "zc": p_der.zc, "N": int(par.cols), "M": int(par.rows),
                 "K_cb_bit": p_der.K_cb_bit, "K": p_der.K,
                 "N_punctured": p_der.N_punctured,
                 "effective_code_rate": fit_cfg.effective_code_rate},
        "snr_list_db": list(snr_list),
        "n_frames_per_snr": args.n_frames,
        "max_iter": args.max_iter,
        "op_filter": op_filter.to_dict(),
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
            "cpp_seeder": ev_cfg.cpp_seeder,
            "dce_bp_enabled": ev_cfg.dce_bp_enabled,
            "dce_bp_max_iter": ev_cfg.dce_bp_max_iter,
            "dce_bp_decimals": ev_cfg.dce_bp_decimals,
            "dce_bp_max_passes": ev_cfg.dce_bp_max_passes,
            "dce_bp_max_decode_evals": ev_cfg.dce_bp_max_decode_evals,
            "dce_bp_threads": ev_cfg.dce_bp_threads,
            "dce_bp_use_cpp": ev_cfg.dce_bp_use_cpp,
            "dce_bp_snr_db": (args.dce_bp_snr_db
                              if args.dce_bp_snr_db is not None else None),
            "dce_bp_n_frames": args.dce_bp_n_frames,
            "dce_bp_oracle_seed": args.dce_bp_oracle_seed,
            "bind_pairs": bool(ev_cfg.bind_pairs),
            "use_cpp_fitness": bool(fit_cfg.use_cpp_fitness),
        },
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[init] run dir: {out_dir}", flush=True)
    print(f"[init] A={args.info_len_A} E={args.code_length_E} "
          f"R={fit_cfg.effective_code_rate:.4f} -> BG{p_der.bgn} set{p_der.set_idx} "
          f"Zc={p_der.zc} N={par.cols} M={par.rows} K_cb_bit={p_der.K_cb_bit}",
          flush=True)
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
    from pushgp_ldpc.baselines import (
        uncoded_rate1_baseline, channel_hard_baseline,
    )
    oms_baseline = oms_seed_genome()
    baseline_m = evaluate_genome_with_ber(oms_baseline, fit_cfg)
    print(f"[baseline] OMS@evo-cfg: fit={baseline_m.fitness:+.4f} "
          f"BER={baseline_m.ber_per_snr}  FER={baseline_m.fer_per_snr}  "
          f"valid={baseline_m.valid}", flush=True)
    # --- Uncoded references (always computed, regardless of seed mode).
    unc_r1 = uncoded_rate1_baseline(fit_cfg)
    unc_ch = channel_hard_baseline(fit_cfg)
    print(f"[baseline] uncoded R=1 hard:           BER={unc_r1['ber_per_snr']}",
          flush=True)
    print(f"[baseline] channel-hard same-pipeline: BER={unc_ch['ber_per_snr']}  "
          f"(R_used={unc_ch['code_rate_used']:.4f})", flush=True)
    baseline_record = {
        "kind": "oms_baseline",
        "fit_cfg": {
            "snr_list_db": list(snr_list),
            "n_frames_per_snr": args.n_frames,
            "max_iter": args.max_iter,
            "code_rate": fit_cfg.effective_code_rate,
            "code": {"info_len_A": args.info_len_A,
                     "code_length_E": args.code_length_E,
                     "bgn": p_der.bgn, "set_idx": p_der.set_idx,
                     "zc": p_der.zc, "N": int(par.cols), "M": int(par.rows),
                     "K_cb_bit": p_der.K_cb_bit},
        },
        "fitness": baseline_m.fitness,
        "ber_per_snr": list(baseline_m.ber_per_snr),
        "fer_per_snr": list(baseline_m.fer_per_snr),
        "n_frames_per_snr": baseline_m.n_frames_per_snr,
        "valid": baseline_m.valid,
        "error": baseline_m.error,
        "uncoded": {
            "rate1_hard": unc_r1,
            "channel_hard_same_pipeline": unc_ch,
        },
    }
    (out_dir / "baseline.json").write_text(
        json.dumps(baseline_record, indent=2), encoding="utf-8")

    logger = GenerationLogger(out_dir, fit_cfg)

    t0 = time.time()
    # ---- Build DCE oracle (par + rx_llrs) if requested ----------------
    # The DCE oracle uses its OWN (small) lifted code, independent of the
    # training code (default BG2 set1 Zc=2 -> N=104).  Behaviour-only:
    # we just need a valid all-zero codeword + AWGN-perturbed LLR vector.
    dce_oracle = None
    if args.dce_bp:
        snr_sorted = sorted(snr_list)
        snr_pick = (args.dce_bp_snr_db if args.dce_bp_snr_db is not None
                    else float(snr_sorted[len(snr_sorted) // 2]))
        from ldpc_5g import (HTYPE, bpsk_modulate, bpsk_llr,
                             encode_codeblock, build_parity as _bp)
        dce_par = _bp(bgn=args.dce_bgn, set_idx=args.dce_set_idx, zc=args.dce_zc)
        htype_dce = HTYPE[dce_par.bgn - 1][dce_par.set_idx - 1]
        Kb_dce = 10 if dce_par.bgn == 2 else 22
        K_dce = Kb_dce * dce_par.zc
        N_dce = dce_par.cols
        # DCE noise: use physical (un-rate-matched) base-graph rate so
        # σ² is well-defined and reproducible across configs.
        rate_dce = float(K_dce) / float(N_dce - 2 * dce_par.zc)
        sigma2 = 1.0 / (2.0 * rate_dce * 10.0 ** (snr_pick / 10.0))
        sigma = float(np.sqrt(sigma2))
        rx_llrs: List[np.ndarray] = []
        for f_idx in range(int(max(1, args.dce_bp_n_frames))):
            rng = np.random.default_rng(args.dce_bp_oracle_seed + f_idx)
            info = rng.integers(0, 2, size=K_dce, dtype=np.int8).astype(np.int64)
            cw_punct = encode_codeblock(info, dce_par, htype_dce)
            cw_full = np.concatenate(
                [info[: 2 * dce_par.zc].astype(np.int8), cw_punct]).astype(np.int8)
            tx = bpsk_modulate(cw_full[2 * dce_par.zc:])
            rx = tx + sigma * rng.standard_normal(tx.shape)
            llr_part = bpsk_llr(rx, sigma2)
            llr = np.zeros(N_dce, dtype=np.float64)
            llr[2 * dce_par.zc:] = llr_part
            rx_llrs.append(llr)
        dce_oracle = {"par": dce_par, "rx_llrs": rx_llrs}
        print(f"[init] DCE oracle: BG{dce_par.bgn} set{dce_par.set_idx} "
              f"Zc={dce_par.zc} N={N_dce}  snr={snr_pick:+.1f}dB  "
              f"frames={len(rx_llrs)}  use_cpp={args.dce_bp_use_cpp}  "
              f"max_iter={args.dce_bp_max_iter}  decimals={args.dce_bp_decimals}",
              flush=True)

    if args.from_scratch:
        # Two-pop evolve_from_scratch with parallel pair-fitness eval.
        from pushgp_ldpc.parallel_eval import ParallelPairEvaluator
        eval_workers = max(1, args.workers)
        print(f"[init] starting ParallelPairEvaluator with {eval_workers} workers",
              flush=True)
        pev = ParallelPairEvaluator(fit_cfg, eval_workers)
        logger.parallel_eval = pev
        try:
            res = evolve_from_scratch(
                logger.fitness_fn, ev_cfg,
                workers=args.workers,
                batch_eval_fn=pev.eval_pairs,
                on_generation=logger.on_generation_two_pop,
                dce_oracle=dce_oracle,
                op_filter=op_filter,
            )
        finally:
            pev.close()
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
