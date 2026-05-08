"""Snapshot end-to-end pipeline for Path-B equivalence verification.

Generates a deterministic snapshot of (algo_id, trial_idx) -> x_hat for all
detectors in build_ir_pool, plus IR statistics. Used to verify that the
Path B (pruned-SSA) IR builder fix preserves detector semantics.

Usage:
  python snapshot_pipeline.py <output_npz_path> [--snr 8.0] [--trials 64]

Run BEFORE the IR builder fix to capture baseline, then AFTER to compare.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolution.ir_pool import build_ir_pool
from evolution.materialize import materialize_to_callable
from evolution.mimo_evaluator import (
    qam16_constellation,
    generate_mimo_sample,
    _nearest_symbols,
)
from algorithm_ir.region.triviality import is_trivial_op, _get_iter_trivial_cache


def ir_stats(ir) -> dict:
    """Per-IR statistics for diff."""
    n_ops = len(ir.ops)
    n_phi = sum(1 for op in ir.ops.values() if op.opcode == "phi")
    n_phi_single_trivial = sum(
        1 for op in ir.ops.values()
        if op.opcode == "phi" and len(set(op.inputs)) <= 1
    )
    try:
        trivial_ids, _ = _get_iter_trivial_cache(ir)
        n_phi_iter_trivial = sum(
            1 for op in ir.ops.values()
            if op.opcode == "phi" and op.id in trivial_ids
        )
    except Exception:
        n_phi_iter_trivial = -1
    n_nontrivial = sum(
        1 for op in ir.ops.values() if not is_trivial_op(op, ir)
    )
    return {
        "n_ops": n_ops,
        "n_phi": n_phi,
        "n_phi_single_trivial": n_phi_single_trivial,
        "n_phi_iter_trivial": n_phi_iter_trivial,
        "n_nontrivial_ops": n_nontrivial,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=str, help="Output .npz path")
    ap.add_argument("--snr", type=float, default=8.0, help="SNR in dB")
    ap.add_argument("--trials", type=int, default=64, help="Trials per detector")
    ap.add_argument("--Nr", type=int, default=16)
    ap.add_argument("--Nt", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Snapshot pipeline ===")
    print(f"Output : {out_path}")
    print(f"SNR    : {args.snr} dB")
    print(f"Trials : {args.trials}")
    print(f"Nr,Nt  : {args.Nr},{args.Nt}")
    print(f"Seed   : {args.seed}")
    print()

    constellation = qam16_constellation()
    pool = build_ir_pool(np.random.default_rng(args.seed))
    print(f"Pool size: {len(pool)} detectors")

    # Pre-generate inputs deterministically (shared across detectors).
    input_rng = np.random.default_rng(args.seed)
    test_inputs = []
    for _ in range(args.trials):
        H, x_true, y, sigma2 = generate_mimo_sample(
            args.Nr, args.Nt, constellation, args.snr, input_rng,
        )
        test_inputs.append((H, x_true, y, sigma2))

    # Snapshot data: per-detector results & stats
    results: dict[str, np.ndarray] = {}        # algo_id -> (trials, Nt) complex
    nearest: dict[str, np.ndarray] = {}        # algo_id -> (trials, Nt) complex (quantized)
    errors: dict[str, np.ndarray] = {}         # algo_id -> (trials,) bool: errored
    err_msgs: dict[str, list] = {}
    stats: dict[str, dict] = {}
    timings: dict[str, float] = {}
    materialize_failed: list[str] = []

    t_start = time.time()
    for idx, genome in enumerate(pool, 1):
        algo_id = genome.algo_id
        try:
            stats[algo_id] = ir_stats(genome.ir)
        except Exception as e:
            stats[algo_id] = {"error": str(e)}

        try:
            fn = materialize_to_callable(genome)
        except Exception as e:
            materialize_failed.append(algo_id)
            err_msgs.setdefault(algo_id, []).append(f"materialize: {e}")
            print(f"  [{idx:3d}/{len(pool)}] {algo_id:30s} MATERIALIZE FAIL: {e}")
            continue

        x_hats = np.zeros((args.trials, args.Nt), dtype=complex)
        x_quant = np.zeros((args.trials, args.Nt), dtype=complex)
        errs = np.zeros(args.trials, dtype=bool)
        t0 = time.time()
        for ti, (H, x_true, y, sigma2) in enumerate(test_inputs):
            try:
                x_hat = fn(H, y, sigma2, constellation)
                x_hat = np.asarray(x_hat, dtype=complex).reshape(args.Nt)
                x_hats[ti] = x_hat
                x_quant[ti] = _nearest_symbols(x_hat, constellation)
            except Exception as e:
                errs[ti] = True
                err_msgs.setdefault(algo_id, []).append(f"trial{ti}: {type(e).__name__}: {e}")
        timings[algo_id] = time.time() - t0
        results[algo_id] = x_hats
        nearest[algo_id] = x_quant
        errors[algo_id] = errs
        n_err = int(errs.sum())
        print(
            f"  [{idx:3d}/{len(pool)}] {algo_id:30s} "
            f"ops={stats[algo_id].get('n_ops','?'):4} phi={stats[algo_id].get('n_phi','?'):3} "
            f"iter_triv={stats[algo_id].get('n_phi_iter_trivial','?'):3} "
            f"errs={n_err:3} t={timings[algo_id]:.2f}s"
        )
    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Materialize failures: {len(materialize_failed)} → {materialize_failed[:5]}{'...' if len(materialize_failed)>5 else ''}")

    # Pack inputs (so post-fix run uses identical inputs)
    H_arr = np.stack([t[0] for t in test_inputs])
    x_true_arr = np.stack([t[1] for t in test_inputs])
    y_arr = np.stack([t[2] for t in test_inputs])
    sigma2_arr = np.array([t[3] for t in test_inputs])

    # Save
    save_dict = {
        "_meta_snr": args.snr,
        "_meta_trials": args.trials,
        "_meta_Nr": args.Nr,
        "_meta_Nt": args.Nt,
        "_meta_seed": args.seed,
        "_meta_pool_size": len(pool),
        "_meta_constellation": constellation,
        "_inputs_H": H_arr,
        "_inputs_x_true": x_true_arr,
        "_inputs_y": y_arr,
        "_inputs_sigma2": sigma2_arr,
        "_algo_ids": np.array([g.algo_id for g in pool]),
        "_materialize_failed": np.array(materialize_failed),
    }
    for algo_id, arr in results.items():
        save_dict[f"xhat__{algo_id}"] = arr
    for algo_id, arr in nearest.items():
        save_dict[f"xquant__{algo_id}"] = arr
    for algo_id, arr in errors.items():
        save_dict[f"errs__{algo_id}"] = arr
    for algo_id, st in stats.items():
        for k, v in st.items():
            save_dict[f"stat__{algo_id}__{k}"] = np.array(v)
    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved to: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    # Print error message summary
    if err_msgs:
        print(f"\n=== Error message summary ({len(err_msgs)} detectors had errors) ===")
        for algo_id, msgs in list(err_msgs.items())[:10]:
            print(f"  {algo_id}: {len(msgs)} errors, first: {msgs[0][:120]}")


if __name__ == "__main__":
    main()
