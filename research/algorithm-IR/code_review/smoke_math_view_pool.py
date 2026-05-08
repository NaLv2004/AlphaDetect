"""Phase 2 smoke: build MathView for ALL 91 detectors in the pool.

Goals:
  (1) 100% coverage (no missing/un-mapped SSA op in any detector).
  (2) No build crashes.
  (3) Per-detector compression report (n_ssa_ops -> n_ssa_op_nodes,
      ratio, kind histogram).

This does NOT validate runtime equivalence; the underlying FunctionIR is
unchanged so Python codegen path is by-construction equivalent.
"""
from __future__ import annotations

import os
import sys
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from algorithm_ir.ir.math_view import build_math_view, compression_stats  # noqa: E402
from evolution.ir_pool import build_ir_pool  # noqa: E402


def _kind_hist(view) -> dict[str, int]:
    return dict(Counter(n.kind for n in view.nodes))


def main() -> int:
    rng = np.random.default_rng(42)
    pool = build_ir_pool(rng)
    print(f"=== Pool size: {len(pool)} detectors")

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    coverage_failures: list[tuple[str, dict]] = []

    for g in pool:
        algo_id = g.algo_id
        ir = g.ir
        n_ops = len(ir.ops)
        n_blocks = len(ir.blocks)
        try:
            view = build_math_view(ir)
        except Exception as exc:  # noqa: BLE001
            failures.append((algo_id, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
            continue
        rep = view.coverage_report()
        if rep["n_missing"] > 0:
            coverage_failures.append((algo_id, rep))
        stats = compression_stats(view)
        rows.append({
            "algo": algo_id,
            "n_ops": n_ops,
            "n_blocks": n_blocks,
            "n_args": stats["n_args"],
            "n_ssa_nodes": stats["n_ssa_op_nodes"],
            "n_total_nodes": stats["n_total_nodes"],
            "n_absorbed": stats["n_absorbed"],
            "n_dropped": stats["n_dropped"],
            "compression": stats["compression_ssa"],
            "kind_hist": _kind_hist(view),
        })

    # ---- Aggregate report -------------------------------------------------
    n_total = len(pool)
    n_ok = len(rows)
    n_fail = len(failures)
    n_cov_bad = len(coverage_failures)

    print(f"\n=== Build results: ok={n_ok}/{n_total}  build_fail={n_fail}  coverage_fail={n_cov_bad}")

    if failures:
        print("\n--- Build failures ---")
        for algo, msg in failures:
            print(f"  {algo:40s} {msg}")
    if coverage_failures:
        print("\n--- Coverage failures ---")
        for algo, rep in coverage_failures:
            miss = rep["missing_op_ids"][:8]
            print(f"  {algo:40s} n_missing={rep['n_missing']} sample={miss}")

    # ---- Per-detector compression table ----------------------------------
    rows.sort(key=lambda r: r["compression"])
    print("\n=== Per-detector compression (sorted by compression ratio) ===")
    print(f"  {'algo':<30s} {'n_ops':>5s} {'n_blk':>5s} {'args':>4s} "
          f"{'ssa_nd':>6s} {'tot_nd':>6s} {'absorb':>6s} {'drop':>4s} "
          f"{'comp':>5s}  kind_hist")
    for r in rows:
        kh = ",".join(f"{k}={v}" for k, v in sorted(r["kind_hist"].items()))
        print(f"  {r['algo']:<30s} {r['n_ops']:>5d} {r['n_blocks']:>5d} "
              f"{r['n_args']:>4d} {r['n_ssa_nodes']:>6d} {r['n_total_nodes']:>6d} "
              f"{r['n_absorbed']:>6d} {r['n_dropped']:>4d} "
              f"{r['compression']:>5.2f}  {kh}")

    # ---- Aggregate stats --------------------------------------------------
    if rows:
        tot_ops = sum(r["n_ops"] for r in rows)
        tot_ssa_nodes = sum(r["n_ssa_nodes"] for r in rows)
        tot_total_nodes = sum(r["n_total_nodes"] for r in rows)
        tot_absorbed = sum(r["n_absorbed"] for r in rows)
        tot_dropped = sum(r["n_dropped"] for r in rows)
        avg_comp = sum(r["compression"] for r in rows) / len(rows)
        print(f"\n=== Aggregate (over {len(rows)} detectors) ===")
        print(f"  total ops          = {tot_ops}")
        print(f"  total ssa nodes    = {tot_ssa_nodes}  ({tot_ssa_nodes/tot_ops:.2%} of ops)")
        print(f"  total nodes (+args)= {tot_total_nodes}")
        print(f"  total absorbed     = {tot_absorbed}")
        print(f"  total dropped      = {tot_dropped}")
        print(f"  mean compression   = {avg_comp:.3f}")

        kind_total: Counter = Counter()
        for r in rows:
            kind_total.update(r["kind_hist"])
        print(f"  kind hist (sum)    = {dict(kind_total)}")

    # ---- Verdict ----------------------------------------------------------
    has_other = any("other" in r["kind_hist"] for r in rows)
    if has_other:
        offenders = [(r["algo"], r["kind_hist"].get("other", 0)) for r in rows
                     if r["kind_hist"].get("other", 0) > 0]
        print(f"\nERROR: {len(offenders)} detectors have 'other'-kind nodes: {offenders[:5]}")
    ok = (n_fail == 0 and n_cov_bad == 0 and n_ok == n_total and not has_other)
    print(f"\n=== Phase 2 smoke: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
