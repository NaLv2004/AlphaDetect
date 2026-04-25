"""R7: post-run inspection of slot-evolution mutation activity.

Reads a training_log.jsonl produced by train_gnn.py and reports:
  - Per-generation aggregate slot-evo counters
  - Per-operator activity (attempted, accepted, evaluated, improved,
    behavior-noop'd) summed across all generations

This script answers the question: did the new typed-GP slot-evolution
framework (R2-R6) actually propose meaningful mutations, and did any of
them actually improve a slot's SER?

Usage:
    python code_review/r7_inspect_mutations.py results/gnn_training/training_log.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def _coerce_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def main(log_path: Path) -> int:
    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}")
        return 2

    n_gens = 0
    total = defaultdict(int)
    per_op_total: dict[str, dict[str, int]] = {}
    per_gen_rows: list[tuple[int, int, int, int, int]] = []

    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_gens += 1
            eng = rec.get("engine_stats") or {}
            evo = eng.get("slot_evo") or {}
            for k, v in evo.items():
                if isinstance(v, (int, float)) and k != "best_delta_sum":
                    total[k] += _coerce_int(v) if isinstance(v, int) else int(v)
                elif k == "best_delta_sum":
                    total[k] = total.get(k, 0.0) + float(v)
            per_op = evo.get("per_operator") or {}
            for op_name, bucket in per_op.items():
                sink = per_op_total.setdefault(op_name, {})
                for kk, vv in (bucket or {}).items():
                    if isinstance(vv, (int, float)):
                        sink[kk] = sink.get(kk, 0) + _coerce_int(vv)
            per_gen_rows.append((
                int(rec.get("gen", -1)),
                _coerce_int(evo.get("n_attempted", 0)),
                _coerce_int(evo.get("n_evaluated", 0)),
                _coerce_int(evo.get("n_improved", 0)),
                _coerce_int(evo.get("n_noop_behavior", 0)),
            ))

    print(f"=== Slot-evolution inspection: {log_path} ===")
    print(f"Generations parsed : {n_gens}")
    if n_gens == 0:
        return 1

    print()
    print("--- Aggregate slot_evo counters (sum across all generations) ---")
    for k in (
        "n_attempted", "n_validated", "n_evaluated", "n_improved",
        "n_noop_behavior", "n_apply_failed", "n_validate_failed",
        "n_eval_failed", "skipped_no_sids", "skipped_no_variants",
    ):
        print(f"  {k:24s} = {total.get(k, 0)}")
    bd_sum = float(total.get("best_delta_sum", 0.0))
    bd_cnt = int(total.get("best_delta_count", 0))
    if bd_cnt:
        print(f"  best_delta_mean          = {bd_sum / bd_cnt:.4f}  (n={bd_cnt})")
    else:
        print(f"  best_delta_mean          = n/a  (no improvements)")

    print()
    print("--- Per-operator activity (sum across all generations) ---")
    if not per_op_total:
        print("  (no per-operator stats found in JSONL — older run?)")
    else:
        cols = ("n_attempted", "n_accepted_structurally",
                "n_evaluated", "n_improved", "n_noop_behavior",
                "n_proposed_none", "n_validate_rejected", "n_noop_ir",
                "n_complexity_rejected", "n_probe_rejected")
        header = f"  {'operator':30s} " + " ".join(f"{c[2:][:10]:>11s}" for c in cols)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for op_name in sorted(per_op_total.keys()):
            b = per_op_total[op_name]
            row = f"  {op_name:30s} " + " ".join(f"{b.get(c, 0):>11d}" for c in cols)
            print(row)

    print()
    print("--- Per-generation summary (gen | attempt | eval | improve | noop_beh) ---")
    for row in per_gen_rows[-20:]:
        print(f"  gen={row[0]:>3d}  attempted={row[1]:>4d}  evaluated={row[2]:>4d}  "
              f"improved={row[3]:>3d}  noop_beh={row[4]:>3d}")

    # Verdict
    print()
    n_evaluated = total.get("n_evaluated", 0)
    n_improved = total.get("n_improved", 0)
    structural_ops = {
        "mut_insert_typed", "mut_delete_typed", "mut_subtree_replace",
        "cx_subtree_typed", "mut_primitive_inject",
    }
    structural_active = sum(
        per_op_total.get(op, {}).get("n_attempted", 0)
        for op in structural_ops
    )
    structural_accepted = sum(
        per_op_total.get(op, {}).get("n_accepted_structurally", 0)
        for op in structural_ops
    )
    print("=== VERDICT ===")
    print(f"  Total micro-evaluations    : {n_evaluated}")
    print(f"  Total improvements         : {n_improved}")
    print(f"  R5 structural-op attempts  : {structural_active}")
    print(f"  R5 structural accepted     : {structural_accepted}")
    if n_evaluated > 0 and structural_active > 0:
        print("  -> Slot evolution IS active and exercising R5 operators.")
    elif n_evaluated == 0:
        print("  -> WARNING: no micro-evaluations occurred.")
    else:
        print("  -> WARNING: R5 structural operators never proposed a mutation.")
    return 0


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/gnn_training/training_log.jsonl")
    raise SystemExit(main(p))
