"""Periodic-report helper: reads `results/logged_evolution/<run>/individuals.jsonl`
and prints, per generation, every pair's fitness PLUS:
  * the **live symbolic expression** for V2C and C2V (dead code stripped),
    obtained by running each program through the provenance-tracing
    `pushgp.trace.TraceVM`.
  * the raw program text (truncated) as a sanity reference.

Usage:
    python -B experiments/report_run.py <run_name> [--gen N | --last]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make `pushgp` importable when run from arbitrary cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import numpy as np  # noqa: E402

from pushgp.serialize import dict_to_program  # noqa: E402
from pushgp.trace import trace_program  # noqa: E402


def render_instr(d: Dict[str, Any]) -> str:
    name = d.get("name", "?")
    cb = d.get("code_block")
    cb2 = d.get("code_block2")
    if cb is None and cb2 is None:
        return name
    parts = [name, "{"]
    if cb is not None:
        parts.append(" ; ".join(render_instr(c) for c in cb))
    parts.append("}")
    if cb2 is not None:
        parts.append("{")
        parts.append(" ; ".join(render_instr(c) for c in cb2))
        parts.append("}")
    return "".join(parts)


def render_program(plist: List[Dict[str, Any]], max_chars: int = 240) -> str:
    s = " ; ".join(render_instr(d) for d in plist)
    if len(s) > max_chars:
        s = s[:max_chars - 3] + "..."
    return s


def trace_one(prog_dicts: List[Dict[str, Any]],
              log_consts: List[float], *, has_llr: bool) -> Dict[str, Any]:
    """Trace a serialized program; return {value, live_expr, fault}."""
    try:
        prog = dict_to_program(prog_dicts)
        evo = np.asarray(
            [10.0 ** float(x) for x in log_consts], dtype=np.float64,
        )
        return trace_program(
            prog,
            ctx_channel_llr=0.5,
            ctx_incoming=np.array(
                [0.6, -0.4, 0.3, -0.7, 0.1, -0.2, 0.5, -0.1],
                dtype=np.float64,
            ),
            ctx_noise_var=1.0,
            ctx_iter=2,
            ctx_max_iter=8,
            ctx_deg=8,
            ctx_edge_index=3,
            ctx_evo_constants=evo,
            ctx_has_channel_llr=has_llr,
        )
    except Exception as exc:
        return {"value": None, "live_expr": None,
                "fault": f"trace error: {type(exc).__name__}: {exc}"}


def fmt_floats(xs: List[float], ndigits: int = 2) -> str:
    return "[" + ", ".join(f"{x:.{ndigits}e}" if x < 1e-2 else f"{x:.{ndigits}f}"
                            for x in xs) + "]"


def truncate(s: Optional[str], n: int) -> str:
    if s is None:
        return "<no live expr — top of float stack empty>"
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def report_gen(records: List[Dict[str, Any]], max_chars: int,
               top_k: Optional[int]) -> None:
    if not records:
        return
    gen = records[0].get("gen", "?")
    print(f"\n=== gen {gen}  ({len(records)} pairs) ===", flush=True)
    recs = sorted(records, key=lambda r: r.get("fitness", float("inf")))
    if top_k is not None:
        recs = recs[: top_k]
    for r in recs:
        f = r.get("fitness", float("nan"))
        ber = r.get("ber_per_snr", [])
        v_sz = r.get("v2c_size", 0)
        c_sz = r.get("c2v_size", 0)
        v_dp = r.get("v2c_max_depth", 0)
        c_dp = r.get("c2v_max_depth", 0)
        k = r.get("log_constants", [])
        valid = r.get("valid", False)
        err = r.get("error")

        v_trace = trace_one(r.get("v2c", []), k, has_llr=True)
        c_trace = trace_one(r.get("c2v", []), k, has_llr=False)

        print(
            f"\n--- pair v={r.get('v_idx','?'):>2} c={r.get('c_idx','?'):>2}  "
            f"fit={f:+.4f}  BER={fmt_floats(ber)}  valid={valid}",
            flush=True,
        )
        if err:
            print(f"  ERROR: {err}", flush=True)
        v_val = v_trace.get("value")
        c_val = c_trace.get("value")
        v_val_str = f"{v_val:+.4f}" if v_val is not None else "None"
        c_val_str = f"{c_val:+.4f}" if c_val is not None else "None"
        print(f"  V2C [size={v_sz} depth={v_dp}, sample_out={v_val_str}]:",
              flush=True)
        print(f"     LIVE: {truncate(v_trace.get('live_expr'), max_chars)}",
              flush=True)
        if v_trace.get("fault"):
            print(f"     trace_fault: {v_trace['fault']}", flush=True)
        print(f"     RAW : {render_program(r.get('v2c', []), max_chars=max_chars)}",
              flush=True)
        print(f"  C2V [size={c_sz} depth={c_dp}, sample_out={c_val_str}]:",
              flush=True)
        print(f"     LIVE: {truncate(c_trace.get('live_expr'), max_chars)}",
              flush=True)
        if c_trace.get("fault"):
            print(f"     trace_fault: {c_trace['fault']}", flush=True)
        print(f"     RAW : {render_program(r.get('c2v', []), max_chars=max_chars)}",
              flush=True)
        if k:
            ks = ", ".join(f"{x:+.2f}" for x in k)
            print(f"  K (log10 consts): [{ks}]", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--gen", type=int, default=None,
                    help="If set, only show this generation. Else show all.")
    ap.add_argument("--last", action="store_true",
                    help="Only show the most recent generation present.")
    ap.add_argument("--top-k", type=int, default=None,
                    help="Only render the K best individuals per gen.")
    ap.add_argument("--max-chars", type=int, default=320,
                    help="Truncate each program/expr to this length.")
    args = ap.parse_args()

    candidates = [
        Path("results") / "logged_evolution" / args.run_name / "individuals.jsonl",
        Path(__file__).resolve().parent.parent / "results" / "logged_evolution"
            / args.run_name / "individuals.jsonl",
        Path(__file__).resolve().parent.parent.parent.parent / "results"
            / "logged_evolution" / args.run_name / "individuals.jsonl",
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if path is None:
        for c in candidates:
            print(f"  tried: {c}", file=sys.stderr)
        print(f"No individuals.jsonl found for run '{args.run_name}'",
              file=sys.stderr)
        return 2

    by_gen: Dict[int, List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            g = rec.get("gen", -1)
            by_gen.setdefault(g, []).append(rec)

    if not by_gen:
        print("No records yet.", file=sys.stderr)
        return 0

    sum_path = path.parent / "gen_summary.jsonl"
    if sum_path.exists():
        with sum_path.open("r", encoding="utf-8") as fh:
            sums = [json.loads(line) for line in fh if line.strip()]
        if sums:
            print("=== generation summary so far ===", flush=True)
            for s in sums:
                print(
                    f"  gen {s['gen']:>2}  best={s['best_fit']:+.4f}  "
                    f"mean={s['mean_fit']:+.4f}  med={s['median_fit']:+.4f}  "
                    f"v_att={s.get('v_offspring_attempts','-')}/"
                    f"inv={s.get('v_offspring_invalid','-')}  "
                    f"c_att={s.get('c_offspring_attempts','-')}/"
                    f"inv={s.get('c_offspring_invalid','-')}  "
                    f"t={s['elapsed_s']:.1f}s",
                    flush=True,
                )

    if args.last:
        gens = [max(by_gen)]
    elif args.gen is not None:
        gens = [args.gen]
    else:
        gens = sorted(by_gen)
    for g in gens:
        report_gen(by_gen.get(g, []), max_chars=args.max_chars,
                   top_k=args.top_k)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

