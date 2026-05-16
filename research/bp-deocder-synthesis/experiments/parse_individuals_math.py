"""Parse individuals.jsonl from a logged-evolution run and emit each
individual's symbolic math formula for V2C and C2V.

Reuses the existing math parser (`experiments/poll_report.py`:
`_tokenize` / `_parse` / `_to_math` / `interpret`) and the live-trace
producer (`pushgp/trace.py::trace_program`).

For every individual record we:
  1. Reconstruct the program from its dict form.
  2. Run `trace_program` with canonical context (one fixed neighbor
     vector + channel LLR) to obtain the dead-code-stripped LIVE
     expression on the float stack at program end.
  3. Re-parse that expression and rewrite it to compact math notation
     (FVec.PopBack -> m_i, L_v -> Lv, Float.EvoConstK -> Kk(value), etc).

Outputs (under <run_dir>/parsed_math/):
  * `parsed_math.jsonl`  — one record per individual
        {gen, idx, fitness, ber_per_snr, fer_per_snr, valid,
         v2c_size, c2v_size, log_constants,
         v2c_live_expr, v2c_math, v2c_notes,
         c2v_live_expr, c2v_math, c2v_notes,
         v2c_fault, c2v_fault}
  * `parsed_math_top.txt` — human-readable top-K per generation (default
                            K=5), sorted by fitness ascending.

Usage:
    # one-shot dump
    python -B experiments/parse_individuals_math.py <run_name>
        [--top-k 5] [--max-chars 500] [--all]

    # follow-mode: tail individuals.jsonl forever, parse every new line,
    # append both JSON + human-readable text to live log files.
    python -B experiments/parse_individuals_math.py <run_name>
        --watch [--poll-interval 2.0] [--max-chars 500]

If `--all` is set, every individual is dumped to the .txt summary;
otherwise only the top-K per generation are.

Both `--records` (i.e. logged individuals) and `--eval` targets are the
**post-DCE** programs / fitness / BER, because `evolve_from_scratch`
applies DCE BEFORE the per-gen `batch_eval_fn` and BEFORE the
`on_generation` callback that writes `individuals.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import numpy as np  # noqa: E402

from pushgp.serialize import dict_to_program  # noqa: E402
from pushgp.trace import trace_program  # noqa: E402

# Re-use the math parser from poll_report (no duplication).
from experiments.poll_report import (  # noqa: E402
    _tokenize, _parse, _to_math, interpret,
)


# --------------------------------------------------------------------------- #
# Locate run dir + IO                                                          #
# --------------------------------------------------------------------------- #
def find_run_dir(run_name: str) -> Path:
    candidates = [
        Path("results") / "logged_evolution" / run_name,
        Path(_PROJ) / "results" / "logged_evolution" / run_name,
        Path(_PROJ).parent.parent / "results" / "logged_evolution" / run_name,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise SystemExit(
        "run dir not found. tried:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# --------------------------------------------------------------------------- #
# Trace + math conversion for a single program side                            #
# --------------------------------------------------------------------------- #
# Canonical context used by `experiments/poll_report.py::trace_one`.  Keeping
# the same fixed neighbor vector / channel LLR / iter / deg means the LIVE
# expressions parsed here are directly comparable with poll_report output.
_CTX_INCOMING = np.array([0.6, -0.4, 0.3, -0.7, 0.1, -0.2, 0.5], dtype=np.float64)


def _trace_side(prog_dicts: List[Dict[str, Any]],
                log_consts: List[float], *, has_llr: bool) -> Dict[str, Any]:
    try:
        prog = dict_to_program(prog_dicts)
        evo = np.asarray([10.0 ** float(x) for x in log_consts],
                         dtype=np.float64)
        return trace_program(
            prog,
            ctx_channel_llr=0.5,
            ctx_incoming=_CTX_INCOMING,
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
                "fault": f"{type(exc).__name__}: {exc}"}


def parse_side(prog_dicts: List[Dict[str, Any]],
               log_consts: List[float], *, has_llr: bool
               ) -> Tuple[Optional[str], str, str, Optional[str]]:
    """Return ``(live_expr, math_expr, interp_with_notes, fault)``."""
    tr = _trace_side(prog_dicts, log_consts, has_llr=has_llr)
    fault = tr.get("fault")
    expr = tr.get("live_expr")
    if not expr:
        return expr, "<empty>", "<empty>", fault
    try:
        toks = _tokenize(expr)
        node, _ = _parse(toks, 0)
        math_only = _to_math(node, evo_consts=[10.0 ** float(x)
                                               for x in log_consts])
    except Exception as e:  # noqa: BLE001
        math_only = f"<parse error: {e}>"
    interp = interpret(expr, evo_consts=[10.0 ** float(x)
                                         for x in log_consts])
    return expr, math_only, interp, fault


def _format_rec_text(rec: Dict[str, Any], max_chars: int) -> str:
    """Format one parsed record as the multi-line block used by the txt
    summary AND the live watch log.  Kept as a single helper so both
    one-shot and watch mode emit identical output."""
    def _trim(s: Optional[str], n: int) -> str:
        if s is None:
            return "<none>"
        return s if len(s) <= n else s[: n - 3] + "..."

    ber = rec.get("ber_per_snr") or []
    ber_s = ("[" + ", ".join(f"{b:.4f}" for b in ber) + "]"
             if ber else "[]")
    lines = [
        f"-- gen={int(rec['gen']):03d} idx={int(rec['idx']):03d}  "
        f"fit={float(rec['fitness']):+.4f}  valid={rec['valid']}  "
        f"v2c_size={rec['v2c_size']} c2v_size={rec['c2v_size']}  "
        f"BER={ber_s}",
        f"   K = {rec['log_constants']}",
        f"   V2C math : {_trim(rec['v2c_math'], max_chars)}",
        f"   V2C notes: {_trim(rec['v2c_notes'], max_chars)}",
    ]
    if rec.get("v2c_fault"):
        lines.append(f"   V2C FAULT: {rec['v2c_fault']}")
    lines.extend([
        f"   C2V math : {_trim(rec['c2v_math'], max_chars)}",
        f"   C2V notes: {_trim(rec['c2v_notes'], max_chars)}",
    ])
    if rec.get("c2v_fault"):
        lines.append(f"   C2V FAULT: {rec['c2v_fault']}")
    return "\n".join(lines) + "\n"


def _parse_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Run the V2C + C2V trace/parse on one raw individuals.jsonl record
    and return the augmented record (same shape as one-shot output)."""
    kvec = list(rec.get("log_constants", []))
    v_dicts = rec.get("v2c", [])
    c_dicts = rec.get("c2v", [])
    v_expr, v_math, v_interp, v_fault = parse_side(
        v_dicts, kvec, has_llr=True
    )
    c_expr, c_math, c_interp, c_fault = parse_side(
        c_dicts, kvec, has_llr=False
    )
    return {
        "gen": int(rec.get("gen", -1)),
        "idx": int(rec.get("idx", -1)),
        "fitness": float(rec.get("fitness", float("inf"))),
        "ber_per_snr": rec.get("ber_per_snr", []),
        "fer_per_snr": rec.get("fer_per_snr", []),
        "valid": bool(rec.get("valid", False)),
        "v2c_size": rec.get("v2c_size"),
        "c2v_size": rec.get("c2v_size"),
        "log_constants": kvec,
        "v2c_live_expr": v_expr,
        "v2c_math": v_math,
        "v2c_notes": v_interp,
        "v2c_fault": v_fault,
        "c2v_live_expr": c_expr,
        "c2v_math": c_math,
        "c2v_notes": c_interp,
        "c2v_fault": c_fault,
    }


def watch_loop(ind_path: Path, out_dir: Path, *, poll_interval: float,
               max_chars: int) -> int:
    """Tail ``ind_path`` forever; parse every new JSON line; append to
    ``parsed_math_live.jsonl`` (full) and ``parsed_math_live.log``
    (human-readable).  Never overwrites — both files are open in append
    mode, so re-runs of `--watch` simply continue where the previous
    invocation left off.
    """
    live_jsonl = out_dir / "parsed_math_live.jsonl"
    live_log = out_dir / "parsed_math_live.log"

    # Persist file-offset cursor so multiple --watch runs don't re-parse
    # already-processed lines.  Stored next to live files.
    cursor_path = out_dir / ".parsed_math_live.cursor"
    if cursor_path.exists():
        try:
            offset = int(cursor_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            offset = 0
    else:
        offset = 0

    print(f"[watch] tailing {ind_path}", flush=True)
    print(f"[watch] live jsonl -> {live_jsonl}", flush=True)
    print(f"[watch] live log   -> {live_log}", flush=True)
    print(f"[watch] resume offset={offset} bytes  "
          f"poll_interval={poll_interval}s  max_chars={max_chars}", flush=True)

    # Header banner in the human log (only if it's a fresh file).
    if not live_log.exists() or live_log.stat().st_size == 0:
        with live_log.open("w", encoding="utf-8") as fh:
            fh.write(f"# Live-parsed math expressions for run\n"
                     f"# Source : {ind_path}\n"
                     f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"# Each block below is one DCE-reduced individual.\n\n")

    n_seen = 0
    n_fault_v = 0
    n_fault_c = 0
    last_status = time.time()

    try:
        while True:
            if not ind_path.exists():
                time.sleep(poll_interval)
                continue
            try:
                size = ind_path.stat().st_size
            except OSError:
                time.sleep(poll_interval)
                continue

            if size < offset:
                # File was truncated / rotated; reset.
                print(f"[watch] file shrank ({size} < {offset}); "
                      f"resetting cursor to 0", flush=True)
                offset = 0

            if size == offset:
                # No new bytes.  Periodic status line every ~30s.
                if time.time() - last_status > 30:
                    print(f"[watch] idle  seen={n_seen}  "
                          f"v_fault={n_fault_v} c_fault={n_fault_c}  "
                          f"cursor={offset}", flush=True)
                    last_status = time.time()
                time.sleep(poll_interval)
                continue

            # New bytes: read incrementally.
            new_records: List[Dict[str, Any]] = []
            with ind_path.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read(size - offset)
            # Stop at the last newline so we never parse a partial line.
            last_nl = chunk.rfind(b"\n")
            if last_nl < 0:
                # No complete line yet.
                time.sleep(poll_interval)
                continue
            complete = chunk[: last_nl + 1]
            consumed = len(complete)
            for raw in complete.splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    print(f"[watch] skip malformed line: {e}", flush=True)
                    continue
                new_records.append(rec)
            offset += consumed
            cursor_path.write_text(str(offset), encoding="utf-8")

            if not new_records:
                continue

            with live_jsonl.open("a", encoding="utf-8") as fh_j, \
                    live_log.open("a", encoding="utf-8") as fh_t:
                for rec in new_records:
                    parsed = _parse_record(rec)
                    if parsed.get("v2c_fault"):
                        n_fault_v += 1
                    if parsed.get("c2v_fault"):
                        n_fault_c += 1
                    fh_j.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    fh_t.write(_format_rec_text(parsed, max_chars) + "\n")
                    n_seen += 1
            print(f"[watch] +{len(new_records)} parsed  "
                  f"total={n_seen}  v_fault={n_fault_v} c_fault={n_fault_c}  "
                  f"cursor={offset}", flush=True)
            last_status = time.time()
    except KeyboardInterrupt:
        print(f"[watch] stopped by user.  total parsed={n_seen}  "
              f"v_fault={n_fault_v} c_fault={n_fault_c}", flush=True)
        return 0


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name", help="Sub-folder under results/logged_evolution/")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Per-gen top-K (by fitness asc) to dump to the .txt "
                         "summary. Default 5.")
    ap.add_argument("--max-chars", type=int, default=500,
                    help="Truncate each printed math line to this many chars.")
    ap.add_argument("--all", action="store_true",
                    help="Dump every individual to the .txt summary (in "
                         "addition to the full parsed_math.jsonl).")
    ap.add_argument("--watch", action="store_true",
                    help="Stay running: tail individuals.jsonl forever and "
                         "append parsed math to <run>/parsed_math/"
                         "parsed_math_live.{jsonl,log}.")
    ap.add_argument("--poll-interval", type=float, default=2.0,
                    help="Seconds between polls in --watch mode (default 2.0).")
    args = ap.parse_args()

    run_dir = find_run_dir(args.run_name)
    ind_path = run_dir / "individuals.jsonl"
    out_dir = run_dir / "parsed_math"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.watch:
        return watch_loop(
            ind_path, out_dir,
            poll_interval=max(0.1, float(args.poll_interval)),
            max_chars=int(args.max_chars),
        )

    if not ind_path.exists():
        raise SystemExit(f"missing individuals.jsonl in {run_dir}")

    out_dir = run_dir / "parsed_math"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_out = out_dir / "parsed_math.jsonl"
    txt_out = out_dir / "parsed_math_top.txt"

    n_total = 0
    n_fault_v = 0
    n_fault_c = 0
    by_gen: Dict[int, List[Dict[str, Any]]] = {}

    print(f"[parse] reading {ind_path}", flush=True)
    with jsonl_out.open("w", encoding="utf-8") as fh:
        for rec in iter_jsonl(ind_path):
            out = _parse_record(rec)
            if out.get("v2c_fault"):
                n_fault_v += 1
            if out.get("c2v_fault"):
                n_fault_c += 1
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            by_gen.setdefault(out["gen"], []).append(out)
            n_total += 1
            if n_total % 100 == 0:
                print(f"[parse]   {n_total} individuals parsed "
                      f"(v_fault={n_fault_v} c_fault={n_fault_c})",
                      flush=True)

    print(f"[parse] total={n_total}  v_fault={n_fault_v}  "
          f"c_fault={n_fault_c}", flush=True)
    print(f"[parse] full jsonl -> {jsonl_out}", flush=True)

    # ------------------- Human-readable summary -------------------------- #
    with txt_out.open("w", encoding="utf-8") as fh:
        fh.write(f"# Run: {args.run_name}\n")
        fh.write(f"# Source: {ind_path}\n")
        fh.write(f"# Total individuals parsed: {n_total}\n")
        fh.write(f"# v_fault={n_fault_v}  c_fault={n_fault_c}\n")
        fh.write(f"# top_k_per_gen={args.top_k}  all={args.all}  "
                 f"max_chars={args.max_chars}\n\n")

        for gen in sorted(by_gen.keys()):
            group = by_gen[gen]
            group_sorted = sorted(group, key=lambda r: r["fitness"])
            target = group_sorted if args.all else group_sorted[: args.top_k]
            fh.write(f"================ gen {gen:03d}  "
                     f"(n={len(group)}  best_fit="
                     f"{group_sorted[0]['fitness']:+.4f}) ================\n")
            for rec in target:
                fh.write("\n" + _format_rec_text(rec, args.max_chars))
            fh.write("\n")

    print(f"[parse] human summary -> {txt_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
