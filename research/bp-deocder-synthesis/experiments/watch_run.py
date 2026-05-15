"""Background watcher: periodically runs poll_report and appends to a log file.

Designed to be started once via run_in_terminal in async mode. Sleeps between
polls so the agent doesn't have to. The agent reads watch_log.txt on demand.

Usage:
    python -B experiments/watch_run.py <run_name> [--interval 90] [--top-k 5]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)


def find_run_dir(run_name: str) -> Path:
    cands = [
        Path("results") / "logged_evolution" / run_name,
        Path(_PROJ) / "results" / "logged_evolution" / run_name,
        Path(_PROJ).parent.parent / "results" / "logged_evolution" / run_name,
    ]
    for c in cands:
        if c.exists():
            return c
    raise SystemExit("run dir not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--interval", type=int, default=90,
                    help="seconds between polls")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=10000,
                    help="hard upper bound on poll cycles (safety)")
    args = ap.parse_args()

    rd = find_run_dir(args.run_name)
    log_p = rd / "watch_log.txt"
    print(f"[watch] writing to {log_p}", flush=True)

    poll_script = Path(_HERE) / "poll_report.py"
    py = sys.executable

    last_indiv_size = -1
    for i in range(args.max_iters):
        try:
            indiv_p = rd / "individuals.jsonl"
            cur_size = indiv_p.stat().st_size if indiv_p.exists() else 0
        except OSError:
            cur_size = 0

        # Only re-poll when individuals.jsonl has grown
        if cur_size != last_indiv_size:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"\n{'='*78}\n[{ts}] poll #{i+1}  (individuals.jsonl bytes={cur_size})\n{'='*78}\n"
            try:
                cp = subprocess.run(
                    [py, "-B", str(poll_script), args.run_name,
                     "--top-k", str(args.top_k)],
                    cwd=_PROJ, capture_output=True, text=True, timeout=120,
                    encoding="utf-8", errors="replace",
                )
                body = cp.stdout + ("\n[stderr]\n" + cp.stderr if cp.stderr else "")
            except Exception as e:
                body = f"[poll_report error] {type(e).__name__}: {e}"
            with log_p.open("a", encoding="utf-8") as fh:
                fh.write(header)
                fh.write(body)
                fh.write("\n")
            print(f"[watch] poll #{i+1} written ({len(body)} chars)", flush=True)
            last_indiv_size = cur_size
        else:
            print(f"[watch] no new data, sleeping {args.interval}s", flush=True)

        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
