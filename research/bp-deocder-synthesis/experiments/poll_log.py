"""Poll a logged-evolution run directory and print a live summary.

Usage:
    python experiments/poll_log.py results/logged_evolution/<run>
    python experiments/poll_log.py            # auto-pick newest run
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path("results") / "logged_evolution"


def _newest_run() -> Optional[Path]:
    if not ROOT.exists():
        return None
    runs = sorted([p for p in ROOT.iterdir() if p.is_dir()],
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def _summarize(run_dir: Path) -> None:
    meta_path = run_dir / "meta.json"
    sum_path = run_dir / "gen_summary.jsonl"
    ind_path = run_dir / "individuals.jsonl"
    if not meta_path.exists():
        print(f"[poll] no meta.json in {run_dir}")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    gens = _read_jsonl(sum_path)
    inds = _read_jsonl(ind_path)

    n_gen_done = len(gens)
    target_gen = meta["evolution"]["generations"]
    pop = meta["evolution"]["pop_size"]
    snrs = meta["snr_list_db"]

    if not gens:
        print(f"[poll] {run_dir.name}: 0/{target_gen} generations logged "
              f"(individuals so far: {len(inds)})")
        return

    last = gens[-1]
    print(
        f"[poll] {run_dir.name}: {n_gen_done}/{target_gen} gens done | "
        f"pop={pop} | SNRs={snrs}"
    )
    print(
        f"       last gen {last['gen']:>3}: best={last['best_fit']:+.4f} "
        f"mean={last['mean_fit']:+.4f} med={last['median_fit']:+.4f} "
        f"valid={last['n_valid_offspring']} t={last['elapsed_s']:.1f}s"
    )

    # Best-ever individual.
    best = None
    for r in inds:
        if best is None or r["fitness"] < best["fitness"]:
            best = r
    if best is not None:
        print(
            f"       best-so-far: gen={best['gen']} idx={best['idx']} "
            f"fit={best['fitness']:+.4f} BER={best['ber_per_snr']} "
            f"FER={best['fer_per_snr']} "
            f"|v2c|={best['v2c_size']}(d{best['v2c_max_depth']}) "
            f"|c2v|={best['c2v_size']}(d{best['c2v_max_depth']})"
        )

    # Trajectory of best fit per gen.
    traj = ", ".join(f"{g['best_fit']:+.3f}" for g in gens[-10:])
    print(f"       best-fit (last 10 gens): {traj}")


def main() -> int:
    run_dir: Optional[Path]
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = _newest_run()
    if run_dir is None or not run_dir.exists():
        print("[poll] no run directory found.")
        return 1
    _summarize(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
