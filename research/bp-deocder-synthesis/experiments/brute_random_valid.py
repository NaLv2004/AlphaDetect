"""Brute-force probe: how many random tries to get N valid programs.

Thin wrapper around `pushgp.parallel_init.parallel_fill_random` for
manual inspection.  Outputs valid program one-liners + JSON dump.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pushgp.parallel_init import (  # noqa: E402
    DEFAULT_WORKERS, parallel_fill_random,
)
from pushgp.serialize import program_to_dict  # noqa: E402


def _prog_oneliner(prog) -> str:
    parts = []
    for ins in prog:
        s = ins.name
        if ins.code_block:
            s += "{" + _prog_oneliner(ins.code_block) + "}"
        if ins.code_block2:
            s += "{" + _prog_oneliner(ins.code_block2) + "}"
        parts.append(s)
    return " ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=["v2c", "c2v", "both"], default="both")
    p.add_argument("--target", type=int, default=16)
    p.add_argument("--max-attempts", type=int, default=500_000)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--chunk-attempts", type=int, default=5000)
    p.add_argument("--min-size", type=int, default=4)
    p.add_argument("--max-size", type=int, default=16)
    p.add_argument("--run-name", type=str,
                   default=datetime.now().strftime("brute_%Y%m%d_%H%M%S"))
    args = p.parse_args()

    out_dir = ROOT / "results" / "brute_random_valid" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[brute] out={out_dir}", flush=True)
    print(f"[brute] workers={args.workers} target={args.target}/side  "
          f"max_attempts={args.max_attempts}/side  size=[{args.min_size},{args.max_size}]",
          flush=True)

    sides = (["v2c", "c2v"] if args.side == "both" else [args.side])

    def _cb(s, nv, na, dt):
        rate = nv / na if na > 0 else 0.0
        print(f"[brute] {s} valid={nv}/{args.target} attempts={na}  "
              f"rate={rate:.4%}  elapsed={dt:.1f}s", flush=True)

    for s in sides:
        progs, attempts = parallel_fill_random(
            side=s,
            n_target=args.target,
            max_attempts=args.max_attempts,
            workers=args.workers,
            chunk_attempts=args.chunk_attempts,
            min_size=args.min_size,
            max_size=args.max_size,
            base_seed=hash(s) & 0xFFFF,
            progress_cb=_cb,
        )
        rate = args.target / attempts if attempts > 0 else 0.0
        out = {
            "side": s,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "attempts": attempts,
            "n_valid_collected": len(progs),
            "acceptance_rate": rate,
            "attempts_per_valid": attempts / len(progs) if progs else None,
            "valid_programs": [program_to_dict(p) for p in progs],
        }
        path = out_dir / f"{s}_valid.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n=== {s.upper()} ({len(progs)} valid, "
              f"{attempts/len(progs):.0f} attempts/valid) ===", flush=True)
        for i, prog in enumerate(progs):
            print(f"  [{i:>2d}] len={len(prog):>2d}  {_prog_oneliner(prog)}",
                  flush=True)
    print("\n[brute] done", flush=True)


if __name__ == "__main__":
    main()
