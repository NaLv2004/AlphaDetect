"""Run a tiny 1-gen training cycle, capture exception_repr for each
graft sample, and print the distribution of exception types."""
from __future__ import annotations
import os, sys, argparse, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_gnn as T

# Monkey-patch the analyser to dump exception_repr to a sidecar file as
# samples are produced (since the regular log only counts buckets).
_orig = T._analyse_effective_graft
SIDE = open(r"D:\Temp\b2_exceptions.jsonl", "w", encoding="utf8")
def _wrap(*a, **kw):
    res = _orig(*a, **kw)
    try:
        SIDE.write(json.dumps({
            "verdict": res.get("quality_verdict"),
            "exc": res.get("exception_repr"),
            "behavior_change_rate": res.get("behavior_change_rate"),
            "child_score": res.get("child_score"),
            "host_score": res.get("host_score"),
        }) + "\n")
        SIDE.flush()
    except Exception:
        pass
    return res
T._analyse_effective_graft = _wrap

# Build argv that mirrors the smoke we already ran.
sys.argv = [
    "train_gnn.py",
    "--gens", "1",
    "--pool-size", "16",
    "--micro-pop-size", "8",
    "--micro-generations", "1",
    "--n-trials", "2",
    "--snr-start", "8", "--snr-target", "8",
    "--seed", "11",
    "--viz-grafts-per-gen", "8",
    "--warmstart-gens", "6", "--warmstart-trials", "1",
    "--warmstart-survivor-cap", "12",
]
T.main()
SIDE.close()
