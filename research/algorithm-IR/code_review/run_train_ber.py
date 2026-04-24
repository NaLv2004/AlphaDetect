"""Wrapper that runs train_gnn for a longer, BER-meaningful run.

Usage:
    python code_review/run_train_ber.py [GENS] [PROPOSALS] [POOL]

Defaults: 5 generations, 30 proposals/gen, pool=20, n_trials=3.

Logs to ``results/gnn_training/ber_run.log``; polls until exit or
TIMEOUT_SEC seconds elapse.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "gnn_training" / "ber_run.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

GENS = sys.argv[1] if len(sys.argv) > 1 else "5"
PROPS = sys.argv[2] if len(sys.argv) > 2 else "30"
POOL = sys.argv[3] if len(sys.argv) > 3 else "20"
N_TRIALS = sys.argv[4] if len(sys.argv) > 4 else "3"
TIMEOUT_SEC = int(os.environ.get("BER_RUN_TIMEOUT", "5400"))  # 90 min

cmd = [
    sys.executable, "-B", "-u", "train_gnn.py",
    "--gens", GENS,
    "--proposals", PROPS,
    "--pool-size", POOL,
    "--n-trials", N_TRIALS,
    "--warmstart-gens", "0",
    "--snr-start", "16",
    "--snr-target", "16",
    "--micro-pop-size", "8",
    "--micro-generations", "1",
    "--viz-grafts-per-gen", "0",
    "--no-fii-view",
]
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

print(f"== launching: {' '.join(cmd)}")
print(f"== log: {LOG}")
print(f"== timeout: {TIMEOUT_SEC}s")

t0 = time.time()
with LOG.open("w", encoding="utf-8", errors="replace") as f:
    f.write(f"# launched {time.ctime()} cmd={' '.join(cmd)}\n")
    f.flush()
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env,
        stdout=f, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    deadline = t0 + TIMEOUT_SEC
    while True:
        rc = proc.poll()
        if rc is not None:
            print(f"== exited rc={rc} after {time.time()-t0:.1f}s")
            break
        if time.time() > deadline:
            print(f"== exceeded {TIMEOUT_SEC}s wallclock; terminating")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            break
        time.sleep(5)

print(f"== log size: {LOG.stat().st_size} bytes")
sys.exit(0)
