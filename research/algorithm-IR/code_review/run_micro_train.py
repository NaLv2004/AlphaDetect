"""Wrapper that runs train_gnn and writes output to a log file in real time."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LOG = ROOT / "results" / "gnn_training" / "micro_run.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-B", "-u", "train_gnn.py",
    "--gens", "1",
    "--proposals", "0",
    "--pool-size", "4",
    "--n-trials", "1",
    "--warmstart-gens", "0",
    "--snr-start", "16",
    "--snr-target", "16",
    "--micro-pop-size", "4",
    "--micro-generations", "1",
    "--viz-grafts-per-gen", "0",
    "--no-fii-view",
]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

t0 = time.time()
print(f"== launching: {' '.join(cmd)}")
print(f"== log: {LOG}")
with LOG.open("w", encoding="utf-8", errors="replace") as f:
    f.write(f"# launched {time.ctime()}\n")
    f.flush()
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env,
        stdout=f, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    timeout_sec = int(os.environ.get("MICRO_RUN_TIMEOUT", "600"))
    deadline = t0 + timeout_sec
    while True:
        rc = proc.poll()
        if rc is not None:
            print(f"== exited rc={rc} after {time.time()-t0:.1f}s")
            break
        if time.time() > deadline:
            print(f"== exceeded {timeout_sec}s wallclock; terminating")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            break
        time.sleep(2)

print(f"== log size: {LOG.stat().st_size} bytes")
sys.exit(0)
