"""Simulate a UI launch end-to-end without spinning the FastAPI server.

Builds the exact CLI that ui_server.LaunchParams.build_cmdline() would
emit, then spawns the training process the same way the UI would.
Tails the stdout log until DONE or timeout, prints the tail.
"""
from __future__ import annotations
import os, sys, time, subprocess
from pathlib import Path

BPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BPS))

from ui.ui_server import LaunchParams  # type: ignore

p = LaunchParams(
    pop_size=8, gens=1, seed=7,
    snr_list="2,3", n_frames=2, max_iter=8, workers=4,
    info_len_A=176, code_length_E=352,
    dce_bgn=2, dce_set_idx=1, dce_zc=2,
    dce_bp=True, dce_bp_max_iter=4, dce_bp_n_frames=1, dce_bp_snr_db=-2.0,
    from_scratch=True, bind_pairs=True, use_cpp_fitness=True,
)
run_name = "ui_smoke_AE"
cmd = p.build_cmdline(run_name)
print("CMD:", " ".join(cmd), flush=True)

logdir = BPS / "results" / "logged_evolution" / run_name
log = BPS / "code_review" / "ui_smoke_AE.log"
err = BPS / "code_review" / "ui_smoke_AE.err"
for f in (log, err):
    if f.exists(): f.unlink()

proc = subprocess.Popen(cmd, cwd=str(BPS),
                         stdout=open(log, "wb"), stderr=open(err, "wb"))
print(f"PID={proc.pid}  log={log}", flush=True)
t0 = time.time()
while True:
    rc = proc.poll()
    if rc is not None:
        print(f"\n[exit] rc={rc} elapsed={time.time()-t0:.1f}s", flush=True)
        break
    if time.time() - t0 > 600:
        print(f"\n[timeout]  killing", flush=True)
        proc.kill(); break
    time.sleep(5)
    sz = log.stat().st_size if log.exists() else 0
    print(f"  t={time.time()-t0:5.1f}s  log={sz} bytes", flush=True)

print("\n---LOG TAIL---")
if log.exists():
    txt = log.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(txt[-60:]))
print("\n---ERR TAIL---")
if err.exists():
    txt = err.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(txt[-30:]))

print(f"\nrun_dir contents: {sorted(os.listdir(logdir)) if logdir.exists() else 'MISSING'}")
