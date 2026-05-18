"""Smoke: measure validator pass rate for stack-aware cpp seeder
under the OMS op_filter, vs the previous baseline.

Run from research/bp-deocder-synthesis:
    C:\\ProgramData\\anaconda3\\envs\\AutoGenOld\\python.exe -B code_review/smoke_stack_aware.py
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpp_seeder"))

import pushgp_cpp_seeder as M

CFG = json.load(open(ROOT / "configs" / "op_filter_oms.json"))
V2C = CFG["v2c"]
C2V = CFG["c2v"]

print(f"v2c ops ({len(V2C)}): {V2C}")
print(f"c2v ops ({len(C2V)}): {C2V}\n")

for side, allowed in [("v2c", V2C), ("c2v", C2V)]:
    t0 = time.perf_counter()
    handles, attempts, fps = M.parallel_seed(
        side=side,
        n_target=50,
        max_attempts=200_000,
        threads=8,
        min_size=4,
        max_size=12,
        deg=6,
        num_configs=4,
        num_permutations=4,
        chunk_attempts=64,
        base_seed=20260518,
        allowed_op_names=allowed,
    )
    dt = time.perf_counter() - t0
    rate = len(handles) / attempts if attempts else 0.0
    rps = attempts / dt if dt > 0 else float("inf")
    print(f"[{side}] n_valid={len(handles)} attempts={attempts} "
          f"pass={rate*100:.3f}% time={dt:.2f}s rate={rps:,.0f} att/s")
