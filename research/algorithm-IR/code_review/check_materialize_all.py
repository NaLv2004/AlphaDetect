"""Quick: verify all detectors in pool materialize without SyntaxError."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from evolution.ir_pool import build_ir_pool
from evolution.materialize import materialize_to_callable

pool = build_ir_pool(np.random.default_rng(42))
failed = []
ok = 0
for g in pool:
    try:
        materialize_to_callable(g)
        ok += 1
    except Exception as e:
        failed.append((g.algo_id, type(e).__name__, str(e)[:120]))
print(f"OK {ok}/{len(pool)}")
if failed:
    print("FAILED:")
    for a, t, m in failed:
        print(f"  {a}: {t}: {m}")
