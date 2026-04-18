"""Check materialized source code for broken detectors."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolution.ir_pool import build_ir_pool
from evolution.materialize import materialize

pool = build_ir_pool()
for g in pool:
    if g.algo_id in ('stack', 'bp', 'amp', 'kbest'):
        print(f"===== {g.algo_id} =====")
        src = materialize(g)
        print(src)
        print()
