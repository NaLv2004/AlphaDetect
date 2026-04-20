"""Quick diagnostic: check if build_ir_pool and materialize work."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting test...")
from evolution.ir_pool import build_ir_pool
import numpy as np
pool = build_ir_pool(np.random.default_rng(42))
print(f"Pool size: {len(pool)}")
from evolution.materialize import materialize
for g in pool[:3]:
    try:
        src = materialize(g)
        print(f"{g.algo_id}: OK ({len(src)} chars)")
    except Exception as e:
        import traceback
        print(f"{g.algo_id}: FAIL - {e}")
        traceback.print_exc()
