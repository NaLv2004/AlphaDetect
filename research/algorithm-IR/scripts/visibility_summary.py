"""Generate per-algorithm visibility summary (total/visible/hidden ops)."""
import numpy as np
from evolution.ir_pool import build_ir_pool
from algorithm_ir.visualize import visibility_stats

pool = build_ir_pool(np.random.default_rng(42))
rows = []
for ge in pool:
    tot, vis, hid, pct = visibility_stats(ge.ir)
    rows.append((ge.algo_id, tot, vis, hid, pct))
rows.sort(key=lambda r: -r[1])

print(f"{'algo':<32} {'total':>6} {'visible':>8} {'hidden':>7} {'hidden%':>8}")
print("-" * 65)
for algo, tot, vis, hid, pct in rows[:25]:
    print(f"{algo:<32} {tot:>6} {vis:>8} {hid:>7} {pct:>7.1f}%")

T = sum(r[1] for r in rows)
V = sum(r[2] for r in rows)
H = sum(r[3] for r in rows)
print("-" * 65)
print(f"{'TOTAL (' + str(len(rows)) + ' algos)':<32} {T:>6} {V:>8} {H:>7} {100*H/T:>7.1f}%")
