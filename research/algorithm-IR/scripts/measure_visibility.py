"""Report total vs GNN-visible op counts for every initial-pool algorithm.

Run from the algorithm-IR root:
    python scripts/measure_visibility.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm_ir.region.triviality import is_trivial_op
from evolution.ir_pool import build_ir_pool

import numpy as np

print(f'{"name":<32}{"total":>8}{"visible":>10}{"hidden":>10}{"hidden%":>10}')
print("-" * 70)

totals: list[tuple[str, int, int, int, float]] = []
for genome in build_ir_pool(np.random.default_rng(0)):
    name = getattr(genome, "algo_id", "?")
    ir = genome.ir
    if ir is None:
        continue
    total = len(ir.ops)
    visible = sum(1 for op in ir.ops.values() if not is_trivial_op(op, ir))
    hidden = total - visible
    pct = 100.0 * hidden / max(total, 1)
    totals.append((name, total, visible, hidden, pct))
    print(f"{name:<32}{total:>8}{visible:>10}{hidden:>10}{pct:>9.1f}%")

if totals:
    grand_total = sum(t[1] for t in totals)
    grand_vis = sum(t[2] for t in totals)
    grand_hid = sum(t[3] for t in totals)
    print("-" * 70)
    print(f'{"TOTAL":<32}{grand_total:>8}{grand_vis:>10}{grand_hid:>10}'
          f'{100.0 * grand_hid / max(grand_total, 1):>9.1f}%')
