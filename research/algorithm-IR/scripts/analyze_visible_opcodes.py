"""Quick analysis: what opcodes survive Plan B's visibility filter?"""
import numpy as np
from collections import Counter
from evolution.ir_pool import build_ir_pool
from algorithm_ir.region.triviality import is_trivial_op

pool = build_ir_pool(np.random.default_rng(42))
ctr_visible = Counter()
ctr_hidden = Counter()
for ge in pool:
    for op in ge.ir.ops.values():
        if is_trivial_op(op, ge.ir):
            ctr_hidden[op.opcode] += 1
        else:
            ctr_visible[op.opcode] += 1

total_v = sum(ctr_visible.values())
total_h = sum(ctr_hidden.values())
print(f"=== VISIBLE OPS ({total_v}) ===")
print(f"{'opcode':<22} {'count':>8} {'pct':>7}")
for op, c in ctr_visible.most_common():
    print(f"{op:<22} {c:>8} {100*c/total_v:>6.1f}%")
print()
print(f"=== HIDDEN OPS ({total_h}) ===")
print(f"{'opcode':<22} {'count':>8} {'pct':>7}")
for op, c in ctr_hidden.most_common():
    print(f"{op:<22} {c:>8} {100*c/total_h:>6.1f}%")
