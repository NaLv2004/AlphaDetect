"""One-off: dump current LMMSE IR after ddcd37d."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evolution.ir_pool import build_ir_pool
import numpy as np

rng = np.random.default_rng(42)
pool = build_ir_pool(rng)
g = next(g for g in pool if g.algo_id == 'lmmse')
ir = g.ir
print('=== LMMSE IR  n_ops=', len(ir.ops), ' n_blocks=', len(ir.blocks))
for blk_id, blk in ir.blocks.items():
    preds = getattr(blk, 'preds', None)
    succs = getattr(blk, 'succs', None)
    print(f'\n[block {blk_id}] preds={preds} succs={succs} n_ops={len(blk.op_ids)}')
    for op_id in blk.op_ids:
        op = ir.ops[op_id]
        attrs = {k: v for k, v in op.attrs.items() if k not in ('xdsl_op',)}
        if 'literal' in attrs:
            s = repr(attrs['literal'])
            if len(s) > 60:
                s = s[:57] + '...'
            attrs['literal'] = s
        ins = ','.join(op.inputs)
        outs = ','.join(op.outputs)
        print(f'  {op_id:8s} {op.opcode:18s} ({ins}) -> ({outs}) {attrs}')

# Also count opcode frequencies
from collections import Counter
c = Counter(op.opcode for op in ir.ops.values())
print('\n=== Opcode counts ===')
for op, n in sorted(c.items(), key=lambda x: -x[1]):
    print(f'  {op:18s} {n}')
