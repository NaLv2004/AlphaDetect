import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from evolution.ir_pool import build_ir_pool
from evolution.gnn_pattern_matcher import _compute_return_slice_values

pool = build_ir_pool()
print(f"pool size: {len(pool)}")
print()
print("genome                       ops vals slice_v %liv dead_ops")
print("-" * 70)

SIDE_EFFECT = {"return", "jump", "branch", "store"}

for g in pool[:25]:
    ir = g.ir
    sv = _compute_return_slice_values(ir)
    live_ops = sum(
        1 for op in ir.ops.values()
        if op.opcode in SIDE_EFFECT or any(o in sv for o in op.outputs)
    )
    dead_ops = len(ir.ops) - live_ops
    pct = 100 * len(sv) / max(1, len(ir.values))
    print(f"{g.algo_id:<28} {len(ir.ops):>4} {len(ir.values):>4} {len(sv):>6} {pct:>5.1f} {dead_ops:>5}")

print()
g = pool[0]
ir = g.ir
sv = _compute_return_slice_values(ir)
dead_op_opcodes = Counter()
dead_op_examples = {}

for op in ir.ops.values():
    if op.opcode in SIDE_EFFECT:
        continue
    if not any(o in sv for o in op.outputs):
        dead_op_opcodes[op.opcode] += 1
        if op.opcode not in dead_op_examples:
            val = ir.values.get(op.outputs[0]) if op.outputs else None
            hint = val.name_hint if val else None
            dead_op_examples[op.opcode] = (op.id, op.outputs[:1], op.attrs.get("var_name"), hint, op.attrs.get("_provenance", {}).get("from_slot_id"))

print(f"Dead ops by opcode in {g.algo_id}:")
for opc, n in dead_op_opcodes.most_common():
    print(f"  {opc:<12} x{n:<4} example: id={dead_op_examples[opc][0]}, out={dead_op_examples[opc][1]}, var_name={dead_op_examples[opc][2]}, hint={dead_op_examples[opc][3]}, slot={dead_op_examples[opc][4]}")

# Look at where dead phi nodes come from — likely from loop SSA bookkeeping
print()
print("Block-level dead-op distribution:")
block_dead = Counter()
block_total = Counter()
for op in ir.ops.values():
    block_total[op.block_id] += 1
    if op.opcode not in SIDE_EFFECT and not any(o in sv for o in op.outputs):
        block_dead[op.block_id] += 1
for bid, total in block_total.most_common():
    dead = block_dead.get(bid, 0)
    print(f"  {bid:<22} dead/total={dead}/{total}")
