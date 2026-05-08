"""Dump LMMSE IR (op listing + regenerated Python) for analysis."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from evolution.ir_pool import build_ir_pool
from algorithm_ir.regeneration.codegen import emit_python_source

rng = np.random.default_rng(42)
pool = build_ir_pool(rng)
genome = next(g for g in pool if g.algo_id == "lmmse")
ir = genome.ir

print(f"=== LMMSE IR ===  algo_id={genome.algo_id}")
print(f"n_ops    = {len(ir.ops)}")
print(f"n_values = {len(ir.values)}")
print(f"n_blocks = {len(ir.blocks)}")
print(f"entry    = {ir.entry_block}")
print()

# count phi
n_phi = sum(1 for op in ir.ops.values() if op.opcode == "cf.phi")
print(f"n_phi    = {n_phi}")
print()

# per-block dump
print("=== Per-block op listing ===")
for bid, block in ir.blocks.items():
    print(f"\n[block {bid}] ops={len(block.op_ids)} preds={block.preds} succs={block.succs}")
    for oid in block.op_ids:
        op = ir.ops[oid]
        operands = ",".join(op.inputs) if op.inputs else ""
        results = ",".join(op.outputs) if op.outputs else ""
        attrs_str = ""
        if op.attrs:
            kv = []
            for k, v in op.attrs.items():
                vs = repr(v)
                if len(vs) > 60:
                    vs = vs[:57] + "..."
                kv.append(f"{k}={vs}")
            attrs_str = " {" + ", ".join(kv) + "}"
        print(f"  {oid:8s} {op.opcode:24s} ({operands}) -> ({results}){attrs_str}")

print("\n=== Regenerated Python source ===")
src = emit_python_source(ir)
print(src)
