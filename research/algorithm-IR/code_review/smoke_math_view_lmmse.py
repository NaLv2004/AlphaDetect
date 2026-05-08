"""Phase 1 smoke: build MathView for LMMSE; verify count = 27 SSA-op nodes (target after C1+C2+C2c+C6+M2+J1 absorption)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from evolution.ir_pool import build_ir_pool
from algorithm_ir.ir.math_view import build_math_view, compression_stats
from algorithm_ir.regeneration.codegen import emit_python_source

rng = np.random.default_rng(42)
pool = build_ir_pool(rng)
g = next(g for g in pool if g.algo_id == 'lmmse')
ir = g.ir
print(f"=== LMMSE FunctionIR: n_ops={len(ir.ops)}  n_blocks={len(ir.blocks)}  n_args={len(ir.arg_values)}")

view = build_math_view(ir)
stats = compression_stats(view)
print("\n=== Compression stats ===")
for k, v in stats.items():
    print(f"  {k:22s} = {v}")

print(f"\nTarget: n_ssa_op_nodes = 27  (after C1+C2-chain+C2c+C6+M2+J1 absorption)")
print(f"Got:    n_ssa_op_nodes = {stats['n_ssa_op_nodes']}  ({'PASS' if stats['n_ssa_op_nodes']==27 else 'FAIL'})")

print("\n=== Coverage report ===")
print(view.coverage_report())

print("\n=== Absorbed (op_id -> target_op_id) ===")
for k, v in sorted(view.absorbed.items()):
    src_op = ir.ops[k]
    tgt_op = ir.ops[v]
    src_lit = ''
    if src_op.opcode == 'const':
        s = repr(src_op.attrs.get('literal'))
        src_lit = f" [{s[:40]}]"
    print(f"  {k:8s} ({src_op.opcode}{src_lit}) -> {v:8s} ({tgt_op.opcode})")

print(f"\n=== Dropped (orphan) ops: {len(view.dropped)} ===")
for k in view.dropped:
    op = ir.ops[k]
    extra = ''
    if op.opcode == 'const':
        s = repr(op.attrs.get('literal'))
        extra = f" [{s[:40]}]"
    elif op.opcode == 'get_attr':
        extra = f" attr={op.attrs.get('attr')}"
    print(f"  {k:8s} {op.opcode}{extra}")

print("\n=== MathView nodes (peer DAG) ===")
for n in view.nodes:
    in_str = ','.join(f"{p.node_id}:{p.port_idx}" for p in n.inputs)
    a_keys = sorted(k for k in n.attrs.keys()
                    if k not in ('xdsl_op','type_info','operator','attr','operators',
                                 'qualified_name','literal','name','target','sources',
                                 'true','false','loop_phi','var_name','arg_value_id','type_hint'))
    extra = (' attrs+=' + ','.join(a_keys)) if a_keys else ''
    print(f"  {n.node_id:7s} {n.kind:14s} {n.opcode:38s} in=[{in_str}] op_ids={sorted(n.op_ids)}{extra}")

# Sanity: codegen still works (FunctionIR is unchanged)
print("\n=== Codegen sanity (FunctionIR unchanged) ===")
src = emit_python_source(ir)
print(f"  emit_python_source OK ({len(src)} chars)")
print("  First 12 lines of regenerated source:")
for line in src.splitlines()[:12]:
    print(f"    {line}")

print("\n=== Phase 1 smoke: ", "PASS" if stats['n_ssa_op_nodes']==27 else "FAIL — must reach 27")
