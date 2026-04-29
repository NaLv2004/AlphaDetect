from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.region.selector import RewriteRegion, define_rewrite_region as build_region
from algorithm_ir.region.extract import extract_region_ir
from algorithm_ir.regeneration.codegen import emit_python_source as emit_python

SRC = '''
def hard_decision(x_soft, constellation):
    Nt = x_soft.shape[0]
    K = constellation.shape[0]
    out = x_soft.copy()
    i = 0
    while i < Nt:
        best_idx = 0
        best_dist = float("inf")
        k = 0
        while k < K:
            d = abs(x_soft[i] - constellation[k])
            if d < best_dist:
                best_dist = d
                best_idx = k
            k = k + 1
        out[i] = constellation[best_idx]
        i = i + 1
    return out
'''

ir = compile_source_to_ir(SRC, 'hard_decision')
print('=== FULL IR ===')
print('arg_values:', ir.arg_values)
print('return_values:', ir.return_values)
print('blocks:')
for bid in ir.blocks:
    blk = ir.blocks[bid]
    print(f'  {bid} preds={blk.preds} succs={blk.succs} ops={blk.op_ids}')
print('ops:')
for oid, op in ir.ops.items():
    outs = []
    for ov in op.outputs:
        v = ir.values[ov]
        outs.append(f'{ov}[{(v.attrs or {}).get("var_name","-")}]')
    print(f'  {oid:8} {op.opcode:10} blk={op.block_id:25} ins={op.inputs} outs={outs} attrs={dict(op.attrs)}')

# Find a region: inner while body's `if d<best_dist` block + step
# Pick all ops in any block whose name includes 'while_body' (the inner one).
# Heuristic: take the deepest while_body block.
inner_bodies = [bid for bid in ir.blocks if 'while_body' in bid]
print('\ninner_body candidates:', inner_bodies)

# Try a region that contains: inner while body + if branches + merge phi
# Goal: include op_55 (phi for best_dist) and its uses, but NOT op_27 (init).
target_ops = []
for bid in ['b_while_body_5', 'b_if_true_7', 'b_if_false_8', 'b_if_merge_9', 'b_while_test_4']:
    for oid in ir.blocks[bid].op_ids:
        if ir.ops[oid].opcode not in ('jump', 'branch', 'return'):
            target_ops.append(oid)
print('\ntarget_ops:', target_ops)

region = build_region(ir, op_ids=target_ops)
print(f'\n=== Region (multi-block) ===')
print('block_ids:', region.block_ids)
print('op_ids:', region.op_ids)
print('entry_values:', region.entry_values)
print('exit_values:', region.exit_values)

print('\n=== EXTRACTING ===')
extracted = extract_region_ir(ir, region)
print('extracted.arg_values:', extracted.arg_values)
print('extracted.return_values:', extracted.return_values)
print('extracted blocks:')
for bid in extracted.blocks:
    print(f'  {bid} ops={extracted.blocks[bid].op_ids}')
print('extracted ops (incl synthetic):')
for oid, op in extracted.ops.items():
    outs = []
    for ov in op.outputs:
        v = extracted.values[ov]
        outs.append(f'{ov}[{(v.attrs or {}).get("var_name","-")}]')
    print(f'  {oid:25} {op.opcode:10} blk={op.block_id} ins={op.inputs} outs={outs} attrs={dict(op.attrs)}')

print('\n=== EMITTED PYTHON for extracted ===')
print(emit_python(extracted))
