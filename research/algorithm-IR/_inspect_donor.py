from evolution.ir_pool import _build_default_donor_irs
ds = _build_default_donor_irs()
from collections import Counter

print(f"Total donors: {len(ds)}")
multi = 0
empty_succs = 0
for spec_name, ir in ds.items():
    nblk = len(ir.blocks)
    if nblk > 1:
        multi += 1
    has_empty_succs = False
    for bid, b in ir.blocks.items():
        if not b.succs and not b.preds and nblk > 1:
            has_empty_succs = True
    if has_empty_succs:
        empty_succs += 1
print(f"Multi-block donors: {multi}")
print(f"Donors with empty preds/succs but multi-block: {empty_succs}")

# Show first multi-block donor's structure
for spec_name, ir in ds.items():
    if len(ir.blocks) > 1:
        print(f"\n=== Donor {spec_name} ({len(ir.blocks)} blocks) ===")
        print(f"entry: {ir.entry_block}")
        for bid, b in ir.blocks.items():
            terms = []
            for o in b.op_ids:
                op = ir.ops[o]
                if op.opcode in ('jump', 'branch', 'return'):
                    terms.append(f"{op.opcode}{op.attrs}")
            print(f"  blk {bid}: succs={b.succs} preds={b.preds}")
            print(f"    terms: {terms}")
        break
