"""Reproduce a failing graft to inspect why its blocks become unreachable."""
import sys, traceback
sys.path.insert(0, '.')

# Monkey-patch graft_general to log the IR state on failure
import algorithm_ir.grafting.graft_general as gg
_orig_check = None

# Insert a hook that, on the unreachable-block ValueError, dumps the IR's CFG
import os, json
DUMP_DIR = "D:\\Temp\\graft_diag"
os.makedirs(DUMP_DIR, exist_ok=True)

# Patch the ValueError raise: instead of just raising, dump CFG first
_orig_raise = ValueError
_dump_count = [0]

# Wrap graft_general to dump on failure
_orig_graft = gg.graft_general

def _wrapped_graft(*args, **kwargs):
    try:
        return _orig_graft(*args, **kwargs)
    except ValueError as e:
        msg = str(e)
        if "unreachable blocks" in msg and _dump_count[0] < 5:
            _dump_count[0] += 1
            # Try to find the IR via locals walk - difficult. Instead just print msg.
            print(f"[DIAG#{_dump_count[0]}] {msg}")
        raise

gg.graft_general = _wrapped_graft

# Also patch _inline_multiblock_donor to dump the donor + post-clone state
_orig_inline = gg._inline_multiblock_donor
_inline_count = [0]

def _wrapped_inline(ir, host_block_id, region_op_ids, cloned_ops, cloned_values, cloned_blocks, donor_entry_block_id, id_map, donor_ir):
    if _inline_count[0] < 3:
        print(f"\n[INLINE#{_inline_count[0]}] donor.entry={donor_ir.entry_block} -> {donor_entry_block_id}")
        print(f"  donor blocks: {len(donor_ir.blocks)}")
        for bid, b in donor_ir.blocks.items():
            new_bid = id_map.get(bid, '???')
            term_info = []
            for o in b.op_ids:
                op = donor_ir.ops.get(o)
                if op and op.opcode in ('jump', 'branch', 'return'):
                    term_info.append(f"{op.opcode}{dict(op.attrs)}")
            print(f"    donor blk {bid} -> {new_bid}: succs={b.succs} terms={term_info}")
        print(f"  AFTER CLONE:")
        for bid, b in cloned_blocks.items():
            term_info = []
            for o in b.op_ids:
                op = cloned_ops.get(o)
                if op and op.opcode in ('jump', 'branch', 'return'):
                    term_info.append(f"{op.opcode}{dict(op.attrs)}")
            print(f"    cloned blk {bid}: succs={b.succs} preds={b.preds} terms={term_info}")
        _inline_count[0] += 1
    return _orig_inline(ir, host_block_id, region_op_ids, cloned_ops, cloned_values, cloned_blocks, donor_entry_block_id, id_map, donor_ir)

gg._inline_multiblock_donor = _wrapped_inline

# Run a small training to trigger
import subprocess
os.environ["PYTHONPATH"] = "."
# Just import and run a single graft
from train_gnn import main
import argparse
sys.argv = ["train_gnn.py", "--gens", "1", "--proposals", "20", "--n-trials", "1",
            "--warmstart-gens", "0", "--snr-start", "20", "--snr-target", "20",
            "--max-region-ops", "48", "--max-cut-values", "6", "--max-boundary-outputs", "4",
            "--timeout", "1.0", "--train-steps", "2"]
try:
    main()
except SystemExit:
    pass
