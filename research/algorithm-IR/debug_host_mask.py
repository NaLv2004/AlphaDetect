"""Debug a single failing host region."""
from __future__ import annotations
import random
import sys
sys.path.insert(0, ".")

from evolution.ir_pool import build_ir_pool
from evolution.gnn_pattern_matcher import _compute_return_slice_values
from evolution.host_region_mask import (
    cut_step_mask,
    filter_dead_code_outputs,
    output_step_mask,
    precompute_op_closures,
    _exit_values,
    _entry_values,
    _nontrivial_op_count,
    is_op_set_connected,
)
from algorithm_ir.region.slicer import (
    enumerate_observable_values,
    enumerate_cut_candidates,
    backward_slice_until_values,
    backward_slice_by_values,
    validate_boundary_region,
)
from algorithm_ir.region.selector import BoundaryRegionSpec, define_rewrite_region

random.seed(7)

MAX_REG_OPS = 24
MAX_REG_INPUTS = 8
MAX_REG_OUTPUTS = 2
MIN_REG_OPS = 1
MAX_BOUNDARY_OUT = 2
MAX_CUT = 3

pool = build_ir_pool()
g = next(g for g in pool if g.algo_id.lower() == "lmmse")
ir = g.ir
return_slice = _compute_return_slice_values(ir)
observable = enumerate_observable_values(ir)
host_obs, _ = filter_dead_code_outputs(observable, return_slice)
print("host_obs values (showing all 22):")
for v in host_obs:
    print(f"  {v}: type={ir.values[v].type_hint}")

op_closures = precompute_op_closures(ir, host_obs)

# Try specific case: select two outputs
test_out_a = host_obs[0]
test_out_b = host_obs[5] if len(host_obs) > 5 else host_obs[-1]
print(f"\nTry outputs=[{test_out_a}, {test_out_b}]")

# Layer 2 step mask check
mask, stop = output_step_mask(ir, op_closures, [test_out_a], [test_out_b])
print(f"Layer 2 mask for {test_out_b} after {test_out_a}: ok={mask}, stop_allowed={stop}")

if mask[0]:
    outs = [test_out_a, test_out_b]
    cut_cands = enumerate_cut_candidates(ir, outs)
    print(f"\nCut candidates ({len(cut_cands)}):")
    for c in cut_cands:
        print(f"  {c}")

    # Layer 4 step 0 mask
    mask4, stop4 = cut_step_mask(
        ir, outs, [], cut_cands,
        max_region_ops=MAX_REG_OPS,
        min_region_ops=MIN_REG_OPS,
        max_region_inputs=MAX_REG_INPUTS,
        max_region_outputs=MAX_REG_OUTPUTS,
    )
    print(f"\nLayer 4 step 0: stop_allowed={stop4}")
    cur_ops = backward_slice_until_values(ir, outs, [])
    cur_size = _nontrivial_op_count(ir, cur_ops)
    cur_inputs = _entry_values(ir, cur_ops)
    cur_exits = _exit_values(ir, cur_ops, outs)
    cur_connected = is_op_set_connected(ir, cur_ops)
    print(f"  current: size={cur_size}, inputs={len(cur_inputs)}, exits={len(cur_exits)}, connected={cur_connected}")
    print(f"  current exit_values: {sorted(cur_exits)}")
    for c, ok in zip(cut_cands, mask4):
        new_ops = backward_slice_until_values(ir, outs, [c])
        new_size = _nontrivial_op_count(ir, new_ops)
        new_inputs = _entry_values(ir, new_ops)
        new_exits = _exit_values(ir, new_ops, outs)
        new_connected = is_op_set_connected(ir, new_ops)
        marker = "OK" if ok else "BLOCK"
        print(f"  cut={c}: size={new_size}, inputs={len(new_inputs)}, exits={len(new_exits)}, conn={new_connected} -> {marker}")

    # Now actually build the region with no cuts and see what happens
    region = define_rewrite_region(
        ir, boundary_spec=BoundaryRegionSpec(output_values=outs, cut_values=[]),
    )
    print(f"\nReal region (no cuts):")
    print(f"  op_ids count: {len(region.op_ids)}")
    print(f"  entry_values: {region.entry_values}")
    print(f"  exit_values: {region.exit_values}")
    val = validate_boundary_region(ir, region,
        min_region_ops=MIN_REG_OPS, max_region_ops=MAX_REG_OPS,
        max_region_inputs=MAX_REG_INPUTS, max_region_outputs=MAX_REG_OUTPUTS)
    print(f"  validity: {val}")
