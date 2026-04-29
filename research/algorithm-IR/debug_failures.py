"""Find a sample where mask passes feasibility but validate fails."""
import random
import sys
sys.path.insert(0, ".")

from evolution.ir_pool import build_ir_pool
from evolution.gnn_pattern_matcher import _compute_return_slice_values
from evolution.host_region_mask import (
    cut_step_mask,
    filter_dead_code_outputs,
    is_output_combo_feasible,
    output_step_mask,
    precompute_op_closures,
    _exit_values,
    _nontrivial_op_count,
)
from algorithm_ir.region.slicer import (
    enumerate_observable_values,
    enumerate_cut_candidates,
    backward_slice_until_values,
    validate_boundary_region,
)
from algorithm_ir.region.selector import BoundaryRegionSpec, define_rewrite_region

random.seed(42)
MAX_REG_OPS = 24
MAX_REG_INPUTS = 8
MAX_REG_OUTPUTS = 2
MIN_REG_OPS = 1

pool = build_ir_pool()
g = next(g for g in pool if g.algo_id.lower() == "kbest")
ir = g.ir
return_slice = _compute_return_slice_values(ir)
observable = enumerate_observable_values(ir)
host_obs, _ = filter_dead_code_outputs(observable, return_slice)
op_closures = precompute_op_closures(ir, host_obs)

print(f"kbest: {len(ir.ops)} ops, {len(host_obs)} obs after layer 1")

found = 0
for trial in range(500):
    # Random feasible output combo
    selected: list[str] = []
    available = list(range(len(host_obs)))
    for step in range(2):
        remaining_ids = [host_obs[i] for i in available]
        mask, stop_allowed = output_step_mask(
            ir, op_closures, selected, remaining_ids,
            max_region_ops=MAX_REG_OPS,
            max_region_outputs=MAX_REG_OUTPUTS,
        )
        kept = [i for i, ok in enumerate(mask) if ok]
        options = list(kept)
        if (step > 0) and stop_allowed:
            options.append(-1)
        if not options:
            break
        ch = random.choice(options)
        if ch == -1:
            break
        selected.append(remaining_ids[ch])
        rm_idx = available[ch]
        available = [i for i in available if i != rm_idx]
    if not selected:
        continue
    # Feasibility check
    if not is_output_combo_feasible(
        ir, selected,
        max_region_ops=MAX_REG_OPS,
        max_region_outputs=MAX_REG_OUTPUTS,
    ):
        continue
    # Try cut sampling
    cut_cands = enumerate_cut_candidates(ir, selected)
    cuts: list[str] = []
    cut_avail = list(range(len(cut_cands)))
    dead_end = False
    for step in range(3):
        rem_ids = [cut_cands[i] for i in cut_avail]
        mask4, stop4 = cut_step_mask(
            ir, selected, cuts, rem_ids,
            max_region_ops=MAX_REG_OPS,
            min_region_ops=MIN_REG_OPS,
            max_region_inputs=MAX_REG_INPUTS,
            max_region_outputs=MAX_REG_OUTPUTS,
        )
        kept4 = [i for i, ok in enumerate(mask4) if ok]
        opts = list(kept4)
        if stop4:
            opts.append(-1)
        if not opts:
            dead_end = True
            break
        ch = random.choice(opts)
        if ch == -1:
            break
        cuts.append(rem_ids[ch])
        rm = cut_avail[ch]
        cut_avail = [i for i in cut_avail if i != rm]
    region = define_rewrite_region(
        ir, boundary_spec=BoundaryRegionSpec(output_values=selected, cut_values=cuts),
    )
    val = validate_boundary_region(
        ir, region,
        min_region_ops=MIN_REG_OPS, max_region_ops=MAX_REG_OPS,
        max_region_inputs=MAX_REG_INPUTS, max_region_outputs=MAX_REG_OUTPUTS,
    )
    if not val.is_valid:
        # Found a failure!
        print(f"\nTrial {trial} FAILURE:")
        print(f"  outputs={selected} cuts={cuts}")
        print(f"  cut_cands available: {len(cut_cands)}")
        print(f"  dead_end during sampling: {dead_end}")
        print(f"  validity: {val}")
        # My helper count:
        my_op_set = set(backward_slice_until_values(ir, selected, cuts))
        my_exits = _exit_values(ir, my_op_set)
        my_size = _nontrivial_op_count(ir, my_op_set)
        print(f"  my mask: op_set size={len(my_op_set)} nontrivial={my_size} exits={len(my_exits)}: {sorted(my_exits)}")
        print(f"  region.op_ids={len(region.op_ids)} entry_values={region.entry_values} exit_values={region.exit_values}")
        # Cut mask state
        cur_ops = backward_slice_until_values(ir, selected, cuts)
        cur_size = _nontrivial_op_count(ir, cur_ops)
        cur_exits = _exit_values(ir, cur_ops)
        print(f"  Layer4 stop check: size={cur_size} <= {MAX_REG_OPS}? exits={len(cur_exits)} <= {MAX_REG_OUTPUTS}?")
        found += 1
        if found >= 3:
            break
