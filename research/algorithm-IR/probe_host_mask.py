"""Quick probe: sample 200 host regions on lmmse with mask enabled,
verify every one passes validate_boundary_region without fallback."""
from __future__ import annotations
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
)
from algorithm_ir.region.slicer import (
    enumerate_observable_values,
    enumerate_cut_candidates,
    validate_boundary_region,
)
from algorithm_ir.region.selector import BoundaryRegionSpec, define_rewrite_region

random.seed(0)

MAX_REG_OPS = 24
MAX_REG_INPUTS = 8
MAX_REG_OUTPUTS = 2
MIN_REG_OPS = 1
MAX_BOUNDARY_OUT = 2
MAX_CUT = 3


def sample_value_seq(
    candidate_ids: list[str],
    *,
    max_selected: int,
    allow_empty: bool,
    step_mask_fn,
):
    available = list(range(len(candidate_ids)))
    selected: list[str] = []
    aborted = False
    for step in range(max_selected):
        if not available:
            break
        remaining_ids = [candidate_ids[i] for i in available]
        mask, stop_allowed = step_mask_fn(list(selected), remaining_ids)
        kept_local = [i for i, ok in enumerate(mask) if ok]
        if not kept_local:
            if (allow_empty or step > 0) and stop_allowed:
                break
            aborted = True
            break
        options = list(kept_local)
        # Append __STOP__ when allowed
        if (allow_empty or step > 0) and stop_allowed:
            options.append(-1)  # -1 = stop sentinel
        chosen = random.choice(options)  # uniform sampling for probe
        if chosen == -1:
            break
        local_idx = chosen
        selected.append(remaining_ids[local_idx])
        remove_idx = available[local_idx]
        available = [i for i in available if i != remove_idx]
    return selected, aborted


def main() -> None:
    pool = build_ir_pool()
    results = {}
    for algo_id_match in ("lmmse", "zf", "osic", "kbest", "bp", "ep", "amp"):
        try:
            g = next(g for g in pool if g.algo_id.lower() == algo_id_match)
        except StopIteration:
            continue
        ir = g.ir
        return_slice = _compute_return_slice_values(ir)
        observable = enumerate_observable_values(ir)
        host_obs, _ = filter_dead_code_outputs(observable, return_slice)
        if not host_obs:
            print(f"{algo_id_match}: layer1 emptied pool, skipping")
            continue
        op_closures = precompute_op_closures(ir, host_obs)

        ok = 0
        invalid_bucket: dict[str, int] = {}
        attempts = 30
        dead_end = 0
        for _ in range(attempts):
            outputs, out_aborted = sample_value_seq(
                host_obs,
                max_selected=MAX_BOUNDARY_OUT,
                allow_empty=False,
                step_mask_fn=lambda sel, rem: output_step_mask(
                    ir, op_closures, sel, rem,
                    max_region_ops=MAX_REG_OPS,
                    max_region_outputs=MAX_REG_OUTPUTS,
                    max_region_inputs=MAX_REG_INPUTS,
                    max_cut_budget=MAX_CUT,
                ),
            )
            if out_aborted or not outputs:
                dead_end += 1
                continue
            if not is_output_combo_feasible(
                ir, outputs,
                max_region_ops=MAX_REG_OPS,
                max_region_outputs=MAX_REG_OUTPUTS,
                max_region_inputs=MAX_REG_INPUTS,
                max_cut_budget=MAX_CUT,
            ):
                invalid_bucket["infeasible_outputs"] = invalid_bucket.get("infeasible_outputs", 0) + 1
                continue
            cut_cands = enumerate_cut_candidates(ir, outputs)
            cuts, cut_aborted = sample_value_seq(
                cut_cands,
                max_selected=MAX_CUT,
                allow_empty=True,
                step_mask_fn=lambda sel, rem: cut_step_mask(
                    ir, outputs, sel, rem,
                    max_region_ops=MAX_REG_OPS,
                    min_region_ops=MIN_REG_OPS,
                    max_region_inputs=MAX_REG_INPUTS,
                    max_region_outputs=MAX_REG_OUTPUTS,
                    max_cut_budget=MAX_CUT,
                ),
            )
            if cut_aborted:
                invalid_bucket["cut_dead_end"] = invalid_bucket.get("cut_dead_end", 0) + 1
                continue
            try:
                region = define_rewrite_region(
                    ir,
                    boundary_spec=BoundaryRegionSpec(
                        output_values=outputs, cut_values=cuts,
                    ),
                )
            except Exception as e:
                invalid_bucket[f"build_exc:{type(e).__name__}"] = invalid_bucket.get(f"build_exc:{type(e).__name__}", 0) + 1
                continue
            validity = validate_boundary_region(
                ir, region,
                min_region_ops=MIN_REG_OPS,
                max_region_ops=MAX_REG_OPS,
                max_region_inputs=MAX_REG_INPUTS,
                max_region_outputs=MAX_REG_OUTPUTS,
            )
            if validity.is_valid:
                ok += 1
            else:
                invalid_bucket[validity.reason] = invalid_bucket.get(validity.reason, 0) + 1
        print(f"{algo_id_match}: ok={ok}/{attempts} dead_end={dead_end} invalid={invalid_bucket}")
        results[algo_id_match] = (ok, attempts, dead_end, invalid_bucket)

    total_ok = sum(v[0] for v in results.values())
    total_attempts = sum(v[1] for v in results.values())
    total_dead = sum(v[2] for v in results.values())
    if total_attempts:
        print(f"\nTOTAL: ok={total_ok}/{total_attempts} ({100.0*total_ok/total_attempts:.1f}%) dead_end={total_dead}")


if __name__ == "__main__":
    main()
