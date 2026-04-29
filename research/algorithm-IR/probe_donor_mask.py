"""Standalone probe for end-to-end (host + donor) masked sampling.

Skips the GNN encoder / training loop entirely.  For each (host, donor)
pair drawn from the IR pool:
  1. Sample a host region using the host mask (uniform sampling over
     the masked candidate set, mirroring ``probe_host_mask.py``).
  2. Validate it via ``validate_boundary_region``.
  3. Derive the boundary signature.
  4. Sample a donor region under that signature using the donor mask
     (uniform sampling, no GNN logits).
  5. Validate the donor region.

Reports per-stage pass counts and the end-to-end host+donor pass rate.
Target: ``end_to_end_rate >= 0.80``.
"""
from __future__ import annotations
import argparse
import random
import sys
import time

sys.path.insert(0, ".")

from evolution.ir_pool import build_ir_pool
from evolution.gnn_pattern_matcher import _compute_return_slice_values
from evolution.host_region_mask import (
    cut_step_mask as host_cut_step_mask,
    filter_dead_code_outputs,
    is_output_combo_feasible,
    output_step_mask as host_output_step_mask,
    precompute_op_closures as host_precompute_op_closures,
)
from evolution.donor_region_mask import (
    donor_pool_signature_compatible,
    donor_cut_pool_union,
    donor_output_step_mask,
    donor_cut_step_mask,
    precompute_op_closures as donor_precompute_op_closures,
)
from evolution.graft_classifier import signature_for_region
from algorithm_ir.region.slicer import (
    enumerate_observable_values,
    enumerate_cut_candidates,
    validate_boundary_region,
)
from algorithm_ir.region.selector import BoundaryRegionSpec, define_rewrite_region
from algorithm_ir.ir.type_lattice import is_subtype


# Match train_gnn smoke-test parameters.
MAX_REG_OPS = 64
MAX_REG_INPUTS = 16
MAX_REG_OUTPUTS = 4
MIN_REG_OPS = 1
MAX_BOUNDARY_OUT = 4
MAX_CUT = 6


def _value_static_type(ir, vid: str) -> str:
    val = ir.values.get(vid)
    if val is None:
        return "unknown"
    th = getattr(val, "type_hint", None)
    if th:
        return str(th)
    attrs = getattr(val, "attrs", None) or {}
    for k in ("type", "static_type", "dtype"):
        v = attrs.get(k)
        if v:
            return str(v)
    return "unknown"


def _types_compatible(donor_t: str, host_t: str) -> bool:
    if not host_t or host_t in ("unknown", "any", "object"):
        return True
    if not donor_t or donor_t in ("unknown", "any", "object"):
        return True
    try:
        return is_subtype(donor_t, host_t) or is_subtype(host_t, donor_t)
    except Exception:
        return True


def sample_value_seq(
    candidate_ids: list[str],
    *,
    max_selected: int,
    allow_empty: bool,
    step_mask_fn,
    rng: random.Random,
) -> tuple[list[str], bool]:
    """Uniform-random masked sequential sampler with STOP support."""
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
        if (allow_empty or step > 0) and stop_allowed:
            options.append(-1)
        chosen = rng.choice(options)
        if chosen == -1:
            break
        local_idx = chosen
        selected.append(remaining_ids[local_idx])
        remove_idx = available[local_idx]
        available = [i for i in available if i != remove_idx]
    return selected, aborted


def sample_donor_outputs_under_sig(
    donor_ir,
    observable: list[str],
    op_closures,
    *,
    entry_types: tuple[str, ...],
    exit_types: tuple[str, ...],
    rng: random.Random,
    diag: dict[str, int] | None = None,
) -> tuple[list[str] | None, str]:
    """Positional, mask-driven donor output sampler. Returns (outs|None, abort_reason)."""
    if len(exit_types) > MAX_BOUNDARY_OUT:
        return None, "donor_too_many_exits"
    chosen: list[str] = []
    chosen_set: set[str] = set()
    for step, host_t in enumerate(exit_types):
        # Type mask first.
        type_mask = [
            (vid not in chosen_set) and _types_compatible(_value_static_type(donor_ir, vid), host_t)
            for vid in observable
        ]
        n_type_ok = sum(type_mask)
        # Layer D2 structural+feasibility mask.
        d2 = donor_output_step_mask(
            donor_ir, op_closures, chosen, observable,
            next_step=step,
            entry_types=entry_types,
            exit_types=exit_types,
            max_region_ops=MAX_REG_OPS,
            min_region_ops=MIN_REG_OPS,
            max_region_inputs=MAX_REG_INPUTS,
            max_cut_budget=MAX_CUT,
        )
        n_d2_ok = sum(d2)
        final = [tm and dm for tm, dm in zip(type_mask, d2)]
        eligible = [i for i, ok in enumerate(final) if ok]
        if not eligible:
            if diag is not None:
                key = f"out_step{step}_of{len(exit_types)}_typeok={n_type_ok}_d2ok={n_d2_ok}"
                diag[key] = diag.get(key, 0) + 1
            return None, "signature_mask_empty_outputs"
        idx = rng.choice(eligible)
        chosen.append(observable[idx])
        chosen_set.add(observable[idx])
    return chosen, ""


def sample_donor_cuts_under_sig(
    donor_ir,
    output_values: list[str],
    *,
    entry_types: tuple[str, ...],
    exit_types: tuple[str, ...],
    rng: random.Random,
) -> tuple[list[str] | None, str]:
    """Positional cut sampler with STOP gate."""
    cut_pool = enumerate_cut_candidates(donor_ir, output_values)
    if len(entry_types) > MAX_CUT:
        return None, "donor_too_many_entries"
    chosen: list[str] = []
    chosen_set: set[str] = set()
    max_iters = len(entry_types)
    for step in range(max_iters):
        d4_mask, stop_allowed = donor_cut_step_mask(
            donor_ir, output_values, chosen, cut_pool,
            entry_types=entry_types,
            exit_types=exit_types,
            max_region_ops=MAX_REG_OPS,
            min_region_ops=MIN_REG_OPS,
            max_region_inputs=MAX_REG_INPUTS,
            max_cut_budget=MAX_CUT,
        )
        # Prefer STOP early when feasible.
        if stop_allowed and step > 0:
            break
        host_t = entry_types[step]
        type_mask = [
            (c not in chosen_set) and _types_compatible(_value_static_type(donor_ir, c), host_t)
            for c in cut_pool
        ]
        final = [tm and dm for tm, dm in zip(type_mask, d4_mask)]
        eligible = [i for i, ok in enumerate(final) if ok]
        options = list(eligible)
        if stop_allowed and step > 0:
            options.append(-1)
        if not options:
            if stop_allowed:
                break
            return None, "signature_mask_empty_cuts"
        idx = rng.choice(options)
        if idx == -1:
            break
        chosen.append(cut_pool[idx])
        chosen_set.add(cut_pool[idx])
    return chosen, ""


def sample_host_region(host_ir, return_slice, rng: random.Random):
    """Returns (region, validity, abort_reason) or (None, None, reason)."""
    observable = enumerate_observable_values(host_ir)
    host_obs, _ = filter_dead_code_outputs(observable, return_slice)
    if not host_obs:
        return None, None, "host_pool_empty"
    op_closures = host_precompute_op_closures(host_ir, host_obs)
    outputs, aborted = sample_value_seq(
        host_obs,
        max_selected=MAX_BOUNDARY_OUT,
        allow_empty=False,
        step_mask_fn=lambda sel, rem: host_output_step_mask(
            host_ir, op_closures, sel, rem,
            max_region_ops=MAX_REG_OPS,
            max_region_outputs=MAX_REG_OUTPUTS,
            max_region_inputs=MAX_REG_INPUTS,
            max_cut_budget=MAX_CUT,
        ),
        rng=rng,
    )
    if aborted or not outputs:
        return None, None, "host_output_dead_end"
    if not is_output_combo_feasible(
        host_ir, outputs,
        max_region_ops=MAX_REG_OPS,
        max_region_outputs=MAX_REG_OUTPUTS,
        max_region_inputs=MAX_REG_INPUTS,
        max_cut_budget=MAX_CUT,
    ):
        return None, None, "host_outputs_infeasible"
    cut_cands = enumerate_cut_candidates(host_ir, outputs)
    cuts, cut_aborted = sample_value_seq(
        cut_cands,
        max_selected=MAX_CUT,
        allow_empty=True,
        step_mask_fn=lambda sel, rem: host_cut_step_mask(
            host_ir, outputs, sel, rem,
            max_region_ops=MAX_REG_OPS,
            min_region_ops=MIN_REG_OPS,
            max_region_inputs=MAX_REG_INPUTS,
            max_region_outputs=MAX_REG_OUTPUTS,
            max_cut_budget=MAX_CUT,
        ),
        rng=rng,
    )
    if cut_aborted:
        return None, None, "host_cut_dead_end"
    try:
        region = define_rewrite_region(
            host_ir,
            boundary_spec=BoundaryRegionSpec(output_values=outputs, cut_values=cuts),
        )
    except Exception as e:
        return None, None, f"host_build_exc:{type(e).__name__}"
    validity = validate_boundary_region(
        host_ir, region,
        min_region_ops=MIN_REG_OPS,
        max_region_ops=MAX_REG_OPS,
        max_region_inputs=MAX_REG_INPUTS,
        max_region_outputs=MAX_REG_OUTPUTS,
    )
    if not validity.is_valid:
        return None, None, f"host_invalid:{validity.reason}"
    return region, validity, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    pool = build_ir_pool()
    print(f"Loaded {len(pool)} algorithms from pool.")

    # Pre-compute host return slices.
    return_slices: dict[str, set] = {}
    for entry in pool:
        try:
            return_slices[entry.algo_id] = _compute_return_slice_values(entry.ir)
        except Exception:
            return_slices[entry.algo_id] = set()

    host_attempts = 0
    host_built = 0
    host_validate_passes = 0
    donor_pair_attempts = 0
    donor_pool_drops = 0
    donor_sampler_attempts = 0
    donor_built = 0
    donor_validate_passes = 0
    donor_buckets: dict[str, int] = {}
    diag: dict[str, int] = {}

    t0 = time.time()
    for _ in range(args.pairs):
        host_attempts += 1
        host = rng.choice(pool)
        donor = rng.choice(pool)
        host_ir = host.ir
        donor_ir = donor.ir

        # ── Host stage ────────────────────────────────────────────
        region, validity, host_reason = sample_host_region(
            host_ir, return_slices.get(host.algo_id, set()), rng,
        )
        if region is None:
            donor_buckets[host_reason] = donor_buckets.get(host_reason, 0) + 1
            continue
        host_built += 1
        host_validate_passes += 1

        # ── Signature ─────────────────────────────────────────────
        try:
            sig = signature_for_region(host_ir, region)
        except Exception as e:
            donor_buckets[f"sig_exc:{type(e).__name__}"] = donor_buckets.get(f"sig_exc:{type(e).__name__}", 0) + 1
            continue

        # ── Donor pool prefilter (Layer D1) ──────────────────────
        donor_pair_attempts += 1
        donor_observable = enumerate_observable_values(donor_ir)
        cut_pool_union = donor_cut_pool_union(donor_ir, donor_observable)
        if not donor_pool_signature_compatible(
            donor_ir, donor_observable, cut_pool_union,
            entry_types=sig.entry_types, exit_types=sig.exit_types,
        ):
            donor_pool_drops += 1
            donor_buckets["donor_pool_signature_mismatch"] = donor_buckets.get("donor_pool_signature_mismatch", 0) + 1
            continue

        # ── Donor output sampling (Layer D2) ──────────────────────
        donor_op_closures = donor_precompute_op_closures(donor_ir, donor_observable)
        donor_sampler_attempts += 1
        # Retry loop: greedy masked sampling can dead-end at later
        # output steps; re-sampling typically recovers.
        d_outs = None
        d_cuts = None
        last_reason = ""
        for _retry in range(10):
            d_outs, last_reason = sample_donor_outputs_under_sig(
                donor_ir, donor_observable, donor_op_closures,
                entry_types=sig.entry_types, exit_types=sig.exit_types,
                rng=rng,
                diag=diag,
            )
            if d_outs is None:
                continue
            d_cuts, last_reason = sample_donor_cuts_under_sig(
                donor_ir, d_outs,
                entry_types=sig.entry_types, exit_types=sig.exit_types,
                rng=rng,
            )
            if d_cuts is not None:
                break
        if d_outs is None or d_cuts is None:
            donor_buckets[last_reason or "donor_unknown_fail"] = donor_buckets.get(last_reason or "donor_unknown_fail", 0) + 1
            continue

        # ── Build & validate donor region ─────────────────────────
        try:
            d_region = define_rewrite_region(
                donor_ir,
                boundary_spec=BoundaryRegionSpec(output_values=d_outs, cut_values=d_cuts),
            )
        except Exception as e:
            donor_buckets[f"donor_build_exc:{type(e).__name__}"] = donor_buckets.get(f"donor_build_exc:{type(e).__name__}", 0) + 1
            continue
        donor_built += 1
        d_validity = validate_boundary_region(
            donor_ir, d_region,
            min_region_ops=MIN_REG_OPS,
            max_region_ops=MAX_REG_OPS,
            max_region_inputs=MAX_REG_INPUTS,
            max_region_outputs=MAX_REG_OUTPUTS,
        )
        if not d_validity.is_valid:
            donor_buckets[f"donor_invalid:{d_validity.reason}"] = donor_buckets.get(f"donor_invalid:{d_validity.reason}", 0) + 1
            continue
        donor_validate_passes += 1

    elapsed = time.time() - t0
    print(f"\n── Probe results ({args.pairs} pairs, {elapsed:.1f}s) ──")
    print(f"Host attempts:           {host_attempts}")
    print(f"Host built:              {host_built}")
    print(f"Host validate_passes:    {host_validate_passes}  ({100.0 * host_validate_passes / max(1, host_attempts):.1f}%)")
    print(f"Donor pair attempts:     {donor_pair_attempts}  (post-host)")
    print(f"Donor pool drops (D1):   {donor_pool_drops}")
    print(f"Donor sampler attempts:  {donor_sampler_attempts}")
    print(f"Donor built:             {donor_built}")
    print(f"Donor validate_passes:   {donor_validate_passes}  (donor rate: {100.0 * donor_validate_passes / max(1, donor_sampler_attempts):.1f}%)")
    print(f"END-TO-END pass rate:    {100.0 * donor_validate_passes / max(1, host_attempts):.1f}%  (target ≥ 80%)")
    print(f"Invalid buckets (top 15):")
    for k, v in sorted(donor_buckets.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {k}: {v}")
    if diag:
        print(f"Diagnostic step buckets (top 15):")
        for k, v in sorted(diag.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
