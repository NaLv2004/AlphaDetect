"""Automatic slot discovery for fully concrete skeletons.

Discovers mutable regions in a FunctionIR via static and/or dynamic analysis.
Uses algorithm_ir.region for rewrite region definition and contract inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.region.selector import RewriteRegion, define_rewrite_region
from algorithm_ir.region.contract import BoundaryContract, infer_boundary_contract
from algorithm_ir.region.slicer import backward_slice_by_values, forward_slice_from_values

from evolution.skeleton_registry import ProgramSpec


@dataclass
class SlotCandidate:
    """A discovered mutable region with auto-generated ProgramSpec."""
    region: RewriteRegion
    contract: BoundaryContract
    score: float
    program_spec: ProgramSpec


def discover_slots(
    func_ir: FunctionIR,
    sample_inputs: list[dict[str, Any]] | None = None,
    mode: str = "auto",
) -> list[SlotCandidate]:
    """Discover candidate mutable slots in a concrete FunctionIR.

    Args:
        func_ir: The concrete function to analyze.
        sample_inputs: Optional runtime inputs for dynamic analysis.
        mode: "static", "dynamic", or "auto" (static + dynamic refinement).

    Returns:
        Ranked list of SlotCandidates (highest score first).
    """
    candidates: list[SlotCandidate] = []

    if mode in ("static", "auto"):
        candidates.extend(_discover_static(func_ir))

    if mode in ("dynamic", "auto") and sample_inputs is not None:
        dynamic = _discover_dynamic(func_ir, sample_inputs)
        if mode == "auto":
            candidates = _merge_candidates(candidates, dynamic)
        else:
            candidates = dynamic

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _discover_static(func_ir: FunctionIR) -> list[SlotCandidate]:
    """Static SESE region enumeration."""
    candidates: list[SlotCandidate] = []

    # Strategy: for each op that produces a value used outside its block,
    # consider backward slices as candidate regions
    for op in func_ir.ops.values():
        if op.opcode in ("const", "phi", "jump", "branch", "return"):
            continue

        # Check if output is scalar-ish
        if not op.outputs:
            continue

        out_val = func_ir.values.get(op.outputs[0])
        if not out_val:
            continue

        # Only consider ops with scalar output types
        if out_val.type_hint and out_val.type_hint not in ("int", "float", "bool", "f64", "i64"):
            continue

        # Backward slice to this value
        try:
            slice_ops = backward_slice_by_values(func_ir, [op.outputs[0]])
        except Exception:
            continue

        if len(slice_ops) < 2 or len(slice_ops) > 20:
            continue

        # Build region
        try:
            region = define_rewrite_region(
                func_ir,
                op_ids=sorted(slice_ops),
                exit_values=[op.outputs[0]],
            )
            contract = infer_boundary_contract(func_ir, region)
        except Exception:
            continue

        # Score: prefer medium-sized regions with few entry ports
        n_entry = len(region.entry_values)
        n_ops = len(region.op_ids)
        score = 1.0 / (1.0 + abs(n_ops - 5)) * 1.0 / (1.0 + n_entry)

        # Auto-generate ProgramSpec from region boundary
        spec = _spec_from_contract(func_ir, region, contract)
        candidates.append(SlotCandidate(region, contract, score, spec))

    return candidates


def _discover_dynamic(
    func_ir: FunctionIR,
    sample_inputs: list[dict[str, Any]],
) -> list[SlotCandidate]:
    """Dynamic analysis: run interpreter, observe value types and ranges."""
    from algorithm_ir.runtime.interpreter import execute_ir

    # Collect traces from sample runs
    candidates: list[SlotCandidate] = []
    runtime_traces = []
    for inputs in sample_inputs[:5]:  # limit samples
        try:
            result = execute_ir(func_ir, inputs, trace=True)
            if hasattr(result, "trace"):
                runtime_traces.append(result.trace)
        except Exception:
            continue

    if not runtime_traces:
        return candidates

    # Reuse static candidates but refine with runtime info
    static = _discover_static(func_ir)
    for cand in static:
        try:
            contract = infer_boundary_contract(
                func_ir, cand.region,
                runtime_trace=runtime_traces[0] if runtime_traces else None,
            )
            # Boost score if runtime confirms scalar output
            if contract.invariants.get("scalar_outputs", False):
                cand.score *= 1.5
            cand.contract = contract
        except Exception:
            pass
        candidates.append(cand)

    return candidates


def _merge_candidates(
    static: list[SlotCandidate],
    dynamic: list[SlotCandidate],
) -> list[SlotCandidate]:
    """Merge static and dynamic candidates, boosting overlapping ones."""
    # Build set of region IDs from dynamic
    dynamic_ids = {c.region.region_id for c in dynamic}
    merged = list(dynamic)  # start with dynamic (they have runtime info)

    for s in static:
        if s.region.region_id not in dynamic_ids:
            merged.append(s)

    return merged


def _spec_from_contract(
    func_ir: FunctionIR,
    region: RewriteRegion,
    contract: BoundaryContract,
) -> ProgramSpec:
    """Auto-generate a ProgramSpec from a region's boundary contract."""
    param_names: list[str] = []
    param_types: list[str] = []

    for vid in contract.input_ports:
        v = func_ir.values.get(vid)
        if v:
            name = v.attrs.get("var_name") or v.name_hint or vid
            param_names.append(name)
            param_types.append(v.type_hint or "float")

    # Infer return type from output ports
    return_type = "float"
    for vid in contract.output_ports:
        v = func_ir.values.get(vid)
        if v and v.type_hint:
            return_type = v.type_hint
            break

    return ProgramSpec(
        name=f"slot_{region.region_id}",
        param_names=param_names,
        param_types=param_types,
        return_type=return_type,
    )
