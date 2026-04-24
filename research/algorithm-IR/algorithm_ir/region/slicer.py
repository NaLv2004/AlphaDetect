from __future__ import annotations

from dataclasses import dataclass
from collections import deque

from algorithm_ir.ir.model import FunctionIR


def backward_slice_by_values(func_ir: FunctionIR, exit_values: list[str]) -> set[str]:
    seen_values = set(exit_values)
    queue = deque(exit_values)
    selected_ops: set[str] = set()
    while queue:
        value_id = queue.popleft()
        val = func_ir.values.get(value_id)
        if val is None:
            continue
        def_op = val.def_op
        if def_op is None or def_op in selected_ops:
            continue
        if def_op not in func_ir.ops:
            continue
        selected_ops.add(def_op)
        for input_value in func_ir.ops[def_op].inputs:
            if input_value not in seen_values:
                seen_values.add(input_value)
                queue.append(input_value)
    return selected_ops


def forward_slice_from_values(func_ir: FunctionIR, seed_values: list[str]) -> set[str]:
    queue = deque(seed_values)
    seen_values = set(seed_values)
    selected_ops: set[str] = set()
    while queue:
        value_id = queue.popleft()
        for use_op in func_ir.values[value_id].use_ops:
            if use_op in selected_ops:
                continue
            selected_ops.add(use_op)
            for output in func_ir.ops[use_op].outputs:
                if output not in seen_values:
                    seen_values.add(output)
                    queue.append(output)
    return selected_ops


@dataclass
class RegionValidity:
    is_valid: bool
    reason: str
    n_ops: int
    n_inputs: int
    n_outputs: int
    is_connected: bool
    selected_vs_effective_match: bool


def backward_slice_until_values(
    func_ir: FunctionIR,
    output_values: list[str],
    cut_values: list[str],
) -> set[str]:
    """Backward-slice from ``output_values`` until reaching ``cut_values``.

    Values in ``cut_values`` act as explicit region-input boundaries: they are
    not expanded through their defining ops.
    """
    cut_set = set(cut_values)
    seen_values = set(output_values)
    queue = deque(output_values)
    selected_ops: set[str] = set()

    while queue:
        value_id = queue.popleft()
        if value_id in cut_set:
            continue
        value = func_ir.values.get(value_id)
        if value is None:
            continue
        def_op = value.def_op
        if def_op is None or def_op in selected_ops:
            continue
        selected_ops.add(def_op)
        for input_value in func_ir.ops[def_op].inputs:
            if input_value not in seen_values:
                seen_values.add(input_value)
                queue.append(input_value)
    return selected_ops


def enumerate_observable_values(func_ir: FunctionIR) -> list[str]:
    """Return structurally observable value candidates.

    Observable means a value can influence behavior outside the local
    computation region through return, control, or side effects.  In
    addition to the strict "externally-visible" set (return/branch/
    side-effect inputs), we also include every value that lies on the
    backward dataflow slice from a return value.  This widens the
    output-head action space from often-just-1 candidate (a single
    return) to dozens, which is necessary for the REINFORCE signal on
    the output policy to provide any gradient.

    Such "internal but on-the-return-slice" values are still
    semantically observable: replacing a region whose output is one of
    them changes the eventual return value.
    """
    observed: set[str] = set(func_ir.return_values)
    side_effect_opcodes = {"set_attr", "set_item", "append", "pop"}

    for op in func_ir.ops.values():
        if op.opcode == "branch":
            observed.update(op.inputs)
        elif op.opcode in side_effect_opcodes:
            observed.update(op.inputs)
        elif op.opcode == "call" and (
            op.attrs.get("effectful")
            or op.attrs.get("has_side_effect")
            or op.attrs.get("escapes")
        ):
            observed.update(op.inputs)

    # Widen with values lying on the backward slice from any
    # externally-visible value.  This lets the GNN choose region
    # outputs anywhere on the path that produces a return/effect,
    # which is what gives the output_head a meaningful action space.
    seed_set = {vid for vid in observed if vid in func_ir.values}
    if seed_set:
        slice_ops = backward_slice_by_values(func_ir, list(seed_set))
        for op_id in slice_ops:
            op = func_ir.ops.get(op_id)
            if op is None:
                continue
            observed.update(op.outputs)

    ordered = sorted(
        observed,
        key=lambda vid: _value_structural_key(func_ir, vid),
    )
    return [vid for vid in ordered if vid in func_ir.values]


def enumerate_cut_candidates(
    func_ir: FunctionIR,
    output_values: list[str],
    *,
    require_connected: bool = True,
) -> list[str]:
    """Enumerate structurally valid cut candidates for a BCIR region.

    When ``require_connected`` is True (default), each candidate is
    further filtered so that the resulting BCIR region (i.e. backward
    slice from ``output_values`` cut at ``{candidate}``) yields a
    *connected* op set under the undirected dataflow adjacency used by
    :func:`validate_boundary_region`.  This eliminates the dominant
    failure mode (~65% of invalid regions in early training were
    ``disconnected_region``) where the GNN proposes a cut that
    severs the closure into multiple islands.
    """
    closure_ops = backward_slice_by_values(func_ir, output_values)
    closure_defined: set[str] = set()
    closure_used: set[str] = set()
    for op_id in closure_ops:
        op = func_ir.ops[op_id]
        closure_defined.update(op.outputs)
        closure_used.update(op.inputs)

    closure_values = closure_defined | closure_used
    output_set = set(output_values)
    raw_candidates = [
        vid
        for vid in closure_values
        if vid not in output_set
        and vid in func_ir.values
        and _actually_truncates(func_ir, vid, closure_ops)
    ]

    if require_connected:
        filtered: list[str] = []
        for vid in raw_candidates:
            sliced = backward_slice_until_values(func_ir, output_values, [vid])
            if not sliced:
                continue
            if _is_region_connected(func_ir, sliced):
                filtered.append(vid)
        raw_candidates = filtered

    return sorted(raw_candidates, key=lambda vid: _value_structural_key(func_ir, vid))


def validate_boundary_region(
    func_ir: FunctionIR,
    region,
    *,
    min_region_ops: int = 1,
    max_region_ops: int = 24,
    max_region_inputs: int = 8,
    max_region_outputs: int = 2,
) -> RegionValidity:
    observable = set(enumerate_observable_values(func_ir))
    region_op_set = set(region.op_ids)

    selected_outputs = set(region.provenance.get("selected_output_values", []))
    effective_outputs = set(region.provenance.get("effective_output_values", []))
    selected_vs_effective_match = selected_outputs <= effective_outputs

    if not region.op_ids:
        return RegionValidity(False, "empty_region", 0, 0, 0, False, selected_vs_effective_match)
    if any(v not in observable for v in selected_outputs):
        return RegionValidity(
            False,
            "non_observable_output",
            len(region.op_ids),
            len(region.entry_values),
            len(region.exit_values),
            _is_region_connected(func_ir, region_op_set),
            selected_vs_effective_match,
        )
    is_connected = _is_region_connected(func_ir, region_op_set)
    if not is_connected:
        return RegionValidity(
            False,
            "disconnected_region",
            len(region.op_ids),
            len(region.entry_values),
            len(region.exit_values),
            False,
            selected_vs_effective_match,
        )
    if len(region.op_ids) < min_region_ops:
        return RegionValidity(
            False, "too_small", len(region.op_ids), len(region.entry_values),
            len(region.exit_values), True, selected_vs_effective_match,
        )
    if len(region.op_ids) > max_region_ops:
        return RegionValidity(
            False, "too_large", len(region.op_ids), len(region.entry_values),
            len(region.exit_values), True, selected_vs_effective_match,
        )
    if len(region.entry_values) > max_region_inputs:
        return RegionValidity(
            False, "too_many_inputs", len(region.op_ids), len(region.entry_values),
            len(region.exit_values), True, selected_vs_effective_match,
        )
    if len(region.exit_values) > max_region_outputs:
        return RegionValidity(
            False, "too_many_outputs", len(region.op_ids), len(region.entry_values),
            len(region.exit_values), True, selected_vs_effective_match,
        )
    return RegionValidity(
        True,
        "ok",
        len(region.op_ids),
        len(region.entry_values),
        len(region.exit_values),
        True,
        selected_vs_effective_match,
    )


def _actually_truncates(func_ir: FunctionIR, value_id: str, closure_ops: set[str]) -> bool:
    value = func_ir.values.get(value_id)
    if value is None or value.def_op is None:
        return False
    if value.def_op not in closure_ops:
        return False
    return True


def _op_position(func_ir: FunctionIR, op_id: str) -> tuple[int, int]:
    op = func_ir.ops[op_id]
    block_order = list(func_ir.blocks.keys()).index(op.block_id)
    op_order = func_ir.blocks[op.block_id].op_ids.index(op_id)
    return block_order, op_order


def _value_structural_key(func_ir: FunctionIR, value_id: str) -> tuple:
    value = func_ir.values.get(value_id)
    if value is None:
        return (10**9, 10**9, "", 10**9, value_id)
    if value.def_op is not None and value.def_op in func_ir.ops:
        block_order, op_order = _op_position(func_ir, value.def_op)
    else:
        use_positions = [
            _op_position(func_ir, use_op)
            for use_op in value.use_ops
            if use_op in func_ir.ops
        ]
        if use_positions:
            block_order, op_order = min(use_positions)
        else:
            block_order, op_order = (10**9, 10**9)
    return (
        block_order,
        op_order,
        value.type_hint or "",
        len(value.use_ops),
        value_id,
    )


def _is_region_connected(func_ir: FunctionIR, region_op_set: set[str]) -> bool:
    if not region_op_set:
        return False
    if len(region_op_set) == 1:
        return True

    adjacency: dict[str, set[str]] = {op_id: set() for op_id in region_op_set}
    for op_id in region_op_set:
        op = func_ir.ops[op_id]
        for value_id in op.inputs:
            value = func_ir.values.get(value_id)
            if value and value.def_op in region_op_set:
                adjacency[op_id].add(value.def_op)
                adjacency[value.def_op].add(op_id)
        for value_id in op.outputs:
            value = func_ir.values.get(value_id)
            if value is None:
                continue
            for use_op in value.use_ops:
                if use_op in region_op_set:
                    adjacency[op_id].add(use_op)
                    adjacency[use_op].add(op_id)

    start = next(iter(region_op_set))
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen == region_op_set
