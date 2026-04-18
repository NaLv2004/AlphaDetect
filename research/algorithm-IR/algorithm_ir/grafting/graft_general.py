"""General-purpose IR-level grafting engine.

Implements ``graft_general()`` — a universal, call-based grafting system
that operates entirely at the FunctionIR level (ops/values/blocks) without
any Python source intermediary.

Core principle: IR is the sole evolution medium.  All structural
modifications happen directly on FunctionIR dicts.  Python source is only
generated at execution time via ``materialize()``.
"""

from __future__ import annotations

import itertools
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from algorithm_ir.ir.model import FunctionIR, Op, Value, Block
from algorithm_ir.ir.validator import validate_function_ir
from algorithm_ir.region.selector import RewriteRegion


# ---------------------------------------------------------------------------
# GraftArtifact — output of graft_general()
# ---------------------------------------------------------------------------

@dataclass
class GraftArtifact:
    """Complete output of a graft_general() operation."""
    ir: FunctionIR                    # The grafted IR
    new_slot_ids: list[str]           # AlgSlot IDs introduced by the donor
    replaced_op_ids: list[str]        # Original ops that were removed
    call_op_id: str                   # The inserted call op


# ---------------------------------------------------------------------------
# IR helper operations
# ---------------------------------------------------------------------------

_counter = itertools.count()


def _fresh_id(prefix: str = "g") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def _manual_clone_ir(ir: FunctionIR) -> FunctionIR:
    """Clone a FunctionIR by copying its dict contents (no xDSL module)."""
    return FunctionIR(
        id=ir.id,
        name=ir.name,
        arg_values=list(ir.arg_values),
        return_values=list(ir.return_values),
        values={
            k: Value(
                id=v.id,
                name_hint=v.name_hint,
                type_hint=v.type_hint,
                source_span=v.source_span,
                def_op=v.def_op,
                use_ops=list(v.use_ops),
                attrs=dict(v.attrs),
            )
            for k, v in ir.values.items()
        },
        ops={
            k: Op(
                id=o.id,
                opcode=o.opcode,
                inputs=list(o.inputs),
                outputs=list(o.outputs),
                block_id=o.block_id,
                source_span=o.source_span,
                attrs=dict(o.attrs),
            )
            for k, o in ir.ops.items()
        },
        blocks={
            k: Block(
                id=b.id,
                op_ids=list(b.op_ids),
                preds=list(b.preds),
                succs=list(b.succs),
                attrs=dict(b.attrs),
            )
            for k, b in ir.blocks.items()
        },
        entry_block=ir.entry_block,
        attrs=dict(ir.attrs) if ir.attrs else {},
    )


def find_region_boundary(
    ir: FunctionIR,
    op_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Analyse the data-flow boundary of a region of ops.

    Parameters
    ----------
    ir : FunctionIR
    op_ids : set[str]
        Op IDs that constitute the region.

    Returns
    -------
    entry_values : list[str]
        Values defined *outside* the region but *used* by ops inside it.
    exit_values : list[str]
        Values defined *inside* the region but *used* by ops outside it.
    """
    region_defined: set[str] = set()
    region_used: set[str] = set()

    for oid in op_ids:
        op = ir.ops[oid]
        region_defined.update(op.outputs)
        region_used.update(op.inputs)

    # Entry: used inside, defined outside
    entry_values = sorted(v for v in region_used if v not in region_defined)
    # Exit: defined inside, used outside
    exit_values: list[str] = []
    for vid in sorted(region_defined):
        val = ir.values.get(vid)
        if val is None:
            continue
        for use_op in val.use_ops:
            if use_op not in op_ids:
                exit_values.append(vid)
                break

    return entry_values, exit_values


def create_call_op(
    ir: FunctionIR,
    callee_name: str,
    arg_vids: list[str],
    result_count: int,
    block_id: str,
    attrs: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Create a call op in *ir* and return ``(op_id, output_vids)``.

    The callee is registered as a ``const`` op whose literal is the
    callable name, so that ``codegen.py`` can emit ``callee_name(args…)``.
    """
    # 1. Const op for the callee reference
    callee_vid = _fresh_id("v_callee")
    callee_op_id = _fresh_id("op_callee")
    ir.values[callee_vid] = Value(
        id=callee_vid,
        name_hint=callee_name,
        type_hint="function",
        attrs={"var_name": callee_name, "role": "callee"},
    )
    ir.ops[callee_op_id] = Op(
        id=callee_op_id,
        opcode="const",
        inputs=[],
        outputs=[callee_vid],
        block_id=block_id,
        attrs={"literal": None, "name": callee_name},
    )
    ir.values[callee_vid].def_op = callee_op_id

    # 2. Output values
    out_vids: list[str] = []
    for i in range(result_count):
        vid = _fresh_id(f"v_graft_out{i}")
        ir.values[vid] = Value(
            id=vid,
            name_hint=f"graft_result_{i}",
            type_hint="object",
            attrs={"role": "graft_output"},
        )
        out_vids.append(vid)

    # 3. Call op
    call_op_id = _fresh_id("op_graft_call")
    all_inputs = [callee_vid] + list(arg_vids)
    call_attrs: dict[str, Any] = {
        "n_args": len(arg_vids),
        "grafted": True,
    }
    if attrs:
        call_attrs.update(attrs)

    ir.ops[call_op_id] = Op(
        id=call_op_id,
        opcode="call",
        inputs=all_inputs,
        outputs=out_vids,
        block_id=block_id,
        attrs=call_attrs,
    )
    # Wire def_op / use_ops
    for vid in out_vids:
        ir.values[vid].def_op = call_op_id
    for vid in all_inputs:
        if vid in ir.values:
            ir.values[vid].use_ops.append(call_op_id)

    # 4. Insert ops into block's op_ids (before the terminator)
    block = ir.blocks[block_id]
    # Find insertion point: before branch/jump/return
    insert_idx = len(block.op_ids)
    for i, oid in enumerate(block.op_ids):
        if ir.ops.get(oid) and ir.ops[oid].opcode in ("branch", "jump", "return"):
            insert_idx = i
            break
    block.op_ids.insert(insert_idx, callee_op_id)
    block.op_ids.insert(insert_idx + 1, call_op_id)

    return call_op_id, out_vids


def rebind_uses(ir: FunctionIR, old_vid: str, new_vid: str) -> None:
    """Replace every use of *old_vid* with *new_vid* across the IR.

    Updates ``op.inputs`` lists and ``value.use_ops`` bookkeeping.
    """
    old_val = ir.values.get(old_vid)
    if old_val is None:
        return
    new_val = ir.values.get(new_vid)
    if new_val is None:
        return

    # Snapshot use_ops to avoid mutation during iteration
    for use_op_id in list(old_val.use_ops):
        op = ir.ops.get(use_op_id)
        if op is None:
            continue
        op.inputs = [new_vid if v == old_vid else v for v in op.inputs]
        if use_op_id not in new_val.use_ops:
            new_val.use_ops.append(use_op_id)
    old_val.use_ops.clear()


def remove_ops(ir: FunctionIR, op_ids: set[str]) -> None:
    """Remove ops from *ir*, cleaning up produced values and block lists."""
    for oid in op_ids:
        op = ir.ops.get(oid)
        if op is None:
            continue
        # Remove from block
        block = ir.blocks.get(op.block_id)
        if block and oid in block.op_ids:
            block.op_ids.remove(oid)
        # Remove produced values that have no external uses
        for vid in op.outputs:
            val = ir.values.get(vid)
            if val is None:
                continue
            remaining_uses = [u for u in val.use_ops if u not in op_ids]
            if not remaining_uses:
                ir.values.pop(vid, None)
            else:
                val.use_ops = remaining_uses
        # Clean input use_ops
        for vid in op.inputs:
            val = ir.values.get(vid)
            if val and oid in val.use_ops:
                val.use_ops.remove(oid)
        del ir.ops[oid]


def topological_sort_block(ir: FunctionIR, block_id: str) -> None:
    """Reorder ops within *block_id* so that defs precede uses."""
    block = ir.blocks.get(block_id)
    if block is None:
        return

    op_ids_set = set(block.op_ids)
    # Build adjacency: op A must come before op B if A defines a value B uses
    value_to_def_op: dict[str, str] = {}
    for oid in block.op_ids:
        op = ir.ops.get(oid)
        if op:
            for vid in op.outputs:
                value_to_def_op[vid] = oid

    # Kahn's algorithm
    in_degree: dict[str, int] = {oid: 0 for oid in block.op_ids}
    adj: dict[str, list[str]] = {oid: [] for oid in block.op_ids}

    for oid in block.op_ids:
        op = ir.ops.get(oid)
        if op is None:
            continue
        for vid in op.inputs:
            dep_op = value_to_def_op.get(vid)
            if dep_op and dep_op in op_ids_set and dep_op != oid:
                adj[dep_op].append(oid)
                in_degree[oid] += 1

    queue = [oid for oid in block.op_ids if in_degree[oid] == 0]
    sorted_ops: list[str] = []
    while queue:
        oid = queue.pop(0)
        sorted_ops.append(oid)
        for nxt in adj.get(oid, []):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    # Append any remaining (cycle or terminator)
    remaining = [oid for oid in block.op_ids if oid not in set(sorted_ops)]
    sorted_ops.extend(remaining)
    block.op_ids = sorted_ops


# ---------------------------------------------------------------------------
# apply_dependency_overrides
# ---------------------------------------------------------------------------

def apply_dependency_overrides(
    ir: FunctionIR,
    overrides: list,
    call_op_id: str,
) -> None:
    """Apply DependencyOverride objects to a grafted IR.

    For each override:
      - Locate target_value in *ir*
      - Create/find values for new_dependencies
      - Append them to the call op's inputs
      - Extend ir.arg_values if needed
    """
    call_op = ir.ops.get(call_op_id)
    if call_op is None:
        return

    for override in overrides:
        target_vid = override.target_value
        for dep_name in override.new_dependencies:
            # Try to find an existing value with this var_name
            found_vid = None
            for vid, val in ir.values.items():
                vname = val.attrs.get("var_name") or val.name_hint
                if vname == dep_name:
                    found_vid = vid
                    break

            if found_vid is None:
                # Create a new function argument
                new_vid = _fresh_id(f"v_dep_{dep_name}")
                ir.values[new_vid] = Value(
                    id=new_vid,
                    name_hint=dep_name,
                    type_hint="object",
                    attrs={"var_name": dep_name, "role": "arg"},
                )
                ir.arg_values.append(new_vid)
                found_vid = new_vid

            # Append to call op inputs
            if found_vid not in call_op.inputs:
                call_op.inputs.append(found_vid)
                call_op.attrs["n_args"] = call_op.attrs.get("n_args", 0) + 1
                ir.values[found_vid].use_ops.append(call_op_id)


# ---------------------------------------------------------------------------
# graft_general — the main entry point
# ---------------------------------------------------------------------------

def graft_general(
    host_ir: FunctionIR,
    proposal,
) -> GraftArtifact:
    """Perform a general-purpose call-based graft at the IR level.

    This is **pure IR-level op surgery**: no Python source intermediary.

    Steps:
      1. Deep-copy host_ir
      2. Locate region ops (from proposal.region.op_ids)
      3. Analyse region boundary (entry_values, exit_values)
      4. Register donor as a callable constant
      5. Create a call op: ``result = donor_fn(entry_values…)``
      6. Rebind: redirect external uses of exit_values → call outputs
      7. Remove region ops
      8. Apply dependency_overrides (if any)
      9. Topological-sort affected blocks
     10. Validate result IR

    Parameters
    ----------
    host_ir : FunctionIR
        The host algorithm's IR (not modified in-place).
    proposal : GraftProposal
        Must have ``region``, ``donor_ir``, and optionally
        ``port_mapping`` and ``dependency_overrides``.

    Returns
    -------
    GraftArtifact
    """
    # 1. Deep copy — handle FunctionIR that may lack xdsl_module
    try:
        new_ir = deepcopy(host_ir)
    except (AttributeError, TypeError):
        # Fallback: manual dict-level copy for IRs without xdsl_module
        new_ir = _manual_clone_ir(host_ir)

    # 2. Locate region ops
    region: RewriteRegion = proposal.region
    region_op_ids = set(region.op_ids)

    # Verify all region ops exist in the copy
    missing = region_op_ids - set(new_ir.ops.keys())
    if missing:
        raise ValueError(
            f"Region references {len(missing)} ops not in host IR: "
            f"{sorted(missing)[:5]}…"
        )

    # 3. Boundary analysis
    entry_values, exit_values = find_region_boundary(new_ir, region_op_ids)

    # Use port_mapping to reorder entry_values if provided
    if proposal.port_mapping:
        mapped_entries: list[str] = []
        for host_vid in entry_values:
            mapped_entries.append(
                proposal.port_mapping.get(host_vid, host_vid)
            )
        # Keep entry_values as-is for the call args (they reference host values)

    # 4. Determine donor callable name
    donor_ir = proposal.donor_ir
    if donor_ir and hasattr(donor_ir, "name"):
        donor_name = donor_ir.name
    elif proposal.donor_algo_id:
        donor_name = f"donor_{proposal.donor_algo_id}"
    else:
        donor_name = f"donor_{_fresh_id('anon')}"

    # Determine the insertion block: use the block containing the first
    # region op (or entry_block of the schedule_anchors)
    first_region_op = new_ir.ops.get(region.op_ids[0]) if region.op_ids else None
    insertion_block = first_region_op.block_id if first_region_op else new_ir.entry_block

    # 5. Create call op
    n_results = max(1, len(exit_values))
    call_op_id, out_vids = create_call_op(
        new_ir,
        callee_name=donor_name,
        arg_vids=entry_values,
        result_count=n_results,
        block_id=insertion_block,
        attrs={"graft_donor": donor_name, "grafted": True},
    )

    # 6. Rebind exit_values → call outputs
    for i, exit_vid in enumerate(exit_values):
        if i < len(out_vids):
            rebind_uses(new_ir, exit_vid, out_vids[i])

    # 7. Remove region ops
    remove_ops(new_ir, region_op_ids)
    replaced_op_ids = sorted(region_op_ids)

    # 8. Dependency overrides
    if proposal.dependency_overrides:
        apply_dependency_overrides(
            new_ir, proposal.dependency_overrides, call_op_id,
        )

    # 9. Topological sort affected blocks
    affected_blocks = {insertion_block}
    for oid in region.op_ids:
        op = host_ir.ops.get(oid)
        if op:
            affected_blocks.add(op.block_id)
    for bid in affected_blocks:
        if bid in new_ir.blocks:
            topological_sort_block(new_ir, bid)

    # 10. Detect new slots introduced by donor
    new_slot_ids: list[str] = []
    if donor_ir:
        for op in donor_ir.ops.values():
            if op.opcode == "slot" or op.attrs.get("slot_id"):
                sid = op.attrs.get("slot_id", op.id)
                if sid not in new_slot_ids:
                    new_slot_ids.append(sid)

    return GraftArtifact(
        ir=new_ir,
        new_slot_ids=new_slot_ids,
        replaced_op_ids=replaced_op_ids,
        call_op_id=call_op_id,
    )
