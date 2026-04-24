"""General-purpose IR-level grafting engine.

Implements ``graft_general()`` — a universal, **inline** grafting system
that operates entirely at the FunctionIR level (ops/values/blocks) without
any Python source intermediary.

Core principle: IR is the sole evolution medium.  All structural
modifications happen directly on FunctionIR dicts.  Python source is only
generated at execution time via ``materialize()``.

Inline grafting means donor ops are cloned directly into the host IR,
so every op is visible and mutable by subsequent evolution.  No opaque
``call`` ops are created.
"""

from __future__ import annotations

import itertools
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from algorithm_ir.ir.model import FunctionIR, Op, Value, Block
from algorithm_ir.ir.validator import rebuild_def_use, validate_function_ir
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
    inlined_op_ids: list[str]         # Donor ops inlined into host
    id_map: dict[str, str] = field(default_factory=dict)  # donor_id → host_id
    # Provenance from the typed-bipartite binding layer.  ``None`` when
    # the legacy name-hint / positional matcher was used (e.g. donor had
    # no args or every candidate was lattice-incompatible).
    typed_binding: dict[str, object] | None = None


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


_TERMINATOR_OPCODES = frozenset({"branch", "jump", "return"})


# ---------------------------------------------------------------------------
# Inline-graft helpers
# ---------------------------------------------------------------------------

def clone_donor_ir(
    donor_ir: FunctionIR,
) -> tuple[dict[str, str], dict[str, Value], dict[str, Op], dict[str, Block]]:
    """Clone all donor values, ops, and blocks with fresh IDs.

    Returns
    -------
    id_map : dict[str, str]
        Mapping from old donor IDs (values, ops, blocks) to new IDs.
    cloned_values : dict[str, Value]
        New Value objects (excluding arg values — those will be rebound).
    cloned_ops : dict[str, Op]
        New Op objects with remapped inputs/outputs/block_ids.
    cloned_blocks : dict[str, Block]
        New Block objects with remapped op_ids/preds/succs.
    """
    id_map: dict[str, str] = {}

    # Map all old IDs → fresh IDs
    for vid in donor_ir.values:
        id_map[vid] = _fresh_id("v_inl")
    for oid in donor_ir.ops:
        id_map[oid] = _fresh_id("op_inl")
    for bid in donor_ir.blocks:
        id_map[bid] = _fresh_id("blk_inl")

    def _remap(old_id: str) -> str:
        return id_map.get(old_id, old_id)

    # Clone values (exclude arg values — they'll be rebound to host values)
    arg_set = set(donor_ir.arg_values)
    cloned_values: dict[str, Value] = {}
    for vid, val in donor_ir.values.items():
        if vid in arg_set:
            continue  # will be rebound, don't clone
        new_vid = _remap(vid)
        cloned_values[new_vid] = Value(
            id=new_vid,
            name_hint=val.name_hint,
            type_hint=val.type_hint,
            source_span=val.source_span,
            def_op=_remap(val.def_op) if val.def_op else None,
            use_ops=[_remap(u) for u in val.use_ops],
            attrs=dict(val.attrs),
        )

    # Clone ops
    cloned_ops: dict[str, Op] = {}
    for oid, op in donor_ir.ops.items():
        new_oid = _remap(oid)
        cloned_ops[new_oid] = Op(
            id=new_oid,
            opcode=op.opcode,
            inputs=[_remap(v) for v in op.inputs],
            outputs=[_remap(v) for v in op.outputs],
            block_id=_remap(op.block_id),
            source_span=op.source_span,
            attrs=dict(op.attrs),
        )
        # Mark as grafted
        cloned_ops[new_oid].attrs["grafted"] = True

    # Clone blocks
    cloned_blocks: dict[str, Block] = {}
    for bid, block in donor_ir.blocks.items():
        new_bid = _remap(bid)
        cloned_blocks[new_bid] = Block(
            id=new_bid,
            op_ids=[_remap(o) for o in block.op_ids],
            preds=[_remap(p) for p in block.preds],
            succs=[_remap(s) for s in block.succs],
            attrs=dict(block.attrs),
        )

    return id_map, cloned_values, cloned_ops, cloned_blocks


def bind_donor_args_to_host_values(
    cloned_ops: dict[str, Op],
    cloned_values: dict[str, Value],
    donor_arg_new_vids: list[str],
    host_arg_vids: list[str],
) -> None:
    """Replace references to donor arg values with host values in cloned ops.

    For each donor arg, all uses in cloned ops are rebound to the
    corresponding host value.
    """
    for i, donor_vid in enumerate(donor_arg_new_vids):
        if i >= len(host_arg_vids):
            break
        host_vid = host_arg_vids[i]
        # Replace in all cloned op inputs
        for op in cloned_ops.values():
            op.inputs = [host_vid if v == donor_vid else v for v in op.inputs]
        # Clean up — donor arg values were not cloned (excluded in clone_donor_ir)


def bind_donor_returns_to_host_exits(
    ir: FunctionIR,
    exit_values: list[str],
    donor_result_vids: list[str],
) -> None:
    """Rebind host exit values to donor result values.

    After inlining, the donor's return value(s) should replace the
    region's exit value(s) in the host IR.
    """
    for i, exit_vid in enumerate(exit_values):
        if i < len(donor_result_vids):
            rebind_uses(ir, exit_vid, donor_result_vids[i])


def _inline_multiblock_donor(
    ir: FunctionIR,
    host_block_id: str,
    region_op_ids: set[str],
    cloned_ops: dict[str, Op],
    cloned_values: dict[str, Value],
    cloned_blocks: dict[str, Block],
    donor_entry_block_id: str,
    id_map: dict[str, str],
    donor_ir: FunctionIR,
) -> None:
    """Inline a multi-block donor into the host IR.

    Splits the host block at the region boundary into pre_block and
    post_block, then inserts donor blocks between them with proper
    CFG connections.
    """
    host_block = ir.blocks[host_block_id]

    # Find region position
    region_start_idx = None
    region_end_idx = None
    for i, oid in enumerate(host_block.op_ids):
        if oid in region_op_ids:
            if region_start_idx is None:
                region_start_idx = i
            region_end_idx = i

    if region_start_idx is None:
        region_start_idx = len(host_block.op_ids)
        region_end_idx = region_start_idx - 1

    pre_ops = host_block.op_ids[:region_start_idx]
    post_ops = host_block.op_ids[region_end_idx + 1:]

    # Create post_block for ops after region
    post_block_id = _fresh_id("blk_post")
    post_block = Block(
        id=post_block_id,
        op_ids=list(post_ops),
        preds=[],
        succs=list(host_block.succs),
        attrs={},
    )

    # Update ops in post_block to reference new block_id
    for oid in post_ops:
        if oid in ir.ops:
            ir.ops[oid].block_id = post_block_id

    # Update successors of host_block's original successors to reference post_block
    for succ_bid in host_block.succs:
        succ_block = ir.blocks.get(succ_bid)
        if succ_block:
            succ_block.preds = [
                post_block_id if p == host_block_id else p
                for p in succ_block.preds
            ]

    # Trim host_block (now pre_block) to just pre_ops + jump to donor entry
    host_block.op_ids = list(pre_ops)
    host_block.succs = [donor_entry_block_id]

    # Add jump op from pre_block to donor entry
    jump_op_id = _fresh_id("op_jump_to_donor")
    ir.ops[jump_op_id] = Op(
        id=jump_op_id,
        opcode="jump",
        inputs=[],
        outputs=[],
        block_id=host_block_id,
        attrs={"target": donor_entry_block_id, "grafted": True},
    )
    host_block.op_ids.append(jump_op_id)

    # Find donor exit blocks (blocks that originally had return ops)
    donor_exit_blocks: list[str] = []
    for bid, block in cloned_blocks.items():
        # A donor exit block is one that had its return op removed
        # (now it has no terminator, or its last op was a return that was removed)
        has_terminator = False
        for oid in block.op_ids:
            if oid in cloned_ops and cloned_ops[oid].opcode in _TERMINATOR_OPCODES:
                has_terminator = True
                break
        if not has_terminator:
            donor_exit_blocks.append(bid)

    # Add jump from each donor exit block to post_block
    for exit_bid in donor_exit_blocks:
        exit_jump_id = _fresh_id("op_jump_to_post")
        cloned_ops[exit_jump_id] = Op(
            id=exit_jump_id,
            opcode="jump",
            inputs=[],
            outputs=[],
            block_id=exit_bid,
            attrs={"target": post_block_id, "grafted": True},
        )
        cloned_blocks[exit_bid].op_ids.append(exit_jump_id)
        cloned_blocks[exit_bid].succs.append(post_block_id)
        post_block.preds.append(exit_bid)

    # Set donor entry block pred
    cloned_blocks[donor_entry_block_id].preds.append(host_block_id)

    # Add all donor blocks, ops, values to host IR
    ir.ops.update(cloned_ops)
    ir.values.update(cloned_values)
    ir.blocks.update(cloned_blocks)
    ir.blocks[post_block_id] = post_block


def _apply_inline_dependency_overrides(
    ir: FunctionIR,
    overrides: list,
    inlined_op_ids: list[str],
) -> None:
    """Apply DependencyOverride objects for inlined graft.

    For each override, find or create the dependency value and add it
    to the IR's arg_values if needed.
    """
    for override in overrides:
        for dep_name in override.new_dependencies:
            found_vid = None
            for vid, val in ir.values.items():
                vname = val.attrs.get("var_name") or val.name_hint
                if vname == dep_name:
                    found_vid = vid
                    break
            if found_vid is None:
                new_vid = _fresh_id(f"v_dep_{dep_name}")
                ir.values[new_vid] = Value(
                    id=new_vid,
                    name_hint=dep_name,
                    type_hint="object",
                    attrs={"var_name": dep_name, "role": "arg"},
                )
                ir.arg_values.append(new_vid)


def topological_sort_block(ir: FunctionIR, block_id: str) -> None:
    """Reorder ops within *block_id* so that defs precede uses.

    Terminators (branch/jump/return) are always placed last to preserve
    correct block structure after grafting inserts new ops.
    """
    block = ir.blocks.get(block_id)
    if block is None:
        return

    # Separate terminators from regular ops — terminators must be last
    terminators = [
        oid for oid in block.op_ids
        if ir.ops.get(oid) and ir.ops[oid].opcode in _TERMINATOR_OPCODES
    ]
    non_term = [
        oid for oid in block.op_ids
        if oid not in set(terminators)
    ]

    op_ids_set = set(non_term)
    # Build adjacency: op A must come before op B if A defines a value B uses
    value_to_def_op: dict[str, str] = {}
    for oid in non_term:
        op = ir.ops.get(oid)
        if op:
            for vid in op.outputs:
                value_to_def_op[vid] = oid

    # Kahn's algorithm on non-terminators only
    in_degree: dict[str, int] = {oid: 0 for oid in non_term}
    adj: dict[str, list[str]] = {oid: [] for oid in non_term}

    for oid in non_term:
        op = ir.ops.get(oid)
        if op is None:
            continue
        for vid in op.inputs:
            dep_op = value_to_def_op.get(vid)
            if dep_op and dep_op in op_ids_set and dep_op != oid:
                adj[dep_op].append(oid)
                in_degree[oid] += 1

    queue = [oid for oid in non_term if in_degree[oid] == 0]
    sorted_ops: list[str] = []
    while queue:
        oid = queue.pop(0)
        sorted_ops.append(oid)
        for nxt in adj.get(oid, []):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    # Append any remaining non-terminators (e.g. cycles — preserve order)
    remaining = [oid for oid in non_term if oid not in set(sorted_ops)]
    sorted_ops.extend(remaining)
    # Always append terminators last
    sorted_ops.extend(terminators)
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
    """Perform a general-purpose **inline** graft at the IR level.

    This is **pure IR-level op surgery**: no Python source intermediary.
    Donor ops are cloned directly into the host IR so every op is visible
    and mutable by subsequent evolution.

    Steps:
      1. Deep-copy host_ir
      2. Locate region ops (from proposal.region.op_ids)
      3. Analyse region boundary (entry_values, exit_values)
      4. Clone donor IR with fresh IDs
      5. Map donor args → host function args
      6. Find donor return op(s), extract return values
      7. Remove donor return ops from cloned blocks
      8. Inline donor blocks into host (single-block or multi-block)
      9. Rebind: redirect external uses of exit_values → donor result values
     10. Remove region ops
     11. Apply dependency_overrides (if any)
     12. Topological-sort affected blocks
     13. Detect new slots from donor

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
    # 1. Deep copy
    try:
        new_ir = deepcopy(host_ir)
    except (AttributeError, TypeError):
        new_ir = _manual_clone_ir(host_ir)
    rebuild_def_use(new_ir)

    # 2. Locate region ops
    region: RewriteRegion = proposal.region
    region_op_ids = set(region.op_ids)

    missing = region_op_ids - set(new_ir.ops.keys())
    if missing:
        raise ValueError(
            f"Region references {len(missing)} ops not in host IR: "
            f"{sorted(missing)[:5]}…"
        )

    # 3. Boundary analysis
    entry_values, exit_values = find_region_boundary(new_ir, region_op_ids)
    host_contract = proposal.contract
    if host_contract is not None:
        if getattr(host_contract, "normalized_input_ports", None):
            entry_values = list(host_contract.normalized_input_ports)
        if getattr(host_contract, "normalized_output_ports", None):
            exit_values = list(host_contract.normalized_output_ports)

    # 4. Clone donor IR with fresh IDs
    donor_ir = proposal.donor_ir
    if donor_ir is None:
        # No donor — just remove the region ops
        remove_ops(new_ir, region_op_ids)
        return GraftArtifact(
            ir=new_ir,
            new_slot_ids=[],
            replaced_op_ids=sorted(region_op_ids),
            inlined_op_ids=[],
            id_map={},
        )

    id_map, cloned_values, cloned_ops, cloned_blocks = clone_donor_ir(donor_ir)

    # 5. Map donor args → host function args
    donor_arg_new_vids = [id_map[v] for v in donor_ir.arg_values]

    # Generic IR-builder name_hints that must NOT be used for
    # cross-IR value matching — they are semantically meaningless
    # and collide across every IR (every binary op produces a value
    # called "binary", etc.).
    _GENERIC_IR_NAMES = frozenset({
        "binary", "unary", "compare", "call", "get_attr", "get_item",
        "iter_init", "iter_next", "iter_has_next", "phi", "const",
    })

    if proposal.port_mapping:
        host_arg_vids = [
            proposal.port_mapping.get(v, v) for v in entry_values
        ]
    elif host_contract is not None and donor_ir.arg_values and (
        len(host_contract.normalized_input_ports) == len(_function_port_signature(donor_ir)["inputs"])
        and len(host_contract.normalized_output_ports) == len(_function_port_signature(donor_ir)["outputs"])
    ):
        # Strict contract path: only used when arities match exactly.
        # When mismatched, we fall through to the name-hint matching
        # below rather than raising — that recovers the majority of
        # otherwise-failing port bindings without imposing a graft
        # pattern prior.
        donor_sig = _function_port_signature(donor_ir)
        host_inputs = list(host_contract.normalized_input_ports)
        host_outputs = list(host_contract.normalized_output_ports)
        sig_in_compatible = all(
            _strip_value_id(h) == _strip_value_id(d)
            for h, d in zip(host_contract.port_signature["inputs"], donor_sig["inputs"])
        )
        sig_out_compatible = all(
            _strip_value_id(h) == _strip_value_id(d)
            for h, d in zip(host_contract.port_signature["outputs"], donor_sig["outputs"])
        )
        if sig_in_compatible and sig_out_compatible:
            host_arg_vids = host_inputs
        else:
            host_contract = None  # fall through to name-hint matching
            host_arg_vids = None
    else:
        host_arg_vids = None

    # ------------------------------------------------------------------
    # Typed bipartite binding (lattice-driven, Hungarian-optimal).
    # Inserted between the strict-contract path and the legacy
    # name-hint matcher.  When the lattice rejects every candidate for
    # any donor arg, this layer returns ``None`` and we fall through to
    # the legacy matcher; that preserves backwards compatibility while
    # still upgrading the majority of bindings.
    # ------------------------------------------------------------------
    if host_arg_vids is None and donor_ir.arg_values:
        try:
            from algorithm_ir.grafting.typed_binding import bind_typed
            tb_result = bind_typed(
                donor_ir, new_ir, region_op_ids, require_feasible=True,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("typed_binding raised: %r", exc)
            tb_result = None
        if tb_result is not None and tb_result.feasible:
            host_arg_vids = [tb_result.mapping[d] for d in donor_ir.arg_values]
            # Stash diagnostics so callers can inspect what the lattice
            # decided.  (graft_general returns this via GraftArtifact's
            # provenance dict; see below.)
            _typed_bind_diag = tb_result.diagnostics
            _typed_bind_used = True
            _typed_bind_cost = tb_result.cost
        else:
            _typed_bind_diag = None
            _typed_bind_used = False
            _typed_bind_cost = None
    else:
        _typed_bind_diag = None
        _typed_bind_used = False
        _typed_bind_cost = None

    if host_arg_vids is None and donor_ir.arg_values:
        # Name-hint matching: for each donor arg, find the host value
        # whose name_hint matches.  This is critical for trimmed donors
        # where arg_values are a subset of the original function args —
        # positional mapping (old approach) binds them incorrectly.
        host_name_to_vid: dict[str, str] = {}
        for vid in new_ir.arg_values:
            val = new_ir.values.get(vid)
            if val:
                hints = [h for h in [
                    val.name_hint,
                    val.attrs.get("var_name"),
                    val.attrs.get("name"),
                ] if h and h not in _GENERIC_IR_NAMES]
                for h in hints:
                    host_name_to_vid[h] = vid

        # Also index all host values by name (not just args) so that
        # intermediate entry values can be matched.
        for vid, val in new_ir.values.items():
            hints = [h for h in [
                val.name_hint,
                val.attrs.get("var_name"),
                val.attrs.get("name"),
            ] if h and h not in _GENERIC_IR_NAMES]
            for h in hints:
                if h not in host_name_to_vid:
                    host_name_to_vid[h] = vid

        host_arg_vids: list[str] = []
        n_donor = len(donor_ir.arg_values)
        n_host = len(new_ir.arg_values)

        for di, donor_vid in enumerate(donor_ir.arg_values):
            donor_val = donor_ir.values.get(donor_vid)
            matched_vid: str | None = None

            if donor_val:
                # Try matching by name hints (skip generic IR names)
                for hint in [
                    donor_val.name_hint,
                    donor_val.attrs.get("var_name"),
                    donor_val.attrs.get("name"),
                ]:
                    if hint and hint not in _GENERIC_IR_NAMES and hint in host_name_to_vid:
                        matched_vid = host_name_to_vid[hint]
                        break

            if matched_vid is None:
                # Fallback: positional mapping (clamped to host arg count)
                if di < n_host:
                    matched_vid = new_ir.arg_values[di]
                else:
                    # Create a const None for unmatchable extra args
                    hint = donor_val.name_hint if donor_val else f"extra_{di}"
                    new_vid = _fresh_id("v_extra")
                    new_ir.values[new_vid] = Value(
                        id=new_vid,
                        name_hint=hint,
                        type_hint="object",
                        attrs={"role": "const", "var_name": hint},
                    )
                    const_oid = _fresh_id("op_extra_const")
                    new_ir.ops[const_oid] = Op(
                        id=const_oid,
                        opcode="const",
                        inputs=[],
                        outputs=[new_vid],
                        block_id=new_ir.entry_block,
                        attrs={"literal": None, "name": hint, "grafted": True},
                    )
                    new_ir.values[new_vid].def_op = const_oid
                    host_block = new_ir.blocks.get(new_ir.entry_block)
                    if host_block:
                        host_block.op_ids.insert(0, const_oid)
                    matched_vid = new_vid

            host_arg_vids.append(matched_vid)
    elif host_arg_vids is None:
        host_arg_vids = list(entry_values)

    bind_donor_args_to_host_values(
        cloned_ops, cloned_values, donor_arg_new_vids, host_arg_vids,
    )

    # Safety: if `host_arg_vids` is shorter than `donor_arg_new_vids`,
    # the leftover donor args have no host binding and any cloned op
    # that uses them will produce a dangling reference.  Materialise
    # one ``const None`` per missing arg and rebind, so we get a
    # syntactically valid graft (which then runs and produces a real
    # signal — better than raising and giving the policy zero
    # information about the structural choice).
    if len(host_arg_vids) < len(donor_arg_new_vids):
        entry_block = new_ir.blocks.get(new_ir.entry_block)
        for i in range(len(host_arg_vids), len(donor_arg_new_vids)):
            donor_vid = donor_arg_new_vids[i]
            fill_vid = _fresh_id("v_argfill")
            new_ir.values[fill_vid] = Value(
                id=fill_vid,
                name_hint=f"argfill_{i}",
                type_hint="object",
                attrs={"role": "const", "var_name": "None"},
            )
            const_oid = _fresh_id("op_argfill_const")
            new_ir.ops[const_oid] = Op(
                id=const_oid,
                opcode="const",
                inputs=[],
                outputs=[fill_vid],
                block_id=new_ir.entry_block,
                attrs={"literal": None, "name": "None", "grafted": True},
            )
            new_ir.values[fill_vid].def_op = const_oid
            if entry_block:
                entry_block.op_ids.insert(0, const_oid)
            for op in cloned_ops.values():
                op.inputs = [fill_vid if v == donor_vid else v for v in op.inputs]

    # 6. Find donor return ops, extract return values
    donor_return_op_ids: list[str] = []
    donor_result_vids: list[str] = []
    for oid, op in list(cloned_ops.items()):
        if op.opcode == "return":
            donor_return_op_ids.append(oid)
            donor_result_vids.extend(op.inputs)

    # 7. Remove return ops from cloned blocks
    for ret_oid in donor_return_op_ids:
        ret_op = cloned_ops[ret_oid]
        block_id = ret_op.block_id
        if block_id in cloned_blocks:
            cloned_blocks[block_id].op_ids = [
                o for o in cloned_blocks[block_id].op_ids if o != ret_oid
            ]
        del cloned_ops[ret_oid]

    # 8. Inline donor blocks into host
    first_region_op = new_ir.ops.get(region.op_ids[0]) if region.op_ids else None
    host_block_id = first_region_op.block_id if first_region_op else new_ir.entry_block

    donor_entry_block_id = id_map[donor_ir.entry_block]

    if len(cloned_blocks) == 1:
        # Simple case: single-block donor — inline ops into host block
        donor_block = cloned_blocks[donor_entry_block_id]
        donor_op_ids = donor_block.op_ids

        # Find region position in host block
        host_block = new_ir.blocks[host_block_id]
        region_start_idx = None
        region_end_idx = None
        for i, oid in enumerate(host_block.op_ids):
            if oid in region_op_ids:
                if region_start_idx is None:
                    region_start_idx = i
                region_end_idx = i

        if region_start_idx is None:
            region_start_idx = len(host_block.op_ids)
            region_end_idx = region_start_idx - 1

        pre_ops = host_block.op_ids[:region_start_idx]
        post_ops = host_block.op_ids[region_end_idx + 1:]

        # Update donor ops block_id
        for oid in donor_op_ids:
            cloned_ops[oid].block_id = host_block_id

        # Insert: pre + donor + post
        host_block.op_ids = pre_ops + donor_op_ids + post_ops

        # Add cloned ops/values to host IR
        new_ir.ops.update(cloned_ops)
        new_ir.values.update(cloned_values)
        # Don't add donor blocks (merged into host block)

    else:
        # Multi-block donor: split host block, insert donor blocks
        _inline_multiblock_donor(
            new_ir, host_block_id, region_op_ids,
            cloned_ops, cloned_values, cloned_blocks,
            donor_entry_block_id, id_map, donor_ir,
        )

    # 9. Rebind exit_values → donor result values
    bind_donor_returns_to_host_exits(
        new_ir, exit_values, donor_result_vids,
    )

    # 10. Remove region ops
    remove_ops(new_ir, region_op_ids)
    replaced_op_ids = sorted(region_op_ids)

    # 11. Dependency overrides
    if proposal.dependency_overrides:
        # For inline graft, dependency overrides add values to the IR
        # that the inlined ops can reference
        _apply_inline_dependency_overrides(
            new_ir, proposal.dependency_overrides,
            [oid for oid in cloned_ops if oid in new_ir.ops],
        )

    # 12. Topological sort affected blocks
    affected_blocks = {host_block_id}
    for oid in region.op_ids:
        op = host_ir.ops.get(oid)
        if op:
            affected_blocks.add(op.block_id)
    # Also sort any donor blocks that were added
    for bid in list(new_ir.blocks.keys()):
        if bid in {id_map.get(b) for b in donor_ir.blocks}:
            affected_blocks.add(bid)
    for bid in affected_blocks:
        if bid in new_ir.blocks:
            topological_sort_block(new_ir, bid)

    # 12b. Auto-repair dangling value references — values referenced
    # by ops but not defined by any op in the IR and not in
    # arg_values.  These arise when port binding can't fully resolve
    # donor-internal references (e.g. cloned-but-deleted ops, name
    # collisions, etc.).  We materialise one ``const None`` per unique
    # dangling vid and rewrite all uses so the graft is syntactically
    # valid and runs.  Crashing-at-execution gives the policy a real
    # signal (-0.5 invalid reward) about the structural choice; raising
    # at construction time only labels every such graft "invalid"
    # without any per-sample variability that the RL signal can use.
    arg_set = set(new_ir.arg_values)
    dangling_vids: set[str] = set()
    for op in list(new_ir.ops.values()):
        for inp_vid in op.inputs:
            if inp_vid in arg_set:
                continue
            val = new_ir.values.get(inp_vid)
            if val is None:
                dangling_vids.add(inp_vid)
                continue
            if val.def_op and val.def_op in new_ir.ops:
                continue
            dangling_vids.add(inp_vid)

    if dangling_vids:
        entry_block = new_ir.blocks.get(new_ir.entry_block)
        rewrite: dict[str, str] = {}
        for d_vid in dangling_vids:
            fill_vid = _fresh_id("v_repair")
            new_ir.values[fill_vid] = Value(
                id=fill_vid,
                name_hint="repair_none",
                type_hint="object",
                attrs={"role": "const", "var_name": "None"},
            )
            const_oid = _fresh_id("op_repair_const")
            new_ir.ops[const_oid] = Op(
                id=const_oid,
                opcode="const",
                inputs=[],
                outputs=[fill_vid],
                block_id=new_ir.entry_block,
                attrs={"literal": None, "name": "None", "grafted": True},
            )
            new_ir.values[fill_vid].def_op = const_oid
            if entry_block:
                entry_block.op_ids.insert(0, const_oid)
            rewrite[d_vid] = fill_vid
        for op in new_ir.ops.values():
            if any(v in rewrite for v in op.inputs):
                op.inputs = [rewrite.get(v, v) for v in op.inputs]

    # 13. Detect new slots from donor
    new_slot_ids: list[str] = []
    if donor_ir:
        for op in donor_ir.ops.values():
            if op.opcode == "slot" or op.attrs.get("slot_id"):
                sid = op.attrs.get("slot_id", op.id)
                if sid not in new_slot_ids:
                    new_slot_ids.append(sid)

    inlined_op_ids = [oid for oid in cloned_ops if oid in new_ir.ops]

    rebuild_def_use(new_ir)
    errors = validate_function_ir(new_ir)
    if errors:
        raise ValueError(f"Invalid grafted IR: {errors[:8]}")

    return GraftArtifact(
        ir=new_ir,
        new_slot_ids=new_slot_ids,
        replaced_op_ids=replaced_op_ids,
        inlined_op_ids=inlined_op_ids,
        id_map=id_map,
        typed_binding=(
            {
                "used": True,
                "cost": _typed_bind_cost,
                "diagnostics": _typed_bind_diag,
            }
            if _typed_bind_used else None
        ),
    )


def _function_port_signature(func_ir: FunctionIR) -> dict[str, list[dict[str, Any]]]:
    return {
        "inputs": [_function_value_signature(func_ir, vid, "input") for vid in func_ir.arg_values],
        "outputs": [_function_value_signature(func_ir, vid, "output") for vid in func_ir.return_values],
    }


def _function_value_signature(
    func_ir: FunctionIR,
    value_id: str,
    direction: str,
) -> dict[str, Any]:
    value = func_ir.values.get(value_id)
    if value is None:
        return {
            "direction": direction,
            "type_hint": None,
            "n_uses": 0,
            "is_control_related": False,
            "is_effect_related": False,
        }
    return {
        "direction": direction,
        "type_hint": value.type_hint,
        "n_uses": len(value.use_ops),
        "is_control_related": any(
            func_ir.ops[use_op].opcode == "branch"
            for use_op in value.use_ops
            if use_op in func_ir.ops
        ),
        "is_effect_related": any(
            func_ir.ops[use_op].opcode in {"set_attr", "set_item", "append", "pop"}
            or (
                func_ir.ops[use_op].opcode == "call"
                and (
                    func_ir.ops[use_op].attrs.get("effectful")
                    or func_ir.ops[use_op].attrs.get("has_side_effect")
                    or func_ir.ops[use_op].attrs.get("escapes")
                )
            )
            for use_op in value.use_ops
            if use_op in func_ir.ops
        ),
    }


def _strip_value_id(signature_entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in signature_entry.items() if k != "value_id"}
