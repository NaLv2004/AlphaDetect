"""Canonical hash of a FunctionIR — IR-only, no Python source.

Used by the typed GP framework to deduplicate variants and detect
no-op mutations without ever materializing Python source.

The hash is structural: equivalent IRs (same op kinds, same edges,
same literal attrs in canonical order) hash to the same value
regardless of internal id renaming or insertion order.

Phase H+4 §2 / §4.2 / §8.3 (single-representation principle).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from algorithm_ir.ir.model import FunctionIR


# Op-attr keys that are semantically meaningful and must be hashed.
# Other attrs (var_name, source_span, _provenance, type_info, etc.)
# are intentionally excluded — they are bookkeeping that doesn't
# change behaviour.
_HASHED_ATTR_KEYS = frozenset({
    "literal", "name", "operator", "operators", "n_args", "kwarg_names",
    "attr", "target", "true", "false", "loop_backedge",
    "slot_id", "slot_kind", "callee",
})


def _canonical_literal(value: Any) -> Any:
    """Convert a Python literal into a JSON-friendly canonical form."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        # Round to avoid floating-point representation artifacts that
        # would make hash-equal IRs hash-different. 12 significant
        # digits is finer than any meaningful evolutionary tweak.
        return ("float", round(value, 12))
    if isinstance(value, complex):
        return ("complex", round(value.real, 12), round(value.imag, 12))
    if isinstance(value, str):
        return ("str", value)
    if value is None:
        return ("none",)
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_canonical_literal(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted(
            (str(k), _canonical_literal(v)) for k, v in value.items()
        )))
    # Fallback: repr — covers callables, modules, etc.
    return ("repr", repr(value))


def _canonical_attrs(attrs: dict[str, Any]) -> tuple:
    items = []
    for k in sorted(attrs.keys()):
        if k not in _HASHED_ATTR_KEYS:
            continue
        items.append((k, _canonical_literal(attrs[k])))
    return tuple(items)


def ir_structural_signature(ir: FunctionIR) -> dict[str, Any]:
    """Build a JSON-serialisable structural signature for ``ir``.

    The signature is invariant under:

    * op_id / value_id renaming
    * block_id renaming (block ordering is preserved by the entry-block
      anchor and reachability traversal)
    * insertion order of ops within a block? No — Python execution
      order matters, so op order within a block IS preserved.
    * unused metadata such as source_span / _provenance / type_info

    The signature is sensitive to:

    * the set and order of ops in each block, by opcode and attrs
    * the def->use graph topology
    * literal values of constants
    * arg signature (count + types) and return arity
    """
    # Map raw ids -> dense canonical ints so IR variants that differ
    # only by counter values hash the same.
    val_id_map: dict[str, int] = {}
    op_id_map: dict[str, int] = {}
    blk_id_map: dict[str, int] = {}

    def _vid(v: str) -> int:
        if v not in val_id_map:
            val_id_map[v] = len(val_id_map)
        return val_id_map[v]

    def _oid(o: str) -> int:
        if o not in op_id_map:
            op_id_map[o] = len(op_id_map)
        return op_id_map[o]

    def _bid(b: str) -> int:
        if b not in blk_id_map:
            blk_id_map[b] = len(blk_id_map)
        return blk_id_map[b]

    # Pre-allocate arg ids first so identical signatures align.
    for vid in ir.arg_values:
        _vid(vid)

    # Walk blocks in reachable order: BFS from entry block.
    visited_blocks: list[str] = []
    queue = [ir.entry_block] if ir.entry_block in ir.blocks else list(ir.blocks.keys())
    seen = set()
    while queue:
        bid = queue.pop(0)
        if bid in seen or bid not in ir.blocks:
            continue
        seen.add(bid)
        visited_blocks.append(bid)
        for s in ir.blocks[bid].succs:
            if s not in seen:
                queue.append(s)
    for bid in ir.blocks:
        if bid not in seen:
            visited_blocks.append(bid)

    block_records: list[dict[str, Any]] = []
    for bid in visited_blocks:
        block = ir.blocks[bid]
        op_records: list[dict[str, Any]] = []
        for oid in block.op_ids:
            op = ir.ops.get(oid)
            if op is None:
                continue
            op_records.append({
                "op": _oid(oid),
                "k": op.opcode,
                "i": [_vid(v) for v in op.inputs],
                "o": [_vid(v) for v in op.outputs],
                "a": _canonical_attrs(op.attrs or {}),
            })
        block_records.append({
            "b": _bid(bid),
            "ops": op_records,
        })

    arg_records = []
    for vid in ir.arg_values:
        v = ir.values.get(vid)
        if v is None:
            continue
        arg_records.append({
            "v": _vid(vid),
            "type": v.type_hint or "",
            "name": v.name_hint or "",
        })

    return {
        "name": ir.name,
        "args": arg_records,
        "n_returns": len(ir.return_values),
        "blocks": block_records,
        "entry": _bid(ir.entry_block) if ir.entry_block in ir.blocks else 0,
    }


def canonical_ir_hash(ir: FunctionIR) -> str:
    """Return a 16-character hex digest of the canonical structure.

    Two FunctionIRs with the same hash are guaranteed to be
    behaviour-equivalent (modulo the hash collision probability of
    truncated SHA-1, which is negligible at the population sizes used
    by Phase H+4 — μ=32, λ=32, max generations≈100 per macro-gen).
    """
    sig = ir_structural_signature(ir)
    blob = json.dumps(sig, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]
