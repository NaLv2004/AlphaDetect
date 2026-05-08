"""MathView-based node feature encoding for the GNN.

Produces fixed-dimensional feature vectors from a ``MathNode`` (and the
underlying ``FunctionIR`` for slot/provenance metadata). This is the SOLE
encoding path used by the GNN; the legacy op-level encoding (one-hot of
``_OPCODE_LIST``) has been removed (Phase 3b — see math view integration
plan, no fallback).

Feature layout (total = ``MATH_VIEW_NODE_DIM``):
    [0 : N_KINDS)               kind one-hot (10 dims)
    [N_KINDS : N_KINDS+N_OPCODES) underlying-opcode one-hot (closed vocab)
    [+ : + CALLEE_BUCKETS)      callee/operator hash buckets (8 dims)
    [+ : + PROV_FEATURES)       provenance hash buckets + boundary flag
                                (16 + 1 = 17 dims, salted)
    [+ : + NUMERIC_FEATURES)    numeric / structural features (6 dims)

The provenance & callee-hash feature shapes are intentionally reused from
the legacy encoding so that overall network capacity (``_VALUE_FEAT_DIM``)
stays comparable; the *content* now derives from MathNode semantics.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .math_view import MathNode, MathView


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# 10 MathView kinds (Phase 3a contract: no "other" allowed)
_KIND_LIST: tuple[str, ...] = (
    "math",
    "tensor_struct",
    "state_update",
    "phi",
    "branch",
    "const",
    "jump",
    "boundary",
    "collection",
    "iter",
)
_KIND_VOCAB: dict[str, int] = {k: i for i, k in enumerate(_KIND_LIST)}
N_KINDS: int = len(_KIND_LIST)

# Underlying SSA opcode vocabulary. Closed list of every opcode that can
# appear as the *primary* opcode of a MathNode (i.e. not absorbed). Hard-
# coded so node feature dims stay stable across runs and ckpts.
_OPCODE_LIST: tuple[str, ...] = (
    "binary",
    "unary",
    "call",
    "compare",
    "subscript",
    "store",
    "load",
    "phi",
    "branch",
    "jump",
    "return",
    "get_attr",
    "get_item",
    "set_item",
    "build_list",
    "build_tuple",
    "build_dict",
    "build_slice",
    "iter_init",
    "iter_next",
    "alloc",
    "assign",
    "augassign",
    "const",
    "method_call",
    "<unk>",
)
_OPCODE_VOCAB: dict[str, int] = {op: i for i, op in enumerate(_OPCODE_LIST)}
N_OPCODES: int = len(_OPCODE_LIST)

CALLEE_BUCKETS: int = 8
PROV_HASH_BUCKETS: int = 16
PROV_FEATURES: int = PROV_HASH_BUCKETS + 1  # +1 boundary flag
NUMERIC_FEATURES: int = 6

MATH_VIEW_NODE_DIM: int = (
    N_KINDS + N_OPCODES + CALLEE_BUCKETS + PROV_FEATURES + NUMERIC_FEATURES
)


# ---------------------------------------------------------------------------
# Provenance hash (salt rotated from outside; mirrors legacy semantics)
# ---------------------------------------------------------------------------

_PROV_HASH_SALT = "math-view-prov-v0"


def set_provenance_hash_salt(salt: str) -> None:
    global _PROV_HASH_SALT
    _PROV_HASH_SALT = salt


def _hash_provenance(slot_id: str | None, is_boundary: bool) -> list[float]:
    feats = [0.0] * PROV_FEATURES
    if slot_id is not None:
        digest = hashlib.blake2s(
            f"{_PROV_HASH_SALT}|{slot_id}".encode(), digest_size=4,
        ).digest()
        bucket = int.from_bytes(digest, "big") % PROV_HASH_BUCKETS
        feats[bucket] = 1.0
    if is_boundary:
        feats[PROV_HASH_BUCKETS] = 1.0
    return feats


def _hash_callee(name: str) -> list[float]:
    if not name:
        return [0.0] * CALLEE_BUCKETS
    digest = hashlib.blake2s(
        f"callee|{name}".encode(), digest_size=4,
    ).digest()
    h = int.from_bytes(digest, "big")
    return [((h >> (i * 4)) & 0xF) / 15.0 for i in range(CALLEE_BUCKETS)]


# ---------------------------------------------------------------------------
# Underlying opcode extraction
# ---------------------------------------------------------------------------

def _underlying_opcode(opcode_label: str) -> str:
    """Map a MathNode.opcode label like ``'binary.MatMult'`` or
    ``'call.np.eye'`` or ``'arg.x'`` back to the underlying SSA opcode
    family used in the closed vocabulary. Unknown families fall through
    to ``'<unk>'``.

    Boundary nodes use opcode labels like ``arg.x`` / ``return``; map
    them to ``return`` (for ``return``) or to the kind-driven default.
    """
    if not opcode_label:
        return "<unk>"
    head = opcode_label.split(".", 1)[0]
    if head in _OPCODE_VOCAB:
        return head
    if head == "arg":
        # boundary arg node — no underlying opcode; use <unk> so the
        # kind one-hot carries the signal.
        return "<unk>"
    return "<unk>"


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def _primary_op(view: MathView, node: MathNode):
    """Return the canonical underlying ``Op`` for ``node`` (lex-min op_id),
    or ``None`` if the node owns no SSA op (boundary arg)."""
    if not node.op_ids:
        return None
    primary_id = min(node.op_ids)
    return view.ir.ops.get(primary_id)


def _callee_string(node: MathNode) -> str:
    """Best-effort string identifying the operator/callee of a MathNode.

    Pulls from MathNode attrs in priority order:
        qualified_name -> callee_name -> op (binary/compare op) ->
        attr (get_attr) -> bound_method_name -> name -> ''
    """
    a: Mapping[str, Any] = node.attrs or {}
    for key in (
        "qualified_name",
        "callee_name",
        "callee",
        "op",
        "attr",
        "bound_method_name",
        "name",
    ):
        v = a.get(key)
        if isinstance(v, str) and v:
            return v
    # Fall back to the second segment of the opcode label
    # (e.g. "binary.MatMult" -> "MatMult").
    if "." in node.opcode:
        return node.opcode.split(".", 1)[1]
    return ""


def node_feature_vector(view: MathView, node: MathNode) -> list[float]:
    """Build the fixed-dim feature row for one MathNode."""
    feats: list[float] = [0.0] * MATH_VIEW_NODE_DIM
    cursor = 0

    # 1. kind one-hot
    kind_idx = _KIND_VOCAB.get(node.kind)
    if kind_idx is None:
        # Phase 3a invariant violated — should never happen. Encode as
        # all-zeros in this slice so the network at least sees an
        # anomaly, but raise loudly so it gets fixed.
        raise ValueError(
            f"MathNode {node.node_id} has unrecognized kind={node.kind!r}"
        )
    feats[cursor + kind_idx] = 1.0
    cursor += N_KINDS

    # 2. underlying opcode one-hot
    underlying = _underlying_opcode(node.opcode)
    feats[cursor + _OPCODE_VOCAB.get(underlying, _OPCODE_VOCAB["<unk>"])] = 1.0
    cursor += N_OPCODES

    # 3. callee / operator hash buckets
    callee_feats = _hash_callee(_callee_string(node))
    feats[cursor:cursor + CALLEE_BUCKETS] = callee_feats
    cursor += CALLEE_BUCKETS

    # 4. provenance from primary op
    prim = _primary_op(view, node)
    slot_id: str | None = None
    is_boundary_prov = False
    if prim is not None:
        prov = prim.attrs.get("_provenance") or {}
        slot_id = prov.get("from_slot_id")
        is_boundary_prov = bool(prov.get("is_slot_boundary", False))
    feats[cursor:cursor + PROV_FEATURES] = _hash_provenance(slot_id, is_boundary_prov)
    cursor += PROV_FEATURES

    # 5. numeric / structural features
    n_inputs = len(node.inputs)
    n_outputs = node.n_outputs
    n_owned = len(node.op_ids)
    has_var_name = 1.0 if (node.attrs or {}).get("var_name_assigned") else 0.0
    has_bound_method = 1.0 if (node.attrs or {}).get("bound_method_name") else 0.0
    is_boundary = 1.0 if node.kind == "boundary" else 0.0
    numeric = [
        min(n_inputs / 8.0, 1.0),
        min(n_outputs / 4.0, 1.0),
        min(n_owned / 4.0, 1.0),
        has_var_name,
        has_bound_method,
        is_boundary,
    ]
    feats[cursor:cursor + NUMERIC_FEATURES] = numeric
    cursor += NUMERIC_FEATURES

    assert cursor == MATH_VIEW_NODE_DIM
    return feats


def zero_node_feature_vector() -> list[float]:
    return [0.0] * MATH_VIEW_NODE_DIM
