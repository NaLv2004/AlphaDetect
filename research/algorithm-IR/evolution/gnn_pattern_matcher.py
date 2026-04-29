"""GNN-based BCIR pattern matcher for structural graft proposals.

This matcher learns three things online:
1. A graph-level pair scorer for host/donor compatibility.
2. A BCIR host-region policy over boundary outputs + cut values.
3. A BCIR donor-region policy over boundary outputs + cut values.

The actual region construction / legality / extraction logic lives in
``algorithm_ir.region``.  The evolution layer only scores candidates and
samples actions.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.ir.type_lattice import is_subtype
from algorithm_ir.region.contract import infer_boundary_contract
from algorithm_ir.region.extract import extract_region_ir
from algorithm_ir.region.selector import BoundaryRegionSpec, RewriteRegion, define_rewrite_region
from algorithm_ir.region.slicer import (
    enumerate_cut_candidates,
    enumerate_observable_values,
    validate_boundary_region,
)
from algorithm_ir.region.triviality import (
    is_trivial_op,
    visible_def_op,
    visible_def_ops,
)
from evolution.pattern_matchers import _fresh_id
from evolution.pool_types import AlgorithmEntry, GraftProposal, SlotStampingProposal
from evolution.graft_classifier import (
    BoundarySignature,
    classify_region,
    signature_for_region,
)
from evolution.host_region_mask import (
    clear_singleton_cut_cache,
    cut_step_mask as _host_cut_step_mask,
    filter_dead_code_outputs as _host_filter_dead_code_outputs,
    is_output_combo_feasible as _host_is_output_combo_feasible,
    output_step_mask as _host_output_step_mask,
    precompute_op_closures as _host_precompute_op_closures,
)
from evolution.donor_region_mask import (
    donor_cut_step_mask as _donor_cut_step_mask,
    donor_output_step_mask as _donor_output_step_mask,
    donor_pool_signature_compatible as _donor_pool_signature_compatible,
    donor_cut_pool_union as _donor_cut_pool_union,
    precompute_op_closures as _donor_precompute_op_closures,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Opcode vocabulary / basic features
# ---------------------------------------------------------------------------

_OPCODE_VOCAB: dict[str, int] = {}
_OPCODE_LIST = [
    "const", "binary", "unary", "call", "compare", "branch", "jump",
    "return", "subscript", "store", "load", "algslot", "phi",
    "attr", "build_list", "build_tuple", "build_dict", "augassign",
    "<unk>",
]
for _index, _opcode in enumerate(_OPCODE_LIST):
    _OPCODE_VOCAB[_opcode] = _index

_N_OPCODES = len(_OPCODE_LIST)
_CALLEE_FEATURES = 8
# Per code_review.md §2.3: extend node features with provenance signal so
# the GNN can see which slot a (FII-inlined) op originated from.  Use
# **hash buckets** (NOT one-hot) of the slot-id string to prevent the
# feature from degenerating into a memorizable categorical prior.  An
# extra scalar flags the slot-boundary marker ops themselves.
_PROV_HASH_BUCKETS = 16
_PROV_FEATURES = _PROV_HASH_BUCKETS + 1  # 16 buckets + 1 boundary flag
_NODE_DIM = _N_OPCODES + _CALLEE_FEATURES + _PROV_FEATURES
_VALUE_STATIC_DIM = 10
_VALUE_FEAT_DIM = _NODE_DIM * 2 + _VALUE_STATIC_DIM


def _opcode_idx(opcode: str) -> int:
    return _OPCODE_VOCAB.get(opcode, _OPCODE_VOCAB["<unk>"])


def _compute_return_slice_values(ir: FunctionIR) -> set[str]:
    """Set of SSA values that can affect the function's observable output.

    "Observable" = either fed into a ``return`` op, OR feeding any
    side-effecting / control-flow op whose effects are themselves
    observable.  We approximate this by treating every op of opcode
    ``return``, ``branch``, ``store``, ``set_item`` as a sink: a value
    that reaches such a sink along the data-dep chain is live.

    This is intentionally conservative (over-approximate the live set):
    the goal of the resulting filter is to reject **only** regions whose
    exit_values demonstrably cannot affect the output (e.g. an isolated
    subexpression in a function with no loops whose result is never
    read).  Treating ``branch`` inputs as sinks is essential because in
    looping algorithms the entire body computation feeds the loop
    termination test, so naively following only ``return`` inputs would
    declare nearly the whole function dead.
    """
    SINKS = ("return", "branch", "store", "set_item")
    slice_values: set[str] = set()
    pending: list[str] = list(ir.return_values)
    for op in ir.ops.values():
        if op.opcode in SINKS:
            pending.extend(op.inputs)
    while pending:
        vid = pending.pop()
        if vid in slice_values:
            continue
        value = ir.values.get(vid)
        if value is None:
            continue
        slice_values.add(vid)
        def_op_id = value.def_op
        if def_op_id and def_op_id in ir.ops:
            pending.extend(ir.ops[def_op_id].inputs)
    return slice_values


def _hash_callee(name: str, dim: int = _CALLEE_FEATURES) -> list[float]:
    h = hash(name) & 0xFFFFFFFF
    return [((h >> (i * 4)) & 0xF) / 15.0 for i in range(dim)]


# Salt is rotated periodically (every 50 macro gens, per §6 risk register)
# to keep the GNN from memorizing slot-id => bucket mappings.
_PROV_HASH_SALT = "fii-prov-v0"


def set_provenance_hash_salt(salt: str) -> None:
    """Rotate the salt used by ``_hash_provenance``.

    Per code_review.md §6: hash buckets are 16-wide, ≥14 slot kinds → guaranteed
    collisions; periodically re-hashing with a new salt prevents a stable
    one-hot prior from forming.
    """
    global _PROV_HASH_SALT
    _PROV_HASH_SALT = salt


def _hash_provenance(slot_id: str | None, is_boundary: bool) -> list[float]:
    """Return a ``_PROV_FEATURES``-dim feature vector for an op's slot
    provenance.  Uses a salted bucket-hash (NOT one-hot) of the slot id."""
    feats = [0.0] * _PROV_FEATURES
    if slot_id is not None:
        # Stable salted hash (Python hash() varies per run; use a fixed
        # algorithm so node features are deterministic across runs of
        # the same model).
        import hashlib
        digest = hashlib.blake2s(
            f"{_PROV_HASH_SALT}|{slot_id}".encode(), digest_size=4,
        ).digest()
        h = int.from_bytes(digest, "big")
        bucket = h % _PROV_HASH_BUCKETS
        feats[bucket] = 1.0
    if is_boundary:
        feats[_PROV_HASH_BUCKETS] = 1.0
    return feats


# ---------------------------------------------------------------------------
# IR -> graph conversion
# ---------------------------------------------------------------------------

def ir_to_graph(ir: FunctionIR) -> Data:
    """Build a torch_geometric graph view of ``ir`` for the GNN encoder.

    Plan B visibility filter (see ``algorithm_ir/region/triviality.py``):
    trivial ops (``const``, ``get_attr``, ``assign``, and trivial
    ``phi(x,x)``) are **excluded as graph nodes** because they inflate
    node counts without carrying real algorithmic structure. Dataflow
    edges are made transitive through hidden ops via
    :func:`visible_def_op`, so the GNN sees an edge from each non-trivial
    producer directly to its non-trivial consumer.
    """
    all_ops = list(ir.ops.values())
    if not all_ops:
        x = torch.zeros(1, _NODE_DIM)
        return Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long))

    # Filter out trivial ops; they remain in the physical IR (codegen,
    # execution, slot_meta, grafting all unchanged) but the GNN never
    # sees them as nodes.
    visible_ops = [op for op in all_ops if not is_trivial_op(op, ir)]
    if not visible_ops:
        # Pathological case: an entirely trivial IR. Fall back to the
        # raw graph to preserve at least one node so encoder can run.
        visible_ops = all_ops

    op_id_to_idx = {op.id: idx for idx, op in enumerate(visible_ops)}

    # Node features (unchanged construction; just over the visible set).
    node_feats: list[list[float]] = []
    for op in visible_ops:
        one_hot = [0.0] * _N_OPCODES
        one_hot[_opcode_idx(op.opcode)] = 1.0
        callee = op.attrs.get("callee", op.attrs.get("name", ""))
        callee_feat = _hash_callee(callee) if callee else [0.0] * _CALLEE_FEATURES
        prov = op.attrs.get("_provenance") or {}
        prov_feat = _hash_provenance(
            prov.get("from_slot_id"),
            bool(prov.get("is_slot_boundary", False)),
        )
        node_feats.append(one_hot + callee_feat + prov_feat)
    x = torch.tensor(node_feats, dtype=torch.float32)

    # For each value in the IR, resolve the *set* of visible ops that
    # transitively produce it. Multi-input trivial ops (e.g.
    # ``build_tuple(a, b, c)``) fan out to every non-trivial producer
    # of every component, so the GNN sees edges from each underlying
    # value source rather than losing connectivity through the packer.
    value_visible_defs: dict[str, list[int]] = {}
    for vid in ir.values:
        producers = visible_def_ops(ir, vid)
        if not producers:
            continue
        idxs = [op_id_to_idx[p.id] for p in producers if p.id in op_id_to_idx]
        if idxs:
            value_visible_defs[vid] = idxs

    # Dataflow edges: every visible op pulls from every visible producer
    # of each of its inputs. Trivial ops collapse silently.
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_set: set[tuple[int, int]] = set()
    for op in visible_ops:
        op_idx = op_id_to_idx[op.id]
        for value_id in op.inputs:
            for def_idx in value_visible_defs.get(value_id, ()):
                if def_idx == op_idx:
                    continue
                key = (def_idx, op_idx)
                if key in edge_set:
                    continue
                edge_set.add(key)
                src_list.append(def_idx)
                dst_list.append(op_idx)

    # Block-order sequential edges (same logic as before, but trivial
    # ops are skipped — we link each visible op to the previous visible
    # op in the same block). Preserves locality without re-introducing
    # spurious edges through hidden nodes.
    for block in ir.blocks.values():
        prev_idx = None
        for op_id in block.op_ids:
            cur_idx = op_id_to_idx.get(op_id)
            if cur_idx is None:
                continue  # trivial op skipped
            if prev_idx is not None and prev_idx != cur_idx:
                key = (prev_idx, cur_idx)
                if key not in edge_set:
                    edge_set.add(key)
                    src_list.append(prev_idx)
                    dst_list.append(cur_idx)
            prev_idx = cur_idx

    edge_index = (
        torch.tensor([src_list, dst_list], dtype=torch.long)
        if src_list
        else torch.zeros(2, 0, dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index)


# ---------------------------------------------------------------------------
# GNN backbone / pair scorer / BCIR policy
# ---------------------------------------------------------------------------

class IRGraphEncoder(nn.Module):
    def __init__(self, node_dim: int = _NODE_DIM, hidden: int = 64, out_dim: int = 32, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden, heads=heads, concat=False)
        self.conv2 = GATConv(hidden, hidden, heads=heads, concat=False)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)


class GraftScorer(nn.Module):
    def __init__(self, emb_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, host_emb: torch.Tensor, donor_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([host_emb, donor_emb], dim=-1))


class BoundaryRegionPolicy(nn.Module):
    """Unified BCIR policy over boundary outputs and cut values."""

    def __init__(self, value_dim: int = _VALUE_FEAT_DIM, context_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.value_encoder = nn.Sequential(
            nn.Linear(value_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden + context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.cut_head = nn.Sequential(
            nn.Linear(hidden * 2 + context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.output_stop_head = nn.Sequential(
            nn.Linear(context_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.cut_stop_head = nn.Sequential(
            nn.Linear(context_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode_values(self, value_feats: torch.Tensor) -> torch.Tensor:
        if value_feats.numel() == 0:
            hidden = self.value_encoder[0].out_features
            return value_feats.new_zeros((0, hidden))
        return self.value_encoder(value_feats)

    def output_logits(
        self,
        value_feats: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every candidate value as a potential output port.

        ``mask`` is an optional 1-D ``BoolTensor`` (or 0/1 ``Tensor``) of
        the same length as ``value_feats``. ``True`` (=1) means the
        candidate is *eligible* (e.g. type-compatible with the host port
        being filled); ``False`` (=0) means it must be vetoed. Vetoed
        positions receive ``-inf`` logits so the subsequent softmax
        assigns them zero probability. ``mask=None`` disables masking.
        """
        value_embs = self.encode_values(value_feats)
        if value_embs.shape[0] == 0:
            return context.new_zeros((0,)), context.new_zeros((1,))
        ctx = context.unsqueeze(0).expand(value_embs.shape[0], -1)
        logits = self.output_head(torch.cat([value_embs, ctx], dim=-1)).squeeze(-1)
        if mask is not None:
            mask = mask.to(device=logits.device, dtype=torch.bool)
            if mask.shape != logits.shape:
                raise ValueError(
                    f"BoundaryRegionPolicy.output_logits: mask shape "
                    f"{tuple(mask.shape)} does not match logits "
                    f"{tuple(logits.shape)}"
                )
            logits = logits.masked_fill(~mask, float("-inf"))
        stop = self.output_stop_head(
            torch.cat([context, value_embs.mean(dim=0)], dim=-1)
        ).reshape(-1)
        return logits, stop

    def cut_logits(
        self,
        value_feats: torch.Tensor,
        context: torch.Tensor,
        output_summary: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every candidate value as a potential cut (entry) port.

        ``mask`` works identically to :meth:`output_logits` — an optional
        1-D Bool/0-1 tensor with one entry per candidate; ``False``
        positions are forced to ``-inf`` before softmaxing. Used by the
        donor sampler to enforce per-step type compatibility against the
        host's :class:`BoundarySignature`.
        """
        value_embs = self.encode_values(value_feats)
        if value_embs.shape[0] == 0:
            return context.new_zeros((0,)), context.new_zeros((1,))
        ctx = context.unsqueeze(0).expand(value_embs.shape[0], -1)
        out_sum = output_summary.unsqueeze(0).expand(value_embs.shape[0], -1)
        logits = self.cut_head(torch.cat([value_embs, ctx, out_sum], dim=-1)).squeeze(-1)
        if mask is not None:
            mask = mask.to(device=logits.device, dtype=torch.bool)
            if mask.shape != logits.shape:
                raise ValueError(
                    f"BoundaryRegionPolicy.cut_logits: mask shape "
                    f"{tuple(mask.shape)} does not match logits "
                    f"{tuple(logits.shape)}"
                )
            logits = logits.masked_fill(~mask, float("-inf"))
        stop = self.cut_stop_head(torch.cat([context, output_summary], dim=-1)).reshape(-1)
        return logits, stop


# ---------------------------------------------------------------------------
# Main matcher
# ---------------------------------------------------------------------------

class GNNPatternMatcher:
    """BCIR-based structural graft proposal generator."""

    def __init__(
        self,
        max_proposals_per_gen: int = 4,
        top_k_pairs: int = 8,
        min_region_size: int = 1,
        max_region_size: int = 64,
        max_boundary_outputs: int = 3,
        max_cut_values: int = 4,
        max_region_ops: int = 64,
        max_region_inputs: int = 12,
        max_region_outputs: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 2048,
        train_interval: int = 1,
        train_steps: int = 1,
        warmstart_generations: int = 0,
        pair_temperature: float = 0.7,
        pair_exploration: float = 0.10,
        region_exploration: float = 0.10,
        donor_exploration: float = 0.10,
        enable_batched_proposals: bool = False,
        proposal_batch_size: int = 64,
        show_progress: bool = False,
        device: str | None = None,
        enable_host_mask: bool = True,
        enable_donor_mask: bool = True,
    ):
        self.max_proposals_per_gen = max_proposals_per_gen
        self.top_k_pairs = top_k_pairs
        self.min_region_size = min_region_size
        self.max_region_size = max_region_size
        self.max_boundary_outputs = max_boundary_outputs
        self.max_cut_values = max_cut_values
        self.max_region_ops = max_region_ops
        self.max_region_inputs = max_region_inputs
        self.max_region_outputs = max_region_outputs
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.train_interval = train_interval
        self.train_steps = train_steps
        self.warmstart_generations = warmstart_generations
        self.pair_temperature = pair_temperature
        self.pair_exploration = pair_exploration
        self.region_exploration = region_exploration
        self.donor_exploration = donor_exploration
        self.enable_batched_proposals = enable_batched_proposals
        self.proposal_batch_size = max(1, int(proposal_batch_size))
        self.show_progress = show_progress
        # When True, host-side region sampling uses the 4-layer mask in
        # ``evolution.host_region_mask`` to guarantee structural and
        # numerical validity *without* relying on the greedy repair
        # fallback in ``_build_boundary_region``.  When False, the
        # legacy permissive sampling + greedy repair pipeline is used.
        self.enable_host_mask = bool(enable_host_mask)
        # When True, donor-side region sampling uses the 4-layer mask in
        # ``evolution.donor_region_mask`` (Layer D1 pool prefilter +
        # Layer D2 output mask + Layer D4 cut/STOP mask) to enforce
        # equality arity match against the host BoundarySignature
        # without any post-hoc repair.
        self.enable_donor_mask = bool(enable_donor_mask)
        # Per-pair retry budget for the masked donor sampler.  Greedy
        # masked sampling can paint itself into dead-end output prefixes;
        # re-sampling from scratch usually recovers.  10 retries gets
        # end-to-end pass rate to ~97% when per-attempt rate is ~30%.
        self.donor_sample_max_retries = 10

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.encoder = IRGraphEncoder().to(self.device)
        self.scorer = GraftScorer().to(self.device)
        self.boundary_region_policy = BoundaryRegionPolicy().to(self.device)

        # Compatibility aliases for older checkpoints and tests.
        self.region_proposer = self.boundary_region_policy
        self.donor_region_selector = self.boundary_region_policy

        self._all_params = (
            list(self.encoder.parameters())
            + list(self.scorer.parameters())
            + list(self.boundary_region_policy.parameters())
        )
        self.optimizer = torch.optim.Adam(self._all_params, lr=lr)

        self._reward_baseline = 0.0
        self._baseline_alpha = 0.1
        # Per-host EMA baselines.  Reward distributions vary by host
        # (a graft of MMSE is much harder to beat than one of a
        # broken random algorithm), so a global baseline gives biased
        # advantage signals.  Falls back to ``self._reward_baseline``
        # for hosts seen for the first time.
        self._host_baselines: dict[str, float] = {}
        self._host_baseline_counts: dict[str, int] = {}
        # Minimum samples before a per-host baseline is trusted.
        # With too few samples the baseline equals the single
        # observed reward, so the advantage degenerates to 0 and
        # there is no learning signal.
        self._host_baseline_min_n = 3
        # Loss-weight for the REINFORCE term, applied as
        # ``mse + lambda * mean(reinforce_terms)``.  Default 0.3 so
        # that the encoder is still primarily trained by the
        # supervised MSE signal but the policy still gets a clear
        # gradient.
        self._lambda_rl = 0.3
        self._experience: list[dict[str, Any]] = []
        self._outcomes: dict[str, dict[str, Any]] = {}
        self._graph_cache: dict[str, Data] = {}
        self._emb_cache: dict[str, torch.Tensor] = {}
        self._generation = 0
        self._total_proposals = 0
        self._total_rewards = 0.0
        self._last_train_stats: dict[str, Any] = {}
        self._last_proposal_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # PatternMatcherFn interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        entries: list[AlgorithmEntry],
        generation: int,
    ) -> list[GraftProposal]:
        self._generation = generation
        if len(entries) < 2:
            return []

        import time as _time
        t0 = _time.perf_counter()

        self._encode_entries(entries)
        t_enc = _time.perf_counter()
        if generation > 0 and generation % self.train_interval == 0:
            self._train_step(n_steps=self.train_steps)
        t_train = _time.perf_counter()

        pair_scores = self._score_pairs(entries)
        t_score = _time.perf_counter()
        selected_pairs = self._select_pair_candidates(pair_scores)
        max_proposals = (
            len(selected_pairs)
            if self.is_warmstart_generation(generation)
            else self.max_proposals_per_gen
        )
        proposals, proposal_stats = self._propose_pairs(
            selected_pairs,
            max_proposals=max_proposals,
        )
        t_prop = _time.perf_counter()

        self._last_proposal_stats = {
            "warmstart": self.is_warmstart_generation(generation),
            "pair_candidates_scored": len(pair_scores),
            "pair_candidates_selected": len(selected_pairs),
            "proposals_built": len(proposals),
            "batched": bool(self.enable_batched_proposals),
            "proposal_batch_size": int(self.proposal_batch_size),
            **proposal_stats,
        }
        logger.info(
            "GNN propose: encode=%.2fs train=%.2fs score=%.2fs propose=%.2fs -> %d proposals from %d entries",
            t_enc - t0,
            t_train - t_enc,
            t_score - t_train,
            t_prop - t_score,
            len(proposals),
            len(entries),
        )
        return proposals

    # ------------------------------------------------------------------
    # M6b §6.7 — Track B: slot stamping (slot discovery)
    # ------------------------------------------------------------------
    def propose_slot_stampings(
        self,
        entries: list[AlgorithmEntry],
        n: int = 4,
    ) -> list[SlotStampingProposal]:
        """Suggest *new* SESE regions to be stamped as slots.

        Track B runs the same boundary policy as Track A but in
        single-IR mode (``mask=None``): for each host genome, sample a
        set of output values, then a set of cut values, build a SESE
        region via :func:`define_rewrite_region`, validate that the
        region is structurally cohesive, and reject any region whose
        op-set overlaps an existing slot's transitive op set.

        Returned :class:`SlotStampingProposal`\\s are consumed by the
        orchestrator after a Case II graft succeeds: they are applied
        via :func:`evolution.slot_dissolve.apply_slot_stamping` which
        tags ops, inserts a fresh ``SlotMeta`` entry, and seeds a
        :class:`SlotPopulation` whose first variant is a
        :class:`SubgraphSnapshot` extracted from the stamped region.

        Parameters
        ----------
        entries
            Pool snapshot. Must already be encoded (caches populated by
            a prior :meth:`__call__`); ``propose_slot_stampings`` does
            not invoke the GNN trainer.
        n
            Maximum number of stamping proposals to emit across the
            pool. Acts as a hard cap.

        Returns
        -------
        list[SlotStampingProposal]
        """
        if n <= 0 or not entries:
            return []
        # Use whatever embeddings are already cached. Track B is a
        # passive observer of the pool's structure; it does not retrain.
        proposals: list[SlotStampingProposal] = []
        context_cache: dict[str, dict[str, Any] | None] = {}
        temperature = self._policy_temperature()
        for entry in entries:
            if len(proposals) >= n:
                break
            ctx = self._get_entry_context(entry, context_cache)
            if ctx is None:
                continue
            ir = ctx["ir"]
            # Skip if there's no slack for a new slot (every op is
            # already tagged).
            already_tagged: set[str] = set()
            for key in (ir.slot_meta or {}).keys():
                already_tagged |= set(ir.slot_full_op_ids(key))
            untagged_ops = set(ir.ops.keys()) - already_tagged
            if not untagged_ops:
                continue
            # ── Policy roll: outputs first ─────────────────────────
            out_logits_list = self._compute_output_logits_batch([
                {
                    "candidate_feats_np": ctx["observable_feats_np"],
                    # Use the host's own embedding as "context" — Track
                    # B has no donor; the policy is being asked "find a
                    # cohesive sub-DAG inside this IR".
                    "context_emb_np": ctx["emb"].detach().cpu().numpy(),
                }
            ])
            if not out_logits_list:
                continue
            out_logits, out_stop = out_logits_list[0]
            outputs = self._sample_value_sequence(
                ctx["observable_values"],
                out_logits,
                out_stop,
                max_selected=self.max_boundary_outputs,
                allow_empty=False,
                temperature=temperature,
                exploration=self.region_exploration,
            )
            if not outputs:
                continue
            # ── Cuts ───────────────────────────────────────────────
            cut_ctx = self._get_cut_context(ctx, outputs)
            if not cut_ctx["candidate_ids"]:
                continue
            cut_logits_list = self._compute_cut_logits_batch([
                {
                    "candidate_feats_np": cut_ctx["candidate_feats_np"],
                    "context_emb_np": ctx["emb"].detach().cpu().numpy(),
                    "output_summary_np": self._summarize_selected_value_feats(
                        ctx["observable_values"],
                        ctx["observable_feats_np"],
                        outputs,
                    ),
                }
            ])
            if not cut_logits_list:
                continue
            cut_logits, cut_stop = cut_logits_list[0]
            cuts = self._sample_value_sequence(
                cut_ctx["candidate_ids"],
                cut_logits,
                cut_stop,
                max_selected=self.max_cut_values,
                allow_empty=True,
                temperature=temperature,
                exploration=self.region_exploration,
            )
            # ── Build & validate SESE region ───────────────────────
            built = self._build_boundary_region(
                ir, output_values=outputs, cut_values=cuts,
            )
            if built is None:
                continue
            region, validity = built
            if not validity.is_valid:
                continue
            region_op_set = set(region.op_ids)
            # Reject if the region overlaps any existing slot.
            overlaps_existing = False
            for key in (ir.slot_meta or {}).keys():
                if region_op_set & set(ir.slot_full_op_ids(key)):
                    overlaps_existing = True
                    break
            if overlaps_existing:
                continue
            # Suggest a deterministic pop_key derived from the op set.
            import hashlib as _hashlib
            digest = _hashlib.sha1(
                ",".join(sorted(region_op_set)).encode("utf-8")
            ).hexdigest()[:8]
            suggested_key = f"auto_{digest}"
            confidence = float(
                1.0 / (1.0 + max(0.0, float(validity.n_inputs + validity.n_outputs)))
            )
            proposals.append(SlotStampingProposal(
                host_algo_id=entry.algo_id,
                op_ids=tuple(sorted(region_op_set)),
                inputs=tuple(region.entry_values),
                outputs=tuple(region.exit_values),
                suggested_pop_key=suggested_key,
                confidence=confidence,
                rationale=(
                    f"Track-B stamp: n_ops={validity.n_ops} "
                    f"n_inputs={validity.n_inputs} "
                    f"n_outputs={validity.n_outputs}"
                ),
            ))
        return proposals

    # ------------------------------------------------------------------
    # Reward feedback
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        proposal_id: str,
        reward: float,
        graft_score: float | None = None,
        host_score: float | None = None,
        is_valid: bool | None = None,
    ) -> None:
        reward = max(reward, -10.0)
        if graft_score is None or not np.isfinite(graft_score):
            graft_score = 1.5
        # Find host_algo_id for the proposal so we can update the
        # per-host baseline online.  This is best-effort: if no
        # matching experience is found (e.g. proposal_id is stale),
        # the per-host update is skipped.
        host_algo_id: str | None = None
        for exp in reversed(self._experience):
            if exp.get("proposal_id") == proposal_id:
                host_algo_id = exp.get("host_algo")
                break
        self._outcomes[proposal_id] = {
            "reward": float(reward),
            "graft_score": float(graft_score),
            "host_score": float(host_score) if host_score is not None else None,
            "is_valid": bool(is_valid) if is_valid is not None else False,
            "host_algo": host_algo_id,
        }
        self._total_rewards += float(reward)
        if host_algo_id is not None:
            cur = self._host_baselines.get(host_algo_id, self._reward_baseline)
            self._host_baselines[host_algo_id] = (
                (1 - self._baseline_alpha) * cur
                + self._baseline_alpha * float(reward)
            )
            self._host_baseline_counts[host_algo_id] = (
                self._host_baseline_counts.get(host_algo_id, 0) + 1
            )

    def is_warmstart_generation(self, generation: int | None = None) -> bool:
        gen = self._generation if generation is None else generation
        return self.warmstart_generations > 0 and gen <= self.warmstart_generations

    def _select_pair_candidates(
        self,
        pair_scores: list[tuple[AlgorithmEntry, AlgorithmEntry, float]],
    ) -> list[tuple[AlgorithmEntry, AlgorithmEntry, float]]:
        if not pair_scores:
            return []
        if self.is_warmstart_generation():
            # Cap warmstart pair count at 4× the steady-state per-gen
            # budget.  Without this cap, the warmstart explored every
            # host×donor pair (often thousands), and each pair now
            # produces a successfully-built graft (auto-repair makes
            # structural success ~100%).  That makes the per-gen
            # evaluation cost explode (hours per generation) without
            # adding training signal beyond what a few hundred diverse
            # samples already provide.
            cap = max(self.max_proposals_per_gen * 2, self.max_proposals_per_gen)
            if cap < len(pair_scores):
                # Random subsample to preserve diversity (no
                # score-based prior — warmstart is meant to be
                # exploratory).
                idx = np.random.choice(len(pair_scores), size=cap, replace=False)
                return [pair_scores[i] for i in idx]
            return pair_scores

        n_take = min(self.max_proposals_per_gen, len(pair_scores))
        if n_take >= len(pair_scores):
            return pair_scores

        predicted = np.array([score for _, _, score in pair_scores], dtype=np.float64)
        probs = self._make_sampling_probs(
            predicted,
            temperature=self.pair_temperature,
            exploration=self.pair_exploration,
            prefer_lower=True,
        )
        # Host-diversity cap: limit how many proposals can target the
        # same host per generation.  Without this, the scorer collapses
        # on a few high-likelihood hosts and all grafts compete against
        # the same (near-optimal) baseline, producing very few
        # "effective" grafts.  Rejection-sample from ``probs`` until we
        # have ``n_take`` entries subject to the per-host cap.
        distinct_hosts = {h.algo_id for h, _, _ in pair_scores}
        n_hosts = max(1, len(distinct_hosts))
        per_host_cap = max(2, int(np.ceil(n_take / n_hosts)) + 1)

        order = np.random.choice(
            len(pair_scores), size=len(pair_scores), replace=False, p=probs
        )
        host_counts: dict[str, int] = {}
        chosen_idx: list[int] = []
        for i in order:
            host_id = pair_scores[int(i)][0].algo_id
            if host_counts.get(host_id, 0) >= per_host_cap:
                continue
            chosen_idx.append(int(i))
            host_counts[host_id] = host_counts.get(host_id, 0) + 1
            if len(chosen_idx) >= n_take:
                break
        if len(chosen_idx) < n_take:
            # Top-up ignoring the cap if we couldn't fill the quota.
            for i in order:
                ii = int(i)
                if ii in chosen_idx:
                    continue
                chosen_idx.append(ii)
                if len(chosen_idx) >= n_take:
                    break
        return [pair_scores[i] for i in chosen_idx]

    def _make_sampling_probs(
        self,
        values: np.ndarray,
        *,
        temperature: float,
        exploration: float,
        prefer_lower: bool,
    ) -> np.ndarray:
        if values.size == 0:
            return values
        logits = -values if prefer_lower else values.copy()
        logits = logits - np.max(logits)
        probs = np.exp(logits / max(float(temperature), 1e-4))
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            probs = np.full(values.shape[0], 1.0 / values.shape[0], dtype=np.float64)
        else:
            probs = probs / total
        eps = float(np.clip(exploration, 0.0, 1.0))
        if eps > 0:
            probs = (1.0 - eps) * probs + eps / probs.shape[0]
        return probs / probs.sum()

    def _encode_entries(self, entries: list[AlgorithmEntry]) -> None:
        self._emb_cache.clear()
        graphs: list[Data] = []
        keys: list[str] = []
        live_ids: set[str] = set()
        _enc_iter: Any = (
            tqdm(entries, desc="Build IR graphs", leave=False)
            if self.show_progress and tqdm is not None
            else entries
        )
        for entry in _enc_iter:
            live_ids.add(entry.algo_id)
            self._graph_cache[entry.algo_id] = ir_to_graph(entry.ir)
            graphs.append(self._graph_cache[entry.algo_id])
            keys.append(entry.algo_id)

        stale_ids = [algo_id for algo_id in self._graph_cache if algo_id not in live_ids]
        for algo_id in stale_ids:
            del self._graph_cache[algo_id]

        if not graphs:
            return

        with torch.no_grad():
            batch = Batch.from_data_list([graph.clone() for graph in graphs]).to(self.device)
            embeddings = self.encoder(batch)
        for index, algo_id in enumerate(keys):
            self._emb_cache[algo_id] = embeddings[index]

    def _score_pairs(
        self,
        entries: list[AlgorithmEntry],
    ) -> list[tuple[AlgorithmEntry, AlgorithmEntry, float]]:
        valid = [
            (entry, self._emb_cache[entry.algo_id])
            for entry in entries
            if entry.algo_id in self._emb_cache
        ]
        if len(valid) < 2:
            return []

        host_indices: list[int] = []
        donor_indices: list[int] = []
        for i in range(len(valid)):
            for j in range(len(valid)):
                if i != j:
                    host_indices.append(i)
                    donor_indices.append(j)
        if not host_indices:
            return []

        embs = torch.stack([emb for _, emb in valid], dim=0)
        host_embs = embs[host_indices]
        donor_embs = embs[donor_indices]
        with torch.no_grad():
            scores = self.scorer(host_embs, donor_embs).squeeze(-1).cpu().numpy()

        results: list[tuple[AlgorithmEntry, AlgorithmEntry, float]] = []
        for idx, (host_idx, donor_idx) in enumerate(zip(host_indices, donor_indices)):
            results.append((valid[host_idx][0], valid[donor_idx][0], float(scores[idx])))
        return results

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def _propose_pairs(
        self,
        selected_pairs: list[tuple[AlgorithmEntry, AlgorithmEntry, float]],
        *,
        max_proposals: int,
    ) -> tuple[list[GraftProposal], dict[str, Any]]:
        if not selected_pairs or max_proposals <= 0:
            return [], self._empty_proposal_stats()

        clear_singleton_cut_cache()
        pair_slice = list(selected_pairs[:max_proposals])
        _show = self.show_progress and tqdm is not None

        invalid_regions: dict[str, int] = {}
        context_cache: dict[str, dict[str, Any] | None] = {}
        pair_infos: list[dict[str, Any]] = []
        _ctx_iter: Any = (
            tqdm(pair_slice, desc=f"Build pair contexts ({len(pair_slice)})", leave=False)
            if _show else pair_slice
        )
        for host_entry, donor_entry, pair_score in _ctx_iter:
            host_ctx = self._get_entry_context(host_entry, context_cache)
            donor_ctx = self._get_entry_context(donor_entry, context_cache)
            if host_ctx is None or donor_ctx is None:
                invalid_regions["missing_context"] = invalid_regions.get("missing_context", 0) + 1
                continue
            pair_infos.append({
                "host_entry": host_entry,
                "donor_entry": donor_entry,
                "pair_score": float(pair_score),
                "host_ctx": host_ctx,
                "donor_ctx": donor_ctx,
            })
        if not pair_infos:
            return [], self._proposal_stats(invalid_regions, [], [], 0)

        # Host outputs
        # When ``enable_host_mask`` is on we use the dead-code-filtered
        # observable pool (Layer 1). The featuriser must agree on the
        # candidate ordering, so we also pass the parallel features.
        host_use_mask = self.enable_host_mask
        for info, out in zip(
            pair_infos,
            self._compute_output_logits_batch([
                {
                    "candidate_feats_np": (
                        info["host_ctx"]["host_observable_feats_np"]
                        if host_use_mask
                        else info["host_ctx"]["observable_feats_np"]
                    ),
                    "context_emb_np": info["donor_ctx"]["emb"].detach().cpu().numpy(),
                }
                for info in pair_infos
            ]),
        ):
            info["host_output_logits"] = out

        region_temperature = self._policy_temperature()
        donor_temperature = self._policy_temperature()

        host_cut_infos: list[dict[str, Any]] = []
        _host_out_iter: Any = (
            tqdm(pair_infos, desc="Sample host outputs", leave=False)
            if _show else pair_infos
        )
        for info in _host_out_iter:
            host_ctx = info["host_ctx"]
            host_obs_vals = (
                host_ctx["host_observable_values"]
                if host_use_mask
                else host_ctx["observable_values"]
            )
            if not host_obs_vals:
                # Layer 1 stripped everything (rare: the host has no
                # return-slice values at all).  Treat as a dead-end so
                # we do not crash on an empty softmax.
                invalid_regions["host_layer1_empty"] = (
                    invalid_regions.get("host_layer1_empty", 0) + 1
                )
                continue
            if host_use_mask:
                op_closures = host_ctx["host_op_closures"]

                def _output_mask_fn(
                    selected,
                    remaining,
                    _ir=host_ctx["ir"],
                    _cl=op_closures,
                    _max_ops=self.max_region_ops,
                    _max_out=self.max_region_outputs,
                    _max_in=self.max_region_inputs,
                    _max_cut=self.max_cut_values,
                ):
                    return _host_output_step_mask(
                        _ir, _cl, selected, remaining,
                        max_region_ops=_max_ops,
                        max_region_outputs=_max_out,
                        max_region_inputs=_max_in,
                        max_cut_budget=_max_cut,
                    )

                step_mask_fn = _output_mask_fn
            else:
                step_mask_fn = None
            host_outputs = self._sample_value_sequence(
                host_obs_vals,
                info["host_output_logits"][0],
                info["host_output_logits"][1],
                max_selected=self.max_boundary_outputs,
                allow_empty=False,
                temperature=region_temperature,
                exploration=self.region_exploration,
                step_mask_fn=step_mask_fn,
            )
            if not host_outputs:
                invalid_regions["host_no_output"] = invalid_regions.get("host_no_output", 0) + 1
                continue
            if host_use_mask:
                # Dead-end check: cuts monotonely reduce ops + exits, so
                # if greedy-cut-all already violates max_region_ops or
                # max_region_outputs, no cut combo can rescue this output
                # choice. Drop now to avoid a guaranteed validation fail.
                if not _host_is_output_combo_feasible(
                    host_ctx["ir"],
                    host_outputs,
                    max_region_ops=self.max_region_ops,
                    max_region_outputs=self.max_region_outputs,
                    max_region_inputs=self.max_region_inputs,
                    max_cut_budget=self.max_cut_values,
                ):
                    invalid_regions["host_infeasible_outputs"] = (
                        invalid_regions.get("host_infeasible_outputs", 0) + 1
                    )
                    continue
            info["host_selected_outputs"] = host_outputs
            info["host_cut_ctx"] = self._get_cut_context(host_ctx, host_outputs)
            host_cut_infos.append(info)

        # Host cuts
        for info, out in zip(
            host_cut_infos,
            self._compute_cut_logits_batch([
                {
                    "candidate_feats_np": info["host_cut_ctx"]["candidate_feats_np"],
                    "context_emb_np": info["donor_ctx"]["emb"].detach().cpu().numpy(),
                    "output_summary_np": self._summarize_selected_value_feats(
                        info["host_ctx"]["observable_values"],
                        info["host_ctx"]["observable_feats_np"],
                        info["host_selected_outputs"],
                    ),
                }
                for info in host_cut_infos
            ]),
        ):
            info["host_cut_logits"] = out

        host_valid_infos: list[dict[str, Any]] = []
        host_region_metrics: list[tuple[int, int, int]] = []
        effective_cut_sizes: list[int] = []
        # Per-stage host counters for clean ``host_validity_rate`` reporting:
        #   host_sampler_attempts  = sampler called (post-Layer-1 candidate pool non-empty)
        #   host_built_regions     = define_rewrite_region returned (no exception)
        #   host_validate_passes   = validate_boundary_region.is_valid (the headline metric)
        host_sampler_attempts = 0
        host_built_regions = 0
        host_validate_passes = 0
        _host_cut_iter: Any = (
            tqdm(host_cut_infos, desc="Sample host cuts", leave=False)
            if _show else host_cut_infos
        )
        for info in _host_cut_iter:
            host_ctx = info["host_ctx"]
            if host_use_mask:
                host_ir = host_ctx["ir"]
                host_outputs_for_mask = info["host_selected_outputs"]

                def _cut_mask_fn(
                    selected,
                    remaining,
                    _ir=host_ir,
                    _outs=host_outputs_for_mask,
                    _max_ops=self.max_region_ops,
                    _min_ops=self.min_region_size,
                    _max_in=self.max_region_inputs,
                    _max_out=self.max_region_outputs,
                    _max_cut=self.max_cut_values,
                    _cut_pool=info["host_cut_ctx"]["candidate_ids"],
                ):
                    return _host_cut_step_mask(
                        _ir, _outs, selected, remaining,
                        max_region_ops=_max_ops,
                        min_region_ops=_min_ops,
                        max_region_inputs=_max_in,
                        max_region_outputs=_max_out,
                        max_cut_budget=_max_cut,
                        cut_pool=_cut_pool,
                    )

                cut_step_fn = _cut_mask_fn
            else:
                cut_step_fn = None
            host_cuts = self._sample_value_sequence(
                info["host_cut_ctx"]["candidate_ids"],
                info["host_cut_logits"][0],
                info["host_cut_logits"][1],
                max_selected=self.max_cut_values,
                allow_empty=True,
                temperature=region_temperature,
                exploration=self.region_exploration,
                step_mask_fn=cut_step_fn,
            )
            info["host_selected_cuts"] = host_cuts
            host_sampler_attempts += 1
            host_build = self._build_boundary_region(
                info["host_ctx"]["ir"],
                output_values=info["host_selected_outputs"],
                cut_values=host_cuts,
                disable_repair=host_use_mask,
            )
            if host_build is None:
                invalid_regions["host_region_build_failed"] = invalid_regions.get("host_region_build_failed", 0) + 1
                continue
            host_built_regions += 1
            host_region, host_validity = host_build
            if not host_validity.is_valid:
                key = f"host_{host_validity.reason}"
                invalid_regions[key] = invalid_regions.get(key, 0) + 1
                continue
            host_validate_passes += 1
            # Live-region precondition: at least one of the region's
            # exit_values must lie on the host's return-slice. Otherwise
            # whatever the donor produces here is dead code and DCE will
            # delete it (manifests downstream as ``structural_fail``).
            host_return_slice = info["host_ctx"].get("return_slice_values")
            if host_return_slice is not None:
                exit_set = set(host_region.exit_values or [])
                if exit_set and not (exit_set & host_return_slice):
                    invalid_regions["host_region_dead_code"] = (
                        invalid_regions.get("host_region_dead_code", 0) + 1
                    )
                    continue
            try:
                host_contract = infer_boundary_contract(info["host_ctx"]["ir"], host_region)
            except Exception:
                invalid_regions["host_contract_failed"] = invalid_regions.get("host_contract_failed", 0) + 1
                continue
            info["host_region"] = host_region
            info["host_validity"] = host_validity
            info["host_contract"] = host_contract
            info["host_effective_outputs"] = list(host_region.provenance.get("effective_output_values", info["host_selected_outputs"]))
            info["host_effective_cuts"] = list(host_region.provenance.get("effective_cut_values", host_cuts))
            # M6a §6.7: derive the typed boundary signature once the host
            # region is built. The signature drives the donor-side type
            # masks and is stamped on the proposal so the engine can log
            # Case I/II/III dispatch counts and the trainer can record
            # mask-empty aborts as negative samples.
            host_signature = signature_for_region(info["host_ctx"]["ir"], host_region)
            info["host_signature"] = host_signature
            host_region_metrics.append((host_validity.n_ops, host_validity.n_inputs, host_validity.n_outputs))
            effective_cut_sizes.append(len(info["host_effective_cuts"]))
            host_valid_infos.append(info)

        # Layer D1: donor-pool signature-compatibility prefilter.
        # Drops (host, donor) pairs where the donor cannot possibly
        # satisfy the host signature on at least one port type.  This
        # cuts the signature_mask_empty rate by removing donors that
        # would deterministically fail later inside
        # ``_sample_donor_under_signature``. Disabled when
        # ``enable_donor_mask`` is False.
        donor_use_mask = self.enable_donor_mask
        if donor_use_mask:
            survivors: list[dict[str, Any]] = []
            for info in host_valid_infos:
                sig = info["host_signature"]
                donor_ctx = info["donor_ctx"]
                if _donor_pool_signature_compatible(
                    donor_ctx["ir"],
                    donor_ctx["observable_values"],
                    donor_ctx.get("donor_cut_pool_union", []),
                    entry_types=sig.entry_types,
                    exit_types=sig.exit_types,
                ):
                    survivors.append(info)
                else:
                    invalid_regions["donor_pool_signature_mismatch"] = (
                        invalid_regions.get("donor_pool_signature_mismatch", 0) + 1
                    )
            host_valid_infos = survivors

        # Donor outputs (batched logits — masking is applied per-step
        # inside ``_sample_donor_under_signature`` below).
        for info, out in zip(
            host_valid_infos,
            self._compute_output_logits_batch([
                {
                    "candidate_feats_np": info["donor_ctx"]["observable_feats_np"],
                    "context_emb_np": info["host_ctx"]["emb"].detach().cpu().numpy(),
                }
                for info in host_valid_infos
            ]),
        ):
            info["donor_output_logits"] = out

        # Per-host signature-driven donor sampling. This replaces the
        # previous "sample outputs → batch cut logits → sample cuts"
        # flow with a positional, type-masked sampler. Aborts (all-zero
        # mask at any step) are recorded as negative training samples
        # under ``signature_mask_empty`` and the proposal is dropped.
        donor_cut_infos: list[dict[str, Any]] = []
        donor_sampler_attempts = 0
        _donor_samp_iter: Any = (
            tqdm(host_valid_infos, desc="Sample donor regions", leave=False)
            if _show else host_valid_infos
        )
        for info in _donor_samp_iter:
            sig: BoundarySignature = info["host_signature"]
            donor_sampler_attempts += 1
            sampled = self._sample_donor_under_signature(
                info, sig, donor_temperature,
            )
            if sampled is None:
                invalid_regions["signature_mask_empty"] = (
                    invalid_regions.get("signature_mask_empty", 0) + 1
                )
                continue
            donor_outputs, donor_cuts = sampled
            if not donor_outputs:
                invalid_regions["donor_no_output"] = invalid_regions.get("donor_no_output", 0) + 1
                continue
            info["donor_selected_outputs"] = donor_outputs
            info["donor_selected_cuts"] = donor_cuts
            donor_cut_infos.append(info)

        proposals: list[GraftProposal] = []
        donor_built_regions = 0
        donor_validate_passes = 0
        _donor_build_iter: Any = (
            tqdm(donor_cut_infos, desc="Build donor regions", leave=False)
            if _show else donor_cut_infos
        )
        for info in _donor_build_iter:
            donor_cuts = info["donor_selected_cuts"]
            donor_build = self._build_boundary_region(
                info["donor_ctx"]["ir"],
                output_values=info["donor_selected_outputs"],
                cut_values=donor_cuts,
            )
            if donor_build is None:
                invalid_regions["donor_region_build_failed"] = invalid_regions.get("donor_region_build_failed", 0) + 1
                continue
            donor_region, donor_validity = donor_build
            donor_built_regions += 1
            if not donor_validity.is_valid:
                key = f"donor_{donor_validity.reason}"
                invalid_regions[key] = invalid_regions.get(key, 0) + 1
                continue
            donor_validate_passes += 1
            try:
                donor_trim = extract_region_ir(info["donor_ctx"]["ir"], donor_region)
            except Exception:
                invalid_regions["donor_extract_failed"] = invalid_regions.get("donor_extract_failed", 0) + 1
                continue
            info["donor_region"] = donor_region
            info["donor_validity"] = donor_validity
            info["donor_effective_outputs"] = list(donor_region.provenance.get("effective_output_values", info["donor_selected_outputs"]))
            info["donor_effective_cuts"] = list(donor_region.provenance.get("effective_cut_values", donor_cuts))
            # M6a §6.4 + §6.5: classify the host region against the
            # host's slot_meta so the engine can dispatch through the
            # correct Case I / II / III pipeline.
            host_ir = info["host_ctx"]["ir"]
            classification = classify_region(
                info["host_region"], host_ir.slot_meta or {}, host_ir,
            )
            info["region_classification"] = classification
            proposal = self._make_boundary_proposal(
                info,
                donor_trim=donor_trim,
                region_temperature=region_temperature,
                donor_temperature=donor_temperature,
            )
            if proposal is not None:
                proposals.append(proposal)

        return proposals, self._proposal_stats(
            invalid_regions=invalid_regions,
            host_region_metrics=host_region_metrics,
            effective_cut_sizes=effective_cut_sizes,
            attempted=len(pair_infos),
            host_sampler_attempts=host_sampler_attempts,
            host_built_regions=host_built_regions,
            host_validate_passes=host_validate_passes,
            host_use_mask=host_use_mask,
            donor_sampler_attempts=donor_sampler_attempts,
            donor_built_regions=donor_built_regions,
            donor_validate_passes=donor_validate_passes,
            donor_use_mask=donor_use_mask,
        )

    def _make_boundary_proposal(
        self,
        info: dict[str, Any],
        *,
        donor_trim: FunctionIR,
        region_temperature: float,
        donor_temperature: float,
    ) -> GraftProposal | None:
        proposal_id = _fresh_id("gnn_graft")
        host_region: RewriteRegion = info["host_region"]
        donor_region: RewriteRegion = info["donor_region"]
        self._append_experience({
            "proposal_id": proposal_id,
            "host_algo": info["host_entry"].algo_id,
            "donor_algo": info["donor_entry"].algo_id,
            "predicted_graft_score": float(info["pair_score"]),
            "generation": self._generation,
            "host_context_emb": info["donor_ctx"]["emb"].detach().cpu().numpy().astype(np.float32),
            "donor_context_emb": info["host_ctx"]["emb"].detach().cpu().numpy().astype(np.float32),
            "host_output_candidates": list(info["host_ctx"]["observable_values"]),
            "host_output_feats": np.array(info["host_ctx"]["observable_feats_np"], copy=True),
            "host_selected_outputs": list(info["host_selected_outputs"]),
            "host_effective_outputs": list(info["host_effective_outputs"]),
            "host_cut_candidates": list(info["host_cut_ctx"]["candidate_ids"]),
            "host_cut_feats": np.array(info["host_cut_ctx"]["candidate_feats_np"], copy=True),
            "host_selected_cuts": list(info["host_selected_cuts"]),
            "host_effective_cuts": list(info["host_effective_cuts"]),
            "host_region_validity": info["host_validity"].reason,
            "donor_output_candidates": list(info["donor_ctx"]["observable_values"]),
            "donor_output_feats": np.array(info["donor_ctx"]["observable_feats_np"], copy=True),
            "donor_selected_outputs": list(info["donor_selected_outputs"]),
            "donor_effective_outputs": list(info["donor_effective_outputs"]),
            "donor_cut_candidates": list(info["donor_cut_ctx"]["candidate_ids"]),
            "donor_cut_feats": np.array(info["donor_cut_ctx"]["candidate_feats_np"], copy=True),
            "donor_selected_cuts": list(info["donor_selected_cuts"]),
            "donor_effective_cuts": list(info["donor_effective_cuts"]),
            "donor_region_validity": info["donor_validity"].reason,
            "host_temperature": float(region_temperature),
            "donor_temperature": float(donor_temperature),
            "contract_signature": dict(info["host_contract"].port_signature),
        })
        confidence = float(1.0 / (1.0 + max(float(info["pair_score"]), 0.0)))
        classification = info.get("region_classification")
        signature = info.get("host_signature")
        case = classification.case if classification is not None else "II"
        attrib_key = (
            classification.attribution_slot_pop_key
            if classification is not None else None
        )
        half_cut = (
            classification.half_cut_slots
            if classification is not None else ()
        )
        return GraftProposal(
            proposal_id=proposal_id,
            host_algo_id=info["host_entry"].algo_id,
            region=host_region,
            contract=info["host_contract"],
            donor_algo_id=info["donor_entry"].algo_id,
            donor_ir=donor_trim,
            donor_region=donor_region,
            dependency_overrides=[],
            confidence=confidence,
            rationale=(
                f"BCIR graft (case {case}): "
                f"host_out={len(info['host_effective_outputs'])} "
                f"host_cut={len(info['host_effective_cuts'])} "
                f"donor_out={len(info['donor_effective_outputs'])} "
                f"donor_cut={len(info['donor_effective_cuts'])} "
                f"pred_score={float(info['pair_score']):.3f}"
            ),
            case=case,
            attribution_slot_pop_key=attrib_key,
            boundary_signature=signature,
            half_cut_slots=tuple(half_cut),
        )

    def _append_experience(self, record: dict[str, Any]) -> None:
        self._experience.append(record)
        self._total_proposals += 1
        if len(self._experience) > self.buffer_size:
            self._experience = self._experience[-self.buffer_size:]
            live_pids = {exp["proposal_id"] for exp in self._experience}
            self._outcomes = {
                pid: outcome for pid, outcome in self._outcomes.items()
                if pid in live_pids
            }

    # ------------------------------------------------------------------
    # Context / batching helpers
    # ------------------------------------------------------------------

    def _get_entry_context(
        self,
        entry: AlgorithmEntry,
        cache: dict[str, dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        cached = cache.get(entry.algo_id)
        if cached is not None or entry.algo_id in cache:
            return cached

        ir = entry.ir
        graph = self._graph_cache.get(entry.algo_id)
        emb = self._emb_cache.get(entry.algo_id)
        observable_values = enumerate_observable_values(ir)
        if graph is None or emb is None or not observable_values:
            cache[entry.algo_id] = None
            return None

        observable_set = set(observable_values)
        observable_feats_np = self._get_value_feats(ir, observable_values, observable_set)
        if observable_feats_np.shape[0] == 0:
            cache[entry.algo_id] = None
            return None

        ctx = {
            "entry": entry,
            "ir": ir,
            "graph": graph,
            "emb": emb,
            "observable_values": observable_values,
            "observable_set": observable_set,
            "observable_feats_np": observable_feats_np,
            "cut_candidate_cache": {},
            # Live-region precondition: only regions whose exit_values
            # intersect the host's return-slice produce non-dead grafts.
            "return_slice_values": _compute_return_slice_values(ir),
        }
        # ── Host-mask precompute ────────────────────────────────────
        # Layer 1: drop dead-code observable values from the host-side
        # candidate pool.  We retain the original ``observable_values``
        # (used for the donor side and for any code path that does not
        # opt in to host masking) and expose a parallel filtered view
        # under ``host_*`` keys.  Layer 2 also needs per-value op
        # closures for connectivity checks.
        host_obs_vals, host_kept_idx = _host_filter_dead_code_outputs(
            observable_values, ctx["return_slice_values"],
        )
        if observable_feats_np.shape[0] >= len(observable_values) and host_kept_idx:
            host_obs_feats_np = observable_feats_np[host_kept_idx]
        else:
            host_obs_feats_np = observable_feats_np[:0]
        ctx["host_observable_values"] = host_obs_vals
        ctx["host_observable_feats_np"] = host_obs_feats_np
        ctx["host_observable_kept_idx"] = host_kept_idx
        ctx["host_op_closures"] = _host_precompute_op_closures(ir, host_obs_vals)
        # ── Donor-mask precompute ───────────────────────────────────
        # Layer D2 needs op closures over the FULL observable pool
        # (donor uses unfiltered observables since dead-code on donor
        # side has different semantics — its outputs become live once
        # grafted).  Layer D1 also needs the union of cut candidates
        # over observable singletons; cap to keep prefilter cheap.
        ctx["donor_op_closures"] = _host_precompute_op_closures(ir, observable_values)
        ctx["donor_cut_pool_union"] = _donor_cut_pool_union(ir, observable_values)
        cache[entry.algo_id] = ctx
        return ctx

    def _get_cut_context(
        self,
        ctx: dict[str, Any],
        output_values: list[str],
    ) -> dict[str, Any]:
        key = tuple(output_values)
        cached = ctx["cut_candidate_cache"].get(key)
        if cached is not None:
            return cached
        candidate_ids = enumerate_cut_candidates(ctx["ir"], output_values)
        candidate_feats_np = self._get_value_feats(
            ctx["ir"],
            candidate_ids,
            ctx["observable_set"],
        )
        cut_ctx = {
            "candidate_ids": candidate_ids,
            "candidate_feats_np": candidate_feats_np,
        }
        ctx["cut_candidate_cache"][key] = cut_ctx
        return cut_ctx

    def _compute_output_logits_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not requests:
            return []

        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        chunk = self.proposal_batch_size if self.enable_batched_proposals else 1
        with torch.no_grad():
            for start in range(0, len(requests), chunk):
                batch_requests = requests[start:start + chunk]
                feat_batch, feat_mask = self._pad_feature_batch(
                    [req["candidate_feats_np"] for req in batch_requests],
                    _VALUE_FEAT_DIM,
                )
                ctx_batch = torch.tensor(
                    np.stack([req["context_emb_np"] for req in batch_requests], axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
                feat_batch = feat_batch.to(self.device)
                feat_mask = feat_mask.to(self.device)
                batch_size, max_len, feat_dim = feat_batch.shape
                value_embs = self.boundary_region_policy.encode_values(
                    feat_batch.reshape(-1, feat_dim)
                ).reshape(batch_size, max_len, -1)
                ctx_exp = ctx_batch.unsqueeze(1).expand(-1, max_len, -1)
                logits = self.boundary_region_policy.output_head(
                    torch.cat([value_embs, ctx_exp], dim=-1)
                ).squeeze(-1)
                pooled = (value_embs * feat_mask.unsqueeze(-1)).sum(dim=1)
                denom = feat_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                pooled = pooled / denom
                stop_logits = self.boundary_region_policy.output_stop_head(
                    torch.cat([ctx_batch, pooled], dim=-1)
                ).reshape(-1)
                lengths = feat_mask.sum(dim=1).long().tolist()
                for row_index, length in enumerate(lengths):
                    outputs.append((
                        logits[row_index, :length].detach().cpu(),
                        stop_logits[row_index].detach().cpu(),
                    ))
        return outputs

    def _compute_cut_logits_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not requests:
            return []

        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        chunk = self.proposal_batch_size if self.enable_batched_proposals else 1
        hidden_dim = self.boundary_region_policy.value_encoder[0].out_features
        with torch.no_grad():
            for start in range(0, len(requests), chunk):
                batch_requests = requests[start:start + chunk]
                feat_batch, feat_mask = self._pad_feature_batch(
                    [req["candidate_feats_np"] for req in batch_requests],
                    _VALUE_FEAT_DIM,
                )
                ctx_batch = torch.tensor(
                    np.stack([req["context_emb_np"] for req in batch_requests], axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
                output_summary = torch.tensor(
                    np.stack([req["output_summary_np"] for req in batch_requests], axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
                if output_summary.shape[1] != hidden_dim:
                    raise ValueError("BCIR output summary dimension mismatch")
                feat_batch = feat_batch.to(self.device)
                feat_mask = feat_mask.to(self.device)
                batch_size, max_len, feat_dim = feat_batch.shape
                value_embs = self.boundary_region_policy.encode_values(
                    feat_batch.reshape(-1, feat_dim)
                ).reshape(batch_size, max_len, -1)
                ctx_exp = ctx_batch.unsqueeze(1).expand(-1, max_len, -1)
                out_exp = output_summary.unsqueeze(1).expand(-1, max_len, -1)
                logits = self.boundary_region_policy.cut_head(
                    torch.cat([value_embs, ctx_exp, out_exp], dim=-1)
                ).squeeze(-1)
                stop_logits = self.boundary_region_policy.cut_stop_head(
                    torch.cat([ctx_batch, output_summary], dim=-1)
                ).reshape(-1)
                lengths = feat_mask.sum(dim=1).long().tolist()
                for row_index, length in enumerate(lengths):
                    outputs.append((
                        logits[row_index, :length].detach().cpu(),
                        stop_logits[row_index].detach().cpu(),
                    ))
        return outputs

    def _pad_feature_batch(
        self,
        arrays: list[np.ndarray],
        feat_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max((int(arr.shape[0]) for arr in arrays), default=0)
        if max_len <= 0:
            return (
                torch.zeros((len(arrays), 1, feat_dim), dtype=torch.float32),
                torch.zeros((len(arrays), 1), dtype=torch.float32),
            )
        batch = np.zeros((len(arrays), max_len, feat_dim), dtype=np.float32)
        mask = np.zeros((len(arrays), max_len), dtype=np.float32)
        for index, arr in enumerate(arrays):
            if arr.size == 0:
                continue
            batch[index, :arr.shape[0], :] = arr
            mask[index, :arr.shape[0]] = 1.0
        return torch.tensor(batch, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def _policy_temperature(self) -> float:
        return max(0.3, 2.0 - 0.04 * self._generation)

    # ------------------------------------------------------------------
    # M6a §6.7 — Signature-driven donor sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _value_static_type(ir: FunctionIR, value_id: str) -> str:
        """Best-effort static type tag for a value (mirrors graft_classifier)."""
        val = ir.values.get(value_id)
        if val is None:
            return "unknown"
        type_hint = getattr(val, "type_hint", None)
        if type_hint:
            return str(type_hint)
        attrs = getattr(val, "attrs", None) or {}
        for key in ("type", "static_type", "dtype"):
            v = attrs.get(key)
            if v:
                return str(v)
        return "unknown"

    @staticmethod
    def _types_compatible(donor_t: str, host_t: str) -> bool:
        """Lattice subtype check tolerant of unknown / any tags.

        ``unknown`` (or empty) on either side is treated as a wildcard so
        the mask never fully zeroes a candidate purely for missing static
        type info — the GNN policy still has to score it. The lattice
        check is applied positionally (donor → host) so the typed-binder
        downstream sees a port mapping it can accept.
        """
        if not host_t or host_t in ("unknown", "any", "object"):
            return True
        if not donor_t or donor_t in ("unknown", "any", "object"):
            return True
        try:
            return is_subtype(donor_t, host_t) or is_subtype(host_t, donor_t)
        except Exception:
            return True  # fall back to permissive on lattice errors

    def _build_donor_mask(
        self,
        donor_ir: FunctionIR,
        candidate_ids: list[str],
        already_chosen_idx: set[int],
        host_type: str,
    ) -> torch.Tensor:
        """Per-step positional mask for donor candidates.

        Returns a 1-D BoolTensor of length ``len(candidate_ids)``. Index
        ``i`` is ``True`` iff (a) it has not already been picked at an
        earlier step, and (b) the candidate's type is lattice-compatible
        with ``host_type``.
        """
        mask = torch.zeros(len(candidate_ids), dtype=torch.bool)
        for i, vid in enumerate(candidate_ids):
            if i in already_chosen_idx:
                continue
            cand_t = self._value_static_type(donor_ir, vid)
            if self._types_compatible(cand_t, host_type):
                mask[i] = True
        return mask

    def _sample_under_signature_step(
        self,
        candidate_ids: list[str],
        full_logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float,
    ) -> int | None:
        """Sample a single index from ``full_logits`` masked to eligible.

        Returns ``None`` iff ``mask`` is all-zero — the caller must treat
        this as an "all-zero mask abort" (negative training sample, no
        fallback per §6.7).
        """
        if not bool(mask.any()):
            return None
        # Apply mask to the policy's already-computed logits.
        logits = full_logits.detach().clone()
        if logits.shape[0] != mask.shape[0]:
            # Defensive: lengths must match. Treat as a hard abort.
            return None
        logits = logits.masked_fill(~mask, float("-inf"))
        temp = max(float(temperature), 1e-3)
        scaled = logits / temp
        # Softmax with safe handling of all -inf would still be unsafe
        # (already guarded by ``mask.any()``).
        probs = torch.softmax(scaled, dim=0)
        # Numerical guard: replace any NaN with zero on masked-out rows.
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if float(probs.sum().item()) <= 0.0:
            return None
        idx = int(torch.multinomial(probs, 1).item())
        return idx

    def _sample_donor_under_signature(
        self,
        info: dict[str, Any],
        signature: BoundarySignature,
        donor_temperature: float,
    ) -> tuple[list[str], list[str]] | None:
        """Sample donor outputs/cuts per :class:`BoundarySignature` (§6.7).

        Wraps a single-attempt sampler in a retry loop: greedy random
        masked sampling can paint itself into dead-end output choices
        (e.g. picking 3 outputs whose only feasible 4th candidate doesn't
        exist).  Re-sampling from scratch with a fresh RNG state typically
        recovers, since most failures are non-deterministic given the
        masked candidate set.
        """
        max_retries = getattr(self, "donor_sample_max_retries", 10)
        for _ in range(max_retries):
            result = self._sample_donor_under_signature_once(
                info, signature, donor_temperature,
            )
            if result is not None:
                return result
        return None

    def _sample_donor_under_signature_once(
        self,
        info: dict[str, Any],
        signature: BoundarySignature,
        donor_temperature: float,
    ) -> tuple[list[str], list[str]] | None:
        """Single-attempt donor sampler.  See :meth:`_sample_donor_under_signature`."""
        donor_ctx = info["donor_ctx"]
        donor_ir = donor_ctx["ir"]
        observable_values: list[str] = list(donor_ctx["observable_values"])
        donor_use_mask = self.enable_donor_mask
        donor_op_closures = donor_ctx.get("donor_op_closures", {}) if donor_use_mask else {}
        # ── Step 1: outputs, positional over signature.exit_types ──
        if not signature.exit_types:
            return None
        full_out_logits = info["donor_output_logits"][0]
        if full_out_logits.shape[0] != len(observable_values):
            return None
        chosen_out_idx: set[int] = set()
        selected_outputs: list[str] = []
        # When the donor mask is enabled we MUST satisfy the signature
        # exactly (equality arity), so do not truncate to
        # ``max_boundary_outputs``: if the host signature has more
        # exits than the cap we should abort the proposal instead of
        # silently producing a smaller donor region.
        if donor_use_mask and len(signature.exit_types) > self.max_boundary_outputs:
            return None
        n_out = (
            len(signature.exit_types) if donor_use_mask
            else min(len(signature.exit_types), self.max_boundary_outputs)
        )
        for step in range(n_out):
            host_t = signature.exit_types[step]
            mask = self._build_donor_mask(
                donor_ir, observable_values, chosen_out_idx, host_t,
            )
            if donor_use_mask:
                # Layer D2: AND the type mask with the connectivity +
                # feasibility mask (at the final-output step the
                # feasibility lookahead becomes the dominant filter).
                d2 = _donor_output_step_mask(
                    donor_ir, donor_op_closures, selected_outputs,
                    observable_values,
                    next_step=step,
                    entry_types=signature.entry_types,
                    exit_types=signature.exit_types,
                    max_region_ops=self.max_region_ops,
                    min_region_ops=self.min_region_size,
                    max_region_inputs=self.max_region_inputs,
                    max_cut_budget=self.max_cut_values,
                )
                d2_t = torch.tensor(d2, dtype=torch.bool)
                # Already-chosen indices were cleared by _build_donor_mask;
                # combine via AND positionally.
                if d2_t.shape[0] == mask.shape[0]:
                    mask = mask & d2_t
            idx = self._sample_under_signature_step(
                observable_values, full_out_logits, mask, donor_temperature,
            )
            if idx is None:
                return None  # all-zero mask → abort
            chosen_out_idx.add(idx)
            selected_outputs.append(observable_values[idx])
        # ── Step 2: cuts, positional over signature.entry_types ──
        cut_ctx = self._get_cut_context(donor_ctx, selected_outputs)
        cut_ids: list[str] = list(cut_ctx["candidate_ids"])
        info["donor_cut_ctx"] = cut_ctx
        # Compute cut logits for this donor info now that outputs are fixed.
        cut_logit_pair_list = self._compute_cut_logits_batch([
            {
                "candidate_feats_np": cut_ctx["candidate_feats_np"],
                "context_emb_np": info["host_ctx"]["emb"].detach().cpu().numpy(),
                "output_summary_np": self._summarize_selected_value_feats(
                    observable_values,
                    donor_ctx["observable_feats_np"],
                    selected_outputs,
                ),
            }
        ])
        if not cut_logit_pair_list:
            return None
        info["donor_cut_logits"] = cut_logit_pair_list[0]
        full_cut_logits = info["donor_cut_logits"][0]
        if full_cut_logits.shape[0] != len(cut_ids):
            # Mismatch can only happen for empty cut sets; treat as abort.
            if not cut_ids and not signature.entry_types:
                return selected_outputs, []
            return None
        chosen_cut_idx: set[int] = set()
        selected_cuts: list[str] = []
        if donor_use_mask and len(signature.entry_types) > self.max_cut_values:
            return None
        # Donor cut sampling under Layer D4: loop until STOP gate
        # opens (current state matches arity exactly) or all-zero mask.
        max_cut_attempts = (
            len(signature.entry_types) if donor_use_mask
            else min(len(signature.entry_types), self.max_cut_values)
        )
        if not donor_use_mask:
            for step in range(max_cut_attempts):
                host_t = signature.entry_types[step]
                mask = self._build_donor_mask(
                    donor_ir, cut_ids, chosen_cut_idx, host_t,
                )
                idx = self._sample_under_signature_step(
                    cut_ids, full_cut_logits, mask, donor_temperature,
                )
                if idx is None:
                    return None
                chosen_cut_idx.add(idx)
                selected_cuts.append(cut_ids[idx])
            return selected_outputs, selected_cuts
        # Mask-on path: re-evaluate D4 each step to pick up STOP gate.
        for step in range(max_cut_attempts):
            d4_mask, stop_allowed = _donor_cut_step_mask(
                donor_ir, selected_outputs, selected_cuts, cut_ids,
                entry_types=signature.entry_types,
                exit_types=signature.exit_types,
                max_region_ops=self.max_region_ops,
                min_region_ops=self.min_region_size,
                max_region_inputs=self.max_region_inputs,
                max_cut_budget=self.max_cut_values,
                cut_pool=cut_ids,
            )
            # If region is already feasible, prefer STOP to avoid
            # over-cutting (which can push n_ops below min_region_ops
            # or otherwise destabilize the region).
            if stop_allowed and step > 0:
                break
            host_t = signature.entry_types[step]
            type_mask = self._build_donor_mask(
                donor_ir, cut_ids, chosen_cut_idx, host_t,
            )
            d4_t = torch.tensor(d4_mask, dtype=torch.bool)
            if d4_t.shape[0] == type_mask.shape[0]:
                final_mask = type_mask & d4_t
            else:
                final_mask = type_mask
            if not bool(final_mask.any()):
                if stop_allowed:
                    break
                return None
            idx = self._sample_under_signature_step(
                cut_ids, full_cut_logits, final_mask, donor_temperature,
            )
            if idx is None:
                if stop_allowed:
                    break
                return None
            chosen_cut_idx.add(idx)
            selected_cuts.append(cut_ids[idx])
        return selected_outputs, selected_cuts

    def _build_boundary_region(
        self,
        ir: FunctionIR,
        *,
        output_values: list[str],
        cut_values: list[str],
        disable_repair: bool = False,
    ) -> tuple[RewriteRegion, Any] | None:
        # Greedy connectivity-aware cut pruning.  Even if each cut
        # individually preserves connectivity (enforced by
        # ``enumerate_cut_candidates(require_connected=True)``), a
        # *combination* of cuts can sever the slice into multiple
        # islands.  When this happens we iteratively drop the cut whose
        # removal restores connectivity, until either the region is
        # connected or no cuts remain.  This is a generic structural
        # repair, not a graft-pattern prior.
        #
        # When ``disable_repair=True`` (host-mask path) we skip the
        # greedy fallback entirely so that the reported host validity
        # rate reflects the mask quality, not the fallback's heroics.
        cuts_to_use = list(cut_values)
        try:
            region = define_rewrite_region(
                ir,
                boundary_spec=BoundaryRegionSpec(
                    output_values=list(output_values),
                    cut_values=list(cuts_to_use),
                ),
            )
        except Exception:
            return None
        validity = validate_boundary_region(
            ir,
            region,
            min_region_ops=self.min_region_size,
            max_region_ops=self.max_region_ops,
            max_region_inputs=self.max_region_inputs,
            max_region_outputs=self.max_region_outputs,
        )
        if disable_repair:
            return region, validity
        if not validity.is_valid and validity.reason == "disconnected_region" and cuts_to_use:
            # Try dropping cuts one at a time, prefer to drop the one
            # whose removal yields a still-valid (or at least connected)
            # region.
            for _ in range(len(cuts_to_use)):
                best_alt: tuple[Any, Any, list[str]] | None = None
                for i in range(len(cuts_to_use)):
                    trial = cuts_to_use[:i] + cuts_to_use[i + 1 :]
                    try:
                        alt_region = define_rewrite_region(
                            ir,
                            boundary_spec=BoundaryRegionSpec(
                                output_values=list(output_values),
                                cut_values=list(trial),
                            ),
                        )
                    except Exception:
                        continue
                    alt_validity = validate_boundary_region(
                        ir,
                        alt_region,
                        min_region_ops=self.min_region_size,
                        max_region_ops=self.max_region_ops,
                        max_region_inputs=self.max_region_inputs,
                        max_region_outputs=self.max_region_outputs,
                    )
                    if alt_validity.reason != "disconnected_region":
                        best_alt = (alt_region, alt_validity, trial)
                        break
                if best_alt is None:
                    break
                region, validity, cuts_to_use = best_alt
                if validity.is_valid:
                    break
        # If still disconnected after dropping all cuts, the outputs
        # themselves come from disjoint dataflow components.  Drop
        # outputs greedily (smallest sub-component first) until the
        # remaining outputs share a connected backward slice.
        outputs_to_use = list(output_values)
        repair_reasons = {
            "disconnected_region",
            "too_large",
            "too_many_outputs",
            "too_many_inputs",
        }
        if (
            not validity.is_valid
            and validity.reason in repair_reasons
            and len(outputs_to_use) > 1
        ):
            for _ in range(len(outputs_to_use) - 1):
                best_alt = None
                best_metric: tuple[int, int, int] | None = None
                for i in range(len(outputs_to_use)):
                    trial_outputs = outputs_to_use[:i] + outputs_to_use[i + 1 :]
                    try:
                        alt_region = define_rewrite_region(
                            ir,
                            boundary_spec=BoundaryRegionSpec(
                                output_values=list(trial_outputs),
                                cut_values=list(cuts_to_use),
                            ),
                        )
                    except Exception:
                        continue
                    alt_validity = validate_boundary_region(
                        ir,
                        alt_region,
                        min_region_ops=self.min_region_size,
                        max_region_ops=self.max_region_ops,
                        max_region_inputs=self.max_region_inputs,
                        max_region_outputs=self.max_region_outputs,
                    )
                    metric = (
                        alt_validity.n_ops,
                        alt_validity.n_outputs,
                        alt_validity.n_inputs,
                    )
                    if alt_validity.is_valid:
                        best_alt = (alt_region, alt_validity, trial_outputs)
                        best_metric = metric
                        break
                    if alt_validity.reason in repair_reasons and (
                        best_metric is None or metric < best_metric
                    ):
                        best_alt = (alt_region, alt_validity, trial_outputs)
                        best_metric = metric
                if best_alt is None:
                    break
                region, validity, outputs_to_use = best_alt
                if validity.is_valid:
                    break
        return region, validity

    def _sample_value_sequence(
        self,
        candidate_ids: list[str],
        logits: torch.Tensor,
        stop_logit: torch.Tensor,
        *,
        max_selected: int,
        allow_empty: bool,
        temperature: float,
        exploration: float,
        step_mask_fn: "Callable[[list[str], list[str]], tuple[list[bool], bool]] | None" = None,
    ) -> list[str]:
        """Sample up to ``max_selected`` value ids by per-step softmax.

        If ``step_mask_fn`` is provided, it is called once per step with
        ``(selected_so_far_value_ids, remaining_value_ids)`` and must
        return ``(per_remaining_bool_mask, stop_allowed)``.  Masked-out
        candidates are excluded from the softmax (set to ``-inf``).  If
        ``stop_allowed`` is False the ``__STOP__`` option is suppressed
        even when ``allow_empty`` would otherwise admit it.

        If after applying the mask no candidate is allowed and STOP is
        also forbidden, the sampler aborts and returns ``selected`` as
        is — the caller can detect this dead-end (e.g. by checking
        ``len(selected) < max_selected and len(available) > 0``).
        """
        if not candidate_ids or logits.numel() == 0:
            return []
        available = list(range(len(candidate_ids)))
        selected: list[str] = []
        stop_value = float(stop_logit.reshape(-1)[0].item())
        for step in range(max_selected):
            if not available:
                break
            remaining_ids = [candidate_ids[idx] for idx in available]
            if step_mask_fn is not None:
                mask, stop_allowed = step_mask_fn(list(selected), remaining_ids)
            else:
                mask = [True] * len(remaining_ids)
                stop_allowed = True
            kept_local: list[int] = [i for i, ok in enumerate(mask) if ok]
            if not kept_local:
                # Nothing legal to pick; if STOP is allowed, emit it;
                # otherwise this is a dead-end and we return what we have.
                if (allow_empty or step > 0) and stop_allowed:
                    break
                if step > 0:  # at least one already chosen — prefer to stop than dead-end
                    break
                return selected  # dead-end on step 0: return empty
            option_logits = [float(logits[available[i]].item()) for i in kept_local]
            option_ids = [remaining_ids[i] for i in kept_local]
            if (allow_empty or step > 0) and stop_allowed:
                option_logits.append(stop_value)
                option_ids.append("__STOP__")
            probs = self._sample_probs_from_logits(
                option_logits,
                temperature=temperature,
                exploration=exploration,
            )
            chosen = int(np.random.choice(len(option_ids), p=probs))
            picked = option_ids[chosen]
            if picked == "__STOP__":
                break
            selected.append(picked)
            local_idx = kept_local[chosen]
            remove_idx = available[local_idx]
            available = [idx for idx in available if idx != remove_idx]
        return selected

    def _sample_probs_from_logits(
        self,
        logits: list[float],
        *,
        temperature: float,
        exploration: float,
    ) -> np.ndarray:
        logits_np = np.array(logits, dtype=np.float64)
        # Replace NaN/Inf with very-negative finite values.
        logits_np = np.where(np.isfinite(logits_np), logits_np, -1e9)
        logits_np = np.clip(logits_np, -1e9, 1e9)
        logits_np = logits_np - np.max(logits_np)
        probs = np.exp(logits_np / max(float(temperature), 1e-4))
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            probs = np.full(logits_np.shape[0], 1.0 / max(logits_np.shape[0], 1), dtype=np.float64)
        else:
            probs = probs / total
        eps = float(np.clip(exploration, 0.0, 1.0))
        if eps > 0:
            probs = (1.0 - eps) * probs + eps / probs.shape[0]
        total2 = probs.sum()
        if not np.isfinite(total2) or total2 <= 0:
            probs = np.full(probs.shape[0], 1.0 / probs.shape[0], dtype=np.float64)
            return probs
        return probs / total2

    def _proposal_stats(
        self,
        *,
        invalid_regions: dict[str, int],
        host_region_metrics: list[tuple[int, int, int]],
        effective_cut_sizes: list[int],
        attempted: int,
        host_sampler_attempts: int = 0,
        host_built_regions: int = 0,
        host_validate_passes: int = 0,
        host_use_mask: bool = False,
        donor_sampler_attempts: int = 0,
        donor_built_regions: int = 0,
        donor_validate_passes: int = 0,
        donor_use_mask: bool = False,
    ) -> dict[str, Any]:
        if attempted <= 0:
            return self._empty_proposal_stats()
        mean_ops = float(np.mean([m[0] for m in host_region_metrics])) if host_region_metrics else 0.0
        mean_inputs = float(np.mean([m[1] for m in host_region_metrics])) if host_region_metrics else 0.0
        mean_outputs = float(np.mean([m[2] for m in host_region_metrics])) if host_region_metrics else 0.0
        mean_cut = float(np.mean(effective_cut_sizes)) if effective_cut_sizes else 0.0
        host_validity_rate = (
            host_validate_passes / host_sampler_attempts
            if host_sampler_attempts > 0
            else 0.0
        )
        donor_validity_rate = (
            donor_validate_passes / donor_sampler_attempts
            if donor_sampler_attempts > 0
            else 0.0
        )
        end_to_end_validity_rate = (
            donor_validate_passes / host_sampler_attempts
            if host_sampler_attempts > 0
            else 0.0
        )
        return {
            "invalid_region_rate": float(sum(invalid_regions.values()) / attempted),
            "invalid_regions": dict(invalid_regions),
            "mean_region_ops": mean_ops,
            "mean_region_inputs": mean_inputs,
            "mean_region_outputs": mean_outputs,
            "effective_cut_size": mean_cut,
            "host_sampler_attempts": int(host_sampler_attempts),
            "host_built_regions": int(host_built_regions),
            "host_validate_passes": int(host_validate_passes),
            "host_validity_rate": float(host_validity_rate),
            "host_use_mask": bool(host_use_mask),
            "donor_sampler_attempts": int(donor_sampler_attempts),
            "donor_built_regions": int(donor_built_regions),
            "donor_validate_passes": int(donor_validate_passes),
            "donor_validity_rate": float(donor_validity_rate),
            "donor_use_mask": bool(donor_use_mask),
            "end_to_end_validity_rate": float(end_to_end_validity_rate),
        }

    def _empty_proposal_stats(self) -> dict[str, Any]:
        return {
            "invalid_region_rate": 0.0,
            "invalid_regions": {},
            "mean_region_ops": 0.0,
            "mean_region_inputs": 0.0,
            "mean_region_outputs": 0.0,
            "effective_cut_size": 0.0,
            "host_sampler_attempts": 0,
            "host_built_regions": 0,
            "host_validate_passes": 0,
            "host_validity_rate": 0.0,
            "host_use_mask": False,
            "donor_sampler_attempts": 0,
            "donor_built_regions": 0,
            "donor_validate_passes": 0,
            "donor_validity_rate": 0.0,
            "donor_use_mask": False,
            "end_to_end_validity_rate": 0.0,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_step(self, n_steps: int = 1) -> None:
        matched = [
            (exp, self._outcomes[exp["proposal_id"]])
            for exp in self._experience
            if exp["proposal_id"] in self._outcomes
        ]
        if len(matched) < 2:
            return

        rewards = [float(outcome.get("reward", 0.0)) for _, outcome in matched]
        target_scores = [float(outcome.get("graft_score", 1.5)) for _, outcome in matched]
        mean_reward = float(np.mean(rewards))
        mean_score = float(np.mean(target_scores))
        # Update fallback global baseline (used for unseen hosts).
        self._reward_baseline = (
            (1 - self._baseline_alpha) * self._reward_baseline
            + self._baseline_alpha * mean_reward
        )
        batch_size = min(64, len(matched))
        total_loss_val = 0.0
        last_mse_val = 0.0
        last_rl_val = 0.0
        last_advantage_mag = 0.0

        for _ in range(n_steps):
            if len(matched) > batch_size:
                indices = np.random.choice(len(matched), batch_size, replace=False)
                batch = [matched[i] for i in indices]
            else:
                batch = matched

            scorer_graphs_h: list[Data] = []
            scorer_graphs_d: list[Data] = []
            scorer_targets: list[float] = []
            for exp, outcome in batch:
                hg = self._graph_cache.get(exp["host_algo"])
                dg = self._graph_cache.get(exp["donor_algo"])
                if hg is None or dg is None:
                    continue
                scorer_graphs_h.append(hg)
                scorer_graphs_d.append(dg)
                scorer_targets.append(float(outcome.get("graft_score", 1.5)))

            mse_term: torch.Tensor | None = None
            if scorer_graphs_h:
                h_batch = Batch.from_data_list([g.clone() for g in scorer_graphs_h]).to(self.device)
                d_batch = Batch.from_data_list([g.clone() for g in scorer_graphs_d]).to(self.device)
                h_emb = self.encoder(h_batch)
                d_emb = self.encoder(d_batch)
                target = torch.tensor(scorer_targets, dtype=torch.float32, device=self.device)
                pred = self.scorer(h_emb, d_emb).squeeze(-1)
                mse_term = F.mse_loss(pred, target)

            rl_terms: list[torch.Tensor] = []
            advantages: list[float] = []
            for exp, outcome in batch:
                action_log_prob = self._boundary_action_log_prob(exp)
                if action_log_prob is None:
                    continue
                # Per-host baseline (only when we have enough samples
                # for it to be meaningful — otherwise fall back to the
                # global EMA, since a 1-sample "baseline" equals the
                # reward and gives zero advantage).
                host_id = exp.get("host_algo")
                if (
                    host_id is not None
                    and self._host_baseline_counts.get(host_id, 0)
                    >= self._host_baseline_min_n
                ):
                    host_baseline = self._host_baselines.get(
                        host_id, self._reward_baseline
                    )
                else:
                    host_baseline = self._reward_baseline
                advantage = float(outcome.get("reward", 0.0) - host_baseline)
                advantages.append(advantage)
                rl_terms.append(-action_log_prob * advantage)

            loss_terms: list[torch.Tensor] = []
            if mse_term is not None:
                loss_terms.append(mse_term)
                last_mse_val = float(mse_term.item())
            if rl_terms:
                # ``mean`` (not sum) keeps the REINFORCE term on the
                # same scale as the MSE term regardless of batch size.
                rl_mean = torch.stack(rl_terms).mean()
                loss_terms.append(self._lambda_rl * rl_mean)
                last_rl_val = float(rl_mean.item())
                last_advantage_mag = float(np.mean(np.abs(advantages))) if advantages else 0.0

            if not loss_terms:
                continue

            self.optimizer.zero_grad()
            total_loss = sum(loss_terms)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._all_params, max_norm=1.0)
            self.optimizer.step()
            total_loss_val = float(total_loss.item())

        live_pids = {exp["proposal_id"] for exp in self._experience}
        for proposal_id in [pid for pid in list(self._outcomes) if pid not in live_pids]:
            self._outcomes.pop(proposal_id, None)
        # Drop per-host baselines for hosts that are gone for good.
        live_hosts = {exp.get("host_algo") for exp in self._experience}
        live_hosts.discard(None)
        stale = [h for h in self._host_baselines if h not in live_hosts]
        for h in stale:
            self._host_baselines.pop(h, None)
            self._host_baseline_counts.pop(h, None)

        self._last_train_stats = {
            "matched_samples": len(matched),
            "train_steps": n_steps,
            "mean_reward": mean_reward,
            "mean_graft_score": mean_score,
            "baseline": self._reward_baseline,
            "n_host_baselines": len(self._host_baselines),
            "loss": total_loss_val,
            "mse_loss": last_mse_val,
            "rl_loss": last_rl_val,
            "mean_abs_advantage": last_advantage_mag,
        }
        logger.info(
            "GNN train: %d samples, %d steps, avg_reward=%.4f, avg_score=%.4f, "
            "global_baseline=%.4f, hosts_baselined=%d, loss=%.4f (mse=%.4f, rl=%.4f, |adv|=%.4f)",
            len(matched),
            n_steps,
            mean_reward,
            mean_score,
            self._reward_baseline,
            len(self._host_baselines),
            total_loss_val,
            last_mse_val,
            last_rl_val,
            last_advantage_mag,
        )

    def _boundary_action_log_prob(self, exp: dict[str, Any]) -> torch.Tensor | None:
        host_log_prob = self._region_side_log_prob(
            output_candidates=exp.get("host_output_candidates"),
            output_feats=exp.get("host_output_feats"),
            effective_outputs=exp.get("host_effective_outputs"),
            cut_candidates=exp.get("host_cut_candidates"),
            cut_feats=exp.get("host_cut_feats"),
            effective_cuts=exp.get("host_effective_cuts"),
            context_emb=exp.get("host_context_emb"),
            temperature=float(exp.get("host_temperature", self._policy_temperature())),
            exploration=self.region_exploration,
        )
        donor_log_prob = self._region_side_log_prob(
            output_candidates=exp.get("donor_output_candidates"),
            output_feats=exp.get("donor_output_feats"),
            effective_outputs=exp.get("donor_effective_outputs"),
            cut_candidates=exp.get("donor_cut_candidates"),
            cut_feats=exp.get("donor_cut_feats"),
            effective_cuts=exp.get("donor_effective_cuts"),
            context_emb=exp.get("donor_context_emb"),
            temperature=float(exp.get("donor_temperature", self._policy_temperature())),
            exploration=self.donor_exploration,
        )
        if host_log_prob is None or donor_log_prob is None:
            return None
        return host_log_prob + donor_log_prob

    def _region_side_log_prob(
        self,
        *,
        output_candidates: list[str] | None,
        output_feats: np.ndarray | None,
        effective_outputs: list[str] | None,
        cut_candidates: list[str] | None,
        cut_feats: np.ndarray | None,
        effective_cuts: list[str] | None,
        context_emb: np.ndarray | None,
        temperature: float,
        exploration: float,
    ) -> torch.Tensor | None:
        if not output_candidates or output_feats is None or context_emb is None or not effective_outputs:
            return None
        output_tensor = torch.tensor(output_feats, dtype=torch.float32, device=self.device)
        context_tensor = torch.tensor(context_emb, dtype=torch.float32, device=self.device)
        output_logits, output_stop = self.boundary_region_policy.output_logits(output_tensor, context_tensor)
        total = self._sequence_log_prob(
            candidate_ids=output_candidates,
            logits=output_logits,
            stop_logit=output_stop,
            chosen_ids=effective_outputs,
            max_selected=self.max_boundary_outputs,
            allow_empty=False,
            temperature=temperature,
            exploration=exploration,
        )

        if cut_candidates is None or cut_feats is None:
            return total
        output_summary = self._summarize_selected_value_feats(
            output_candidates,
            output_feats,
            effective_outputs,
        )
        cut_tensor = torch.tensor(cut_feats, dtype=torch.float32, device=self.device)
        output_summary_tensor = torch.tensor(output_summary, dtype=torch.float32, device=self.device)
        cut_logits, cut_stop = self.boundary_region_policy.cut_logits(
            cut_tensor,
            context_tensor,
            output_summary_tensor,
        )
        total = total + self._sequence_log_prob(
            candidate_ids=cut_candidates,
            logits=cut_logits,
            stop_logit=cut_stop,
            chosen_ids=effective_cuts or [],
            max_selected=self.max_cut_values,
            allow_empty=True,
            temperature=temperature,
            exploration=exploration,
        )
        return total

    def _sequence_log_prob(
        self,
        *,
        candidate_ids: list[str],
        logits: torch.Tensor,
        stop_logit: torch.Tensor,
        chosen_ids: list[str],
        max_selected: int,
        allow_empty: bool,
        temperature: float,
        exploration: float,
    ) -> torch.Tensor:
        available = list(range(len(candidate_ids)))
        chosen_queue = list(chosen_ids)
        total = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        stop_value = stop_logit.reshape(-1)[0]

        for step in range(max_selected):
            if not available:
                break
            current_logits = torch.stack([logits[idx] for idx in available], dim=0)
            option_ids = [candidate_ids[idx] for idx in available]
            if allow_empty or step > 0:
                current_logits = torch.cat([current_logits, stop_value.view(1)], dim=0)
                option_ids = option_ids + ["__STOP__"]
            probs = self._probs_from_tensor_logits(
                current_logits,
                temperature=temperature,
                exploration=exploration,
            )
            target = chosen_queue.pop(0) if chosen_queue else "__STOP__"
            if target not in option_ids:
                return torch.tensor(0.0, dtype=torch.float32, device=self.device)
            target_idx = option_ids.index(target)
            total = total + torch.log(probs[target_idx].clamp_min(1e-12))
            if target == "__STOP__":
                break
            selected_idx = available[target_idx]
            available = [idx for idx in available if idx != selected_idx]
        return total

    def _probs_from_tensor_logits(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        exploration: float,
    ) -> torch.Tensor:
        # Guard against NaN/Inf logits — these caused Windows access
        # violations when softmax produced NaN probs that then propagated
        # into multinomial sampling.
        if logits.numel() == 0:
            return logits
        finite_mask = torch.isfinite(logits)
        if not bool(finite_mask.all()):
            safe_logits = torch.where(
                finite_mask,
                logits,
                torch.full_like(logits, -1e9),
            )
        else:
            safe_logits = logits
        # Clamp to prevent overflow from extreme logits.
        safe_logits = safe_logits.clamp(min=-1e9, max=1e9)
        scaled = safe_logits / max(float(temperature), 1e-4)
        probs = torch.softmax(scaled, dim=0)
        # Replace any residual NaN with uniform.
        if bool(torch.isnan(probs).any()) or bool((probs.sum() <= 0).item()):
            probs = torch.full_like(probs, 1.0 / probs.shape[0])
        eps = float(np.clip(exploration, 0.0, 1.0))
        if eps > 0:
            probs = (1.0 - eps) * probs + eps / probs.shape[0]
        total = probs.sum()
        if total <= 0 or not bool(torch.isfinite(total).item()):
            return torch.full_like(probs, 1.0 / probs.shape[0])
        return probs / total

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _get_op_feats(self, ir: FunctionIR, op_ids: list[str]) -> np.ndarray:
        feats: list[list[float]] = []
        for op_id in op_ids:
            op = ir.ops.get(op_id)
            if op is None:
                continue
            one_hot = [0.0] * _N_OPCODES
            one_hot[_opcode_idx(op.opcode)] = 1.0
            callee = op.attrs.get("callee", op.attrs.get("name", ""))
            callee_feat = _hash_callee(callee) if callee else [0.0] * _CALLEE_FEATURES
            prov = op.attrs.get("_provenance") or {}
            prov_feat = _hash_provenance(
                prov.get("from_slot_id"),
                bool(prov.get("is_slot_boundary", False)),
            )
            feats.append(one_hot + callee_feat + prov_feat)
        if not feats:
            return np.zeros((1, _NODE_DIM), dtype=np.float32)
        return np.array(feats, dtype=np.float32)

    def _get_value_feats(
        self,
        ir: FunctionIR,
        value_ids: list[str],
        observable_values: set[str] | None = None,
    ) -> np.ndarray:
        observable_values = observable_values or set()
        rows: list[np.ndarray] = []
        for value_id in value_ids:
            value = ir.values.get(value_id)
            if value is None:
                continue
            if value.def_op and value.def_op in ir.ops:
                def_feat = self._get_op_feats(ir, [value.def_op])[0]
            else:
                def_feat = np.zeros((_NODE_DIM,), dtype=np.float32)
            use_ops = [use_op for use_op in value.use_ops if use_op in ir.ops]
            if use_ops:
                use_feat = self._get_op_feats(ir, use_ops).mean(axis=0)
            else:
                use_feat = np.zeros((_NODE_DIM,), dtype=np.float32)
            static = np.array([
                1.0 if value_id in ir.arg_values else 0.0,
                1.0 if value_id in ir.return_values else 0.0,
                1.0 if value_id in observable_values else 0.0,
                1.0 if value.def_op is None else 0.0,
                min(float(len(value.use_ops)) / 8.0, 1.0),
                1.0 if len(value.use_ops) > 1 else 0.0,
                *self._hash_meta(value.type_hint or "", dim=2),
                *self._hash_meta(value.name_hint or "", dim=2),
            ], dtype=np.float32)
            rows.append(np.concatenate([def_feat, use_feat, static], axis=0))
        if not rows:
            return np.zeros((0, _VALUE_FEAT_DIM), dtype=np.float32)
        return np.stack(rows, axis=0)

    def _hash_meta(self, text: str, dim: int = 2) -> list[float]:
        if not text:
            return [0.0] * dim
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xF) / 15.0 for i in range(dim)]

    def _summarize_region_feats(self, ir: FunctionIR, region: RewriteRegion) -> np.ndarray:
        op_feats = self._get_op_feats(ir, list(region.op_ids))
        if op_feats.shape[0] == 0:
            return np.zeros((_NODE_DIM,), dtype=np.float32)
        return op_feats.mean(axis=0)

    def _summarize_selected_value_feats(
        self,
        candidate_ids: list[str],
        candidate_feats_np: np.ndarray,
        selected_ids: list[str],
    ) -> np.ndarray:
        hidden_dim = self.boundary_region_policy.value_encoder[0].out_features
        if candidate_feats_np.size == 0 or not selected_ids:
            return np.zeros((hidden_dim,), dtype=np.float32)
        index_map = {value_id: idx for idx, value_id in enumerate(candidate_ids)}
        selected_rows = [
            candidate_feats_np[index_map[value_id]]
            for value_id in selected_ids
            if value_id in index_map
        ]
        if not selected_rows:
            return np.zeros((hidden_dim,), dtype=np.float32)
        with torch.no_grad():
            tensor = torch.tensor(np.stack(selected_rows, axis=0), dtype=torch.float32, device=self.device)
            encoded = self.boundary_region_policy.encode_values(tensor)
        return encoded.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_proposals": self._total_proposals,
            "total_rewards": self._total_rewards,
            "experience_buffer": len(self._experience),
            "outcome_buffer": len(self._outcomes),
            "generation": self._generation,
            "reward_baseline": self._reward_baseline,
            "warmstart_generations": self.warmstart_generations,
            "last_train": dict(self._last_train_stats),
            "last_proposals": dict(self._last_proposal_stats),
        }
