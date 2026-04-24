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
from typing import Any

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
from algorithm_ir.region.contract import infer_boundary_contract
from algorithm_ir.region.extract import extract_region_ir
from algorithm_ir.region.selector import BoundaryRegionSpec, RewriteRegion, define_rewrite_region
from algorithm_ir.region.slicer import (
    enumerate_cut_candidates,
    enumerate_observable_values,
    validate_boundary_region,
)
from evolution.pattern_matchers import _fresh_id
from evolution.pool_types import AlgorithmEntry, GraftProposal

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
    """Set of all SSA values transitively feeding any ``return`` op.

    A graft region whose ``exit_values`` are disjoint from this set is
    purely dead code: even if the donor splice succeeds at the IR level,
    the donor's outputs do not flow to the function return and DCE
    deletes them, manifesting downstream as a ``structural_fail`` verdict
    in ``train_gnn``. Pre-filtering here keeps such proposals out of the
    pipeline entirely.
    """
    slice_values: set[str] = set()
    pending: list[str] = list(ir.return_values)
    for op in ir.ops.values():
        if op.opcode == "return":
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
    ops_list = list(ir.ops.values())
    if not ops_list:
        x = torch.zeros(1, _NODE_DIM)
        return Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long))

    op_id_to_idx = {op.id: idx for idx, op in enumerate(ops_list)}
    node_feats: list[list[float]] = []
    for op in ops_list:
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

    src_list: list[int] = []
    dst_list: list[int] = []
    value_def_op: dict[str, int] = {}
    for op in ops_list:
        for value_id in op.outputs:
            value_def_op[value_id] = op_id_to_idx[op.id]

    for op in ops_list:
        op_idx = op_id_to_idx[op.id]
        for value_id in op.inputs:
            def_idx = value_def_op.get(value_id)
            if def_idx is not None and def_idx != op_idx:
                src_list.append(def_idx)
                dst_list.append(op_idx)

    for block in ir.blocks.values():
        prev_idx = None
        for op_id in block.op_ids:
            cur_idx = op_id_to_idx.get(op_id)
            if cur_idx is None:
                continue
            if prev_idx is not None:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value_embs = self.encode_values(value_feats)
        if value_embs.shape[0] == 0:
            return context.new_zeros((0,)), context.new_zeros((1,))
        ctx = context.unsqueeze(0).expand(value_embs.shape[0], -1)
        logits = self.output_head(torch.cat([value_embs, ctx], dim=-1)).squeeze(-1)
        stop = self.output_stop_head(
            torch.cat([context, value_embs.mean(dim=0)], dim=-1)
        ).reshape(-1)
        return logits, stop

    def cut_logits(
        self,
        value_feats: torch.Tensor,
        context: torch.Tensor,
        output_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value_embs = self.encode_values(value_feats)
        if value_embs.shape[0] == 0:
            return context.new_zeros((0,)), context.new_zeros((1,))
        ctx = context.unsqueeze(0).expand(value_embs.shape[0], -1)
        out_sum = output_summary.unsqueeze(0).expand(value_embs.shape[0], -1)
        logits = self.cut_head(torch.cat([value_embs, ctx, out_sum], dim=-1)).squeeze(-1)
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
        for entry in entries:
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

        pair_slice = list(selected_pairs[:max_proposals])
        if self.show_progress and tqdm is not None and pair_slice:
            pair_slice = list(tqdm(
                pair_slice,
                total=len(pair_slice),
                desc="GNN graft proposals",
                leave=False,
            ))

        invalid_regions: dict[str, int] = {}
        context_cache: dict[str, dict[str, Any] | None] = {}
        pair_infos: list[dict[str, Any]] = []
        for host_entry, donor_entry, pair_score in pair_slice:
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
        for info, out in zip(
            pair_infos,
            self._compute_output_logits_batch([
                {
                    "candidate_feats_np": info["host_ctx"]["observable_feats_np"],
                    "context_emb_np": info["donor_ctx"]["emb"].detach().cpu().numpy(),
                }
                for info in pair_infos
            ]),
        ):
            info["host_output_logits"] = out

        region_temperature = self._policy_temperature()
        donor_temperature = self._policy_temperature()

        host_cut_infos: list[dict[str, Any]] = []
        for info in pair_infos:
            host_outputs = self._sample_value_sequence(
                info["host_ctx"]["observable_values"],
                info["host_output_logits"][0],
                info["host_output_logits"][1],
                max_selected=self.max_boundary_outputs,
                allow_empty=False,
                temperature=region_temperature,
                exploration=self.region_exploration,
            )
            if not host_outputs:
                invalid_regions["host_no_output"] = invalid_regions.get("host_no_output", 0) + 1
                continue
            info["host_selected_outputs"] = host_outputs
            info["host_cut_ctx"] = self._get_cut_context(info["host_ctx"], host_outputs)
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
        for info in host_cut_infos:
            host_cuts = self._sample_value_sequence(
                info["host_cut_ctx"]["candidate_ids"],
                info["host_cut_logits"][0],
                info["host_cut_logits"][1],
                max_selected=self.max_cut_values,
                allow_empty=True,
                temperature=region_temperature,
                exploration=self.region_exploration,
            )
            info["host_selected_cuts"] = host_cuts
            host_build = self._build_boundary_region(
                info["host_ctx"]["ir"],
                output_values=info["host_selected_outputs"],
                cut_values=host_cuts,
            )
            if host_build is None:
                invalid_regions["host_region_build_failed"] = invalid_regions.get("host_region_build_failed", 0) + 1
                continue
            host_region, host_validity = host_build
            if not host_validity.is_valid:
                key = f"host_{host_validity.reason}"
                invalid_regions[key] = invalid_regions.get(key, 0) + 1
                continue
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
            host_region_metrics.append((host_validity.n_ops, host_validity.n_inputs, host_validity.n_outputs))
            effective_cut_sizes.append(len(info["host_effective_cuts"]))
            host_valid_infos.append(info)

        # Donor outputs
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

        donor_cut_infos: list[dict[str, Any]] = []
        for info in host_valid_infos:
            donor_outputs = self._sample_value_sequence(
                info["donor_ctx"]["observable_values"],
                info["donor_output_logits"][0],
                info["donor_output_logits"][1],
                max_selected=self.max_boundary_outputs,
                allow_empty=False,
                temperature=donor_temperature,
                exploration=self.donor_exploration,
            )
            if not donor_outputs:
                invalid_regions["donor_no_output"] = invalid_regions.get("donor_no_output", 0) + 1
                continue
            info["donor_selected_outputs"] = donor_outputs
            info["donor_cut_ctx"] = self._get_cut_context(info["donor_ctx"], donor_outputs)
            donor_cut_infos.append(info)

        # Donor cuts
        for info, out in zip(
            donor_cut_infos,
            self._compute_cut_logits_batch([
                {
                    "candidate_feats_np": info["donor_cut_ctx"]["candidate_feats_np"],
                    "context_emb_np": info["host_ctx"]["emb"].detach().cpu().numpy(),
                    "output_summary_np": self._summarize_selected_value_feats(
                        info["donor_ctx"]["observable_values"],
                        info["donor_ctx"]["observable_feats_np"],
                        info["donor_selected_outputs"],
                    ),
                }
                for info in donor_cut_infos
            ]),
        ):
            info["donor_cut_logits"] = out

        proposals: list[GraftProposal] = []
        for info in donor_cut_infos:
            donor_cuts = self._sample_value_sequence(
                info["donor_cut_ctx"]["candidate_ids"],
                info["donor_cut_logits"][0],
                info["donor_cut_logits"][1],
                max_selected=self.max_cut_values,
                allow_empty=True,
                temperature=donor_temperature,
                exploration=self.donor_exploration,
            )
            info["donor_selected_cuts"] = donor_cuts
            donor_build = self._build_boundary_region(
                info["donor_ctx"]["ir"],
                output_values=info["donor_selected_outputs"],
                cut_values=donor_cuts,
            )
            if donor_build is None:
                invalid_regions["donor_region_build_failed"] = invalid_regions.get("donor_region_build_failed", 0) + 1
                continue
            donor_region, donor_validity = donor_build
            if not donor_validity.is_valid:
                key = f"donor_{donor_validity.reason}"
                invalid_regions[key] = invalid_regions.get(key, 0) + 1
                continue
            try:
                donor_trim = extract_region_ir(info["donor_ctx"]["ir"], donor_region)
            except Exception:
                invalid_regions["donor_extract_failed"] = invalid_regions.get("donor_extract_failed", 0) + 1
                continue
            info["donor_region"] = donor_region
            info["donor_validity"] = donor_validity
            info["donor_effective_outputs"] = list(donor_region.provenance.get("effective_output_values", info["donor_selected_outputs"]))
            info["donor_effective_cuts"] = list(donor_region.provenance.get("effective_cut_values", donor_cuts))
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
                f"BCIR graft: host_out={len(info['host_effective_outputs'])} "
                f"host_cut={len(info['host_effective_cuts'])} "
                f"donor_out={len(info['donor_effective_outputs'])} "
                f"donor_cut={len(info['donor_effective_cuts'])} "
                f"pred_score={float(info['pair_score']):.3f}"
            ),
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

    def _build_boundary_region(
        self,
        ir: FunctionIR,
        *,
        output_values: list[str],
        cut_values: list[str],
    ) -> tuple[RewriteRegion, Any] | None:
        # Greedy connectivity-aware cut pruning.  Even if each cut
        # individually preserves connectivity (enforced by
        # ``enumerate_cut_candidates(require_connected=True)``), a
        # *combination* of cuts can sever the slice into multiple
        # islands.  When this happens we iteratively drop the cut whose
        # removal restores connectivity, until either the region is
        # connected or no cuts remain.  This is a generic structural
        # repair, not a graft-pattern prior.
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
    ) -> list[str]:
        if not candidate_ids or logits.numel() == 0:
            return []
        available = list(range(len(candidate_ids)))
        selected: list[str] = []
        stop_value = float(stop_logit.reshape(-1)[0].item())
        for step in range(max_selected):
            if not available:
                break
            option_logits = [float(logits[idx].item()) for idx in available]
            option_ids = [candidate_ids[idx] for idx in available]
            if allow_empty or step > 0:
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
            remove_idx = available[chosen]
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
    ) -> dict[str, Any]:
        if attempted <= 0:
            return self._empty_proposal_stats()
        mean_ops = float(np.mean([m[0] for m in host_region_metrics])) if host_region_metrics else 0.0
        mean_inputs = float(np.mean([m[1] for m in host_region_metrics])) if host_region_metrics else 0.0
        mean_outputs = float(np.mean([m[2] for m in host_region_metrics])) if host_region_metrics else 0.0
        mean_cut = float(np.mean(effective_cut_sizes)) if effective_cut_sizes else 0.0
        return {
            "invalid_region_rate": float(sum(invalid_regions.values()) / attempted),
            "invalid_regions": dict(invalid_regions),
            "mean_region_ops": mean_ops,
            "mean_region_inputs": mean_inputs,
            "mean_region_outputs": mean_outputs,
            "effective_cut_size": mean_cut,
        }

    def _empty_proposal_stats(self) -> dict[str, Any]:
        return {
            "invalid_region_rate": 0.0,
            "invalid_regions": {},
            "mean_region_ops": 0.0,
            "mean_region_inputs": 0.0,
            "mean_region_outputs": 0.0,
            "effective_cut_size": 0.0,
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
