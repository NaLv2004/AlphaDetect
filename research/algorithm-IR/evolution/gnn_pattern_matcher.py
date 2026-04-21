"""GNN-based Pattern Matcher for structural grafting proposals.

Encodes each algorithm's IR as a graph, embeds it using a GNN backbone,
scores host-donor pairs for graft compatibility, and proposes graft
regions.  Trains online via REINFORCE using graft-outcome rewards.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool

import uuid as _uuid

from algorithm_ir.ir.model import FunctionIR, Op, Value, Block
from algorithm_ir.region.selector import RewriteRegion
from evolution.pool_types import (
    AlgorithmEntry,
    GraftProposal,
    PatternMatcherFn,
)
from evolution.pattern_matchers import _build_region_from_ops, _fresh_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Opcode vocabulary (fixed feature set for IR ops)
# ---------------------------------------------------------------------------

_OPCODE_VOCAB: dict[str, int] = {}
_OPCODE_LIST = [
    "const", "binary", "unary", "call", "compare", "branch", "jump",
    "return", "subscript", "store", "load", "algslot", "phi",
    "attr", "build_list", "build_tuple", "build_dict", "augassign",
    "<unk>",
]
for _i, _op in enumerate(_OPCODE_LIST):
    _OPCODE_VOCAB[_op] = _i

_N_OPCODES = len(_OPCODE_LIST)
_CALLEE_FEATURES = 8  # hashed callee name features
_NODE_DIM = _N_OPCODES + _CALLEE_FEATURES  # total node feature dim
_TERMINATOR_OPCODES = frozenset({"branch", "jump", "return"})


def _opcode_idx(opcode: str) -> int:
    return _OPCODE_VOCAB.get(opcode, _OPCODE_VOCAB["<unk>"])


def _hash_callee(name: str, dim: int = _CALLEE_FEATURES) -> list[float]:
    """Deterministic hash of callee name to a fixed-dim feature vector."""
    h = hash(name) & 0xFFFFFFFF
    feats = []
    for i in range(dim):
        feats.append(((h >> (i * 4)) & 0xF) / 15.0)
    return feats


# ---------------------------------------------------------------------------
# IR → PyG Data conversion
# ---------------------------------------------------------------------------

def ir_to_graph(ir: FunctionIR) -> Data:
    """Convert a FunctionIR into a PyTorch Geometric Data object.

    Nodes = ops.  Edges = data-flow (value def→use) + control-flow.
    Node features = one-hot opcode + hashed callee.
    """
    ops_list = list(ir.ops.values())
    if not ops_list:
        # Empty IR → single dummy node
        x = torch.zeros(1, _NODE_DIM)
        return Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long))

    op_id_to_idx: dict[str, int] = {}
    for idx, op in enumerate(ops_list):
        op_id_to_idx[op.id] = idx

    # Node features
    node_feats = []
    for op in ops_list:
        one_hot = [0.0] * _N_OPCODES
        one_hot[_opcode_idx(op.opcode)] = 1.0
        callee = op.attrs.get("callee", op.attrs.get("name", ""))
        callee_feat = _hash_callee(callee) if callee else [0.0] * _CALLEE_FEATURES
        node_feats.append(one_hot + callee_feat)

    x = torch.tensor(node_feats, dtype=torch.float32)

    # Edges: data-flow (value producers → consumers)
    src_list: list[int] = []
    dst_list: list[int] = []

    # Build value → defining op mapping
    val_def_op: dict[str, int] = {}
    for op in ops_list:
        for vid in op.outputs:
            val_def_op[vid] = op_id_to_idx[op.id]

    for op in ops_list:
        op_idx = op_id_to_idx[op.id]
        for vid in op.inputs:
            def_idx = val_def_op.get(vid)
            if def_idx is not None and def_idx != op_idx:
                src_list.append(def_idx)
                dst_list.append(op_idx)

    # Control-flow edges: sequential ops within same block
    for block in ir.blocks.values():
        prev_idx = None
        for oid in block.op_ids:
            cur_idx = op_id_to_idx.get(oid)
            if cur_idx is not None:
                if prev_idx is not None:
                    src_list.append(prev_idx)
                    dst_list.append(cur_idx)
                prev_idx = cur_idx

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# ---------------------------------------------------------------------------
# GNN backbone: GAT-based encoder
# ---------------------------------------------------------------------------

class IRGraphEncoder(nn.Module):
    """GAT encoder: IR graph → fixed-dim embedding."""

    def __init__(self, node_dim: int = _NODE_DIM, hidden: int = 64, out_dim: int = 32, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden, heads=heads, concat=False)
        self.conv2 = GATConv(hidden, hidden, heads=heads, concat=False)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Returns [batch_size, out_dim] graph embeddings."""
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long)

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Graft scorer: predicts graft quality from host + donor embeddings
# ---------------------------------------------------------------------------

class GraftScorer(nn.Module):
    """MLP scorer: (host_emb, donor_emb) → scalar graft quality logit."""

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
        """Returns [N, 1] predicted graft scores (lower is better)."""
        combined = torch.cat([host_emb, donor_emb], dim=-1)
        return self.net(combined)


# ---------------------------------------------------------------------------
# Region proposer: node-level scores for region selection
# ---------------------------------------------------------------------------

class RegionProposer(nn.Module):
    """Host-region policy that samples a contiguous start + length window."""

    LENGTH_BUCKETS = [1, 2, 3, 4, 5, 6, 8]

    def __init__(self, node_dim: int = _NODE_DIM, hidden: int = 64, context_dim: int = 32):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden, heads=4, concat=False)
        self.conv2 = GATConv(hidden, hidden, heads=4, concat=False)
        self.start_head = nn.Sequential(
            nn.Linear(hidden + context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.length_head = nn.Sequential(
            nn.Linear(hidden + context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(self.LENGTH_BUCKETS)),
        )

    def forward(
        self,
        data: Data,
        donor_emb: torch.Tensor,
        candidate_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ([N_candidates] start logits, [B] length-bucket logits)."""
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))

        n_nodes = x.size(0)
        donor_exp = donor_emb.expand(n_nodes, -1)
        ctx = torch.cat([x, donor_exp], dim=-1)
        if candidate_indices is not None and candidate_indices.numel() > 0:
            ctx = ctx[candidate_indices]

        if ctx.shape[0] == 0:
            start_zeros = torch.zeros(1, device=data.x.device)
            length_zeros = torch.zeros(
                len(self.LENGTH_BUCKETS), device=data.x.device,
            )
            return start_zeros, length_zeros

        start_logits = self.start_head(ctx).squeeze(-1)
        pooled = ctx.mean(0)
        length_logits = self.length_head(pooled)
        return start_logits, length_logits


# ---------------------------------------------------------------------------
# Donor region selector: cross-attention GNN
# ---------------------------------------------------------------------------

class DonorRegionSelectorGNN(nn.Module):
    """Cross-attention GNN that selects the best start position AND length
    in the donor's entry block for grafting.

    Architecture:
      1. Project host region op features → query space (avg-pooled)
      2. Project donor entry block op features → key/value space
      3. Start scoring: for each donor op, score = MLP([donor_proj, host_query])
      4. Length prediction: MLP(host_query) → logits over discrete length buckets
      5. Output: [N] start-position logits + [B] length-bucket logits

    Length buckets: [2, 3, 4, 5, 6, 8, 10, 12, 15]  (9 discrete choices)

    Trained via REINFORCE:
      loss = -(log P(start) + log P(length)) * advantage
    """

    LENGTH_BUCKETS = [2, 3, 4, 5, 6, 8, 10, 12, 15]

    def __init__(self, node_dim: int = _NODE_DIM, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.host_proj = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.donor_proj = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        # Concat(donor_op_emb, host_query) → start score
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Length head: from host_query → length bucket logits
        self.length_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(self.LENGTH_BUCKETS)),
        )

    def forward(
        self,
        host_region_feats: torch.Tensor,  # [H, node_dim]
        donor_entry_feats: torch.Tensor,  # [N, node_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ([N] start logits, [B] length-bucket logits)."""
        n_donor = donor_entry_feats.shape[0]
        n_buckets = len(self.LENGTH_BUCKETS)

        if host_region_feats.shape[0] == 0:
            return (torch.zeros(n_donor, device=donor_entry_feats.device),
                    torch.zeros(n_buckets, device=donor_entry_feats.device))
        if n_donor == 0:
            return (torch.zeros(1, device=host_region_feats.device),
                    torch.zeros(n_buckets, device=host_region_feats.device))

        h = self.host_proj(host_region_feats)   # [H, hidden]
        d = self.donor_proj(donor_entry_feats)  # [N, hidden]

        # Average-pool host features → single query vector
        h_query = h.mean(0, keepdim=True)       # [1, hidden]
        h_exp = h_query.expand(d.shape[0], -1)  # [N, hidden]

        combined = torch.cat([d, h_exp], dim=-1)  # [N, hidden*2]
        start_logits = self.score_head(combined).squeeze(-1)  # [N]

        length_logits = self.length_head(h_query.squeeze(0))  # [B]

        return start_logits, length_logits


# ---------------------------------------------------------------------------
# Utility: build a trimmed FunctionIR from selected donor ops
# ---------------------------------------------------------------------------

def build_trimmed_donor_ir(
    donor_ir: FunctionIR,
    selected_op_ids: list[str],
) -> FunctionIR | None:
    """Build a valid single-block FunctionIR from a contiguous sub-region of
    donor entry block ops.

    The trimmed IR:
    - Contains exactly *selected_op_ids* + a synthetic return op
    - Values consumed but not defined in the region become function args
    - Returns the first "exit value" (defined in region but used outside),
      falling back to the last op's first output.

    This lets ``graft_general`` inline only the selected sub-computation
    instead of the entire donor algorithm.

    Returns ``None`` if the region produces no usable outputs.
    """
    if not selected_op_ids:
        return None

    # --- Dependency expansion: include defining ops for intermediate ---
    # values that are not donor function args.  Without this, the trimmed
    # donor has dangling intermediate args (name_hint="binary" etc.) that
    # get mis-matched to host values.
    donor_arg_set = set(donor_ir.arg_values)
    expanded_ops = list(selected_op_ids)
    expanded_set = set(expanded_ops)
    MAX_EXPANSION = len(selected_op_ids) * 3  # don't blow up

    for _round in range(3):  # at most 3 rounds of backward expansion
        needs: set[str] = set()
        defined_in_selection: set[str] = set()
        for oid in expanded_ops:
            op = donor_ir.ops.get(oid)
            if op is None:
                continue
            defined_in_selection.update(op.outputs)
            needs.update(op.inputs)
        unresolved = needs - defined_in_selection - donor_arg_set
        if not unresolved:
            break  # all dependencies satisfied
        added = False
        for vid in sorted(unresolved):
            if len(expanded_ops) >= MAX_EXPANSION:
                break
            val = donor_ir.values.get(vid)
            if val and val.def_op and val.def_op not in expanded_set:
                # Insert defining op at the beginning (before selected ops)
                expanded_ops.insert(0, val.def_op)
                expanded_set.add(val.def_op)
                added = True
        if not added:
            break

    selected_op_ids = expanded_ops
    selected_set = set(selected_op_ids)

    # Classify values
    defined_vids: set[str] = set()
    used_vids: set[str] = set()
    for oid in selected_op_ids:
        op = donor_ir.ops.get(oid)
        if op is None:
            continue
        defined_vids.update(op.outputs)
        used_vids.update(op.inputs)

    # Entry vals: consumed by region but not defined inside it
    entry_vids: list[str] = sorted(used_vids - defined_vids)

    # Exit vals: defined in region AND used by ops outside the region
    exit_vids: list[str] = []
    for vid in sorted(defined_vids):
        val = donor_ir.values.get(vid)
        if val is None:
            continue
        for use_oid in val.use_ops:
            if use_oid not in selected_set:
                exit_vids.append(vid)
                break

    # Fallback: return the last defined value
    if not exit_vids and selected_op_ids:
        for oid in reversed(selected_op_ids):
            last_op = donor_ir.ops.get(oid)
            if last_op and last_op.outputs:
                exit_vids = [last_op.outputs[0]]
                break

    if not exit_vids:
        return None  # Can't construct a meaningful return

    block_id = "trim_entry"
    new_ops: dict[str, Op] = {}

    for oid in selected_op_ids:
        op = donor_ir.ops.get(oid)
        if op is None:
            continue
        new_ops[oid] = Op(
            id=op.id,
            opcode=op.opcode,
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            block_id=block_id,
            source_span=op.source_span,
            attrs=dict(op.attrs),
        )

    # Synthetic return op
    ret_oid = f"ret_{_uuid.uuid4().hex[:6]}"
    new_ops[ret_oid] = Op(
        id=ret_oid,
        opcode="return",
        inputs=exit_vids[:1],
        outputs=[],
        block_id=block_id,
        attrs={},
    )

    # Build value table
    all_vids = set(entry_vids) | defined_vids
    new_vals: dict[str, Value] = {}
    for vid in all_vids:
        orig = donor_ir.values.get(vid)
        if orig is None:
            new_vals[vid] = Value(id=vid, name_hint=vid, type_hint="object")
        else:
            new_vals[vid] = Value(
                id=vid,
                name_hint=orig.name_hint,
                type_hint=orig.type_hint,
                source_span=orig.source_span,
                def_op=orig.def_op if orig.def_op in new_ops else None,
                use_ops=[u for u in orig.use_ops if u in new_ops],
                attrs=dict(orig.attrs),
            )

    # Function args: entry vals that exist in our value table
    func_args = [v for v in entry_vids if v in new_vals]

    block = Block(
        id=block_id,
        op_ids=list(selected_op_ids) + [ret_oid],
    )

    return FunctionIR(
        id=donor_ir.id + "_trim",
        name=donor_ir.name + "_trim",
        arg_values=func_args,
        return_values=exit_vids[:1],
        values=new_vals,
        ops=new_ops,
        blocks={block_id: block},
        entry_block=block_id,
        attrs={},
    )


# ---------------------------------------------------------------------------
# GNN Pattern Matcher (main class)
# ---------------------------------------------------------------------------

class GNNPatternMatcher:
    """GNN-based structural graft proposal generator.

    Architecture:
      1. IRGraphEncoder: encodes each algorithm IR into an embedding
      2. GraftScorer: scores (host, donor) pairs for compatibility
      3. RegionProposer: selects which ops in the host to replace

    Training:
      - Online REINFORCE: after each generation, receives graft outcomes
        (reward = fitness improvement) and updates the model.
      - Experience buffer stores (state, action, reward) tuples.

    Usage:
      matcher = GNNPatternMatcher()
      # As a PatternMatcherFn:
      proposals = matcher(entries, generation)
      # After evaluating grafts:
      matcher.record_outcome(proposal_id, reward)
      matcher.train_step()
    """

    def __init__(
        self,
        max_proposals_per_gen: int = 4,
        top_k_pairs: int = 8,
        min_region_size: int = 2,
        max_region_size: int = 10,
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
        device: str | None = None,
    ):
        self.max_proposals_per_gen = max_proposals_per_gen
        self.top_k_pairs = top_k_pairs
        self.min_region_size = min_region_size
        self.max_region_size = max_region_size
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

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Models
        self.encoder = IRGraphEncoder().to(self.device)
        self.scorer = GraftScorer().to(self.device)
        self.region_proposer = RegionProposer().to(self.device)
        self.donor_region_selector = DonorRegionSelectorGNN().to(self.device)

        # Convenience list of all trainable params for grad-clipping
        self._all_params = (
            list(self.encoder.parameters())
            + list(self.scorer.parameters())
            + list(self.region_proposer.parameters())
            + list(self.donor_region_selector.parameters())
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self._all_params, lr=lr)

        # Exponential moving-average baseline for REINFORCE variance reduction
        self._reward_baseline: float = 0.0
        self._baseline_alpha: float = 0.1  # EMA coefficient

        # Experience buffer: list of (proposal_id, log_prob, baseline_score)
        self._experience: list[dict[str, Any]] = []
        # Outcome storage: proposal_id → reward
        self._outcomes: dict[str, dict[str, Any]] = {}
        # Graph cache: algo_id → Data
        self._graph_cache: dict[str, Data] = {}
        # Embedding cache (per generation)
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
        """Generate graft proposals using the GNN model."""
        self._generation = generation

        if len(entries) < 2:
            return []

        import time as _time
        _t0 = _time.perf_counter()

        # Encode all IRs first so that _train_step has fresh caches
        # (avoids training on stale/evicted algos from last generation).
        self._encode_entries(entries)
        _t_enc = _time.perf_counter()

        # Train if we have enough experience
        if generation > 0 and generation % self.train_interval == 0:
            self._train_step(n_steps=self.train_steps)
        _t_train = _time.perf_counter()

        # Score all (host, donor) pairs
        pair_scores = self._score_pairs(entries)
        _t_score = _time.perf_counter()

        selected_pairs = self._select_pair_candidates(pair_scores)

        proposals: list[GraftProposal] = []
        max_proposals = (
            len(selected_pairs)
            if self.is_warmstart_generation(generation)
            else self.max_proposals_per_gen
        )
        for host_entry, donor_entry, predicted_score in selected_pairs:
            if len(proposals) >= max_proposals:
                break
            proposal = self._propose_graft(
                host_entry, donor_entry, predicted_score,
            )
            if proposal is not None:
                proposals.append(proposal)
        _t_prop = _time.perf_counter()
        self._last_proposal_stats = {
            "warmstart": self.is_warmstart_generation(generation),
            "pair_candidates_scored": len(pair_scores),
            "pair_candidates_selected": len(selected_pairs),
            "proposals_built": len(proposals),
        }

        logger.info(
            "GNN propose: encode=%.2fs train=%.2fs score=%.2fs propose=%.2fs "
            "→ %d proposals from %d entries",
            _t_enc - _t0, _t_train - _t_enc, _t_score - _t_train,
            _t_prop - _t_score, len(proposals), len(entries),
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
        """Record the outcome of a graft proposal for training."""
        reward = max(reward, -10.0)
        if graft_score is None or not np.isfinite(graft_score):
            graft_score = 1.5
        self._outcomes[proposal_id] = {
            "reward": reward,
            "graft_score": float(graft_score),
            "host_score": float(host_score) if host_score is not None else None,
            "is_valid": bool(is_valid) if is_valid is not None else False,
        }
        self._total_rewards += reward

    def is_warmstart_generation(self, generation: int | None = None) -> bool:
        """Whether this generation should enumerate every host/donor pair."""
        gen = self._generation if generation is None else generation
        return self.warmstart_generations > 0 and gen <= self.warmstart_generations

    def _select_pair_candidates(
        self,
        pair_scores: list[tuple[AlgorithmEntry, AlgorithmEntry, float]],
    ) -> list[tuple[AlgorithmEntry, AlgorithmEntry, float]]:
        """Warm-start with all pairs, then sample stochastically from all pairs."""
        if not pair_scores:
            return []
        if self.is_warmstart_generation():
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
        chosen = np.random.choice(
            len(pair_scores), size=n_take, replace=False, p=probs,
        )
        return [pair_scores[i] for i in chosen]

    def _make_sampling_probs(
        self,
        values: np.ndarray,
        *,
        temperature: float,
        exploration: float,
        prefer_lower: bool,
    ) -> np.ndarray:
        """Softmax-with-uniform-mix sampling probabilities."""
        if values.size == 0:
            return values
        logits = -values if prefer_lower else values.copy()
        logits = logits - np.max(logits)
        temp = max(float(temperature), 1e-4)
        probs = np.exp(logits / temp)
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            probs = np.full(values.shape[0], 1.0 / values.shape[0], dtype=np.float64)
        else:
            probs = probs / total
        eps = float(np.clip(exploration, 0.0, 1.0))
        if eps > 0:
            probs = (1.0 - eps) * probs + eps / probs.shape[0]
        probs = probs / probs.sum()
        return probs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _encode_entries(self, entries: list[AlgorithmEntry]) -> None:
        """Encode all entries to graph embeddings.

        Rebuilds graphs from the *current* IR every generation to avoid
        stale op-ID references after grafting.  Also prunes cache entries
        for algo_ids no longer in the population.
        """
        self._emb_cache.clear()
        graphs: list[Data] = []
        keys: list[str] = []

        live_ids: set[str] = set()
        for entry in entries:
            live_ids.add(entry.algo_id)
            # Always rebuild graph from current IR — grafted genomes have
            # new op IDs that invalidate old cached Data objects.
            self._graph_cache[entry.algo_id] = ir_to_graph(entry.ir)
            graphs.append(self._graph_cache[entry.algo_id])
            keys.append(entry.algo_id)

        # Prune stale cache entries to bound memory growth
        stale = [k for k in self._graph_cache if k not in live_ids]
        for k in stale:
            del self._graph_cache[k]

        if not graphs:
            return

        # Batch encode
        with torch.no_grad():
            batch = Batch.from_data_list(
                [g.clone() for g in graphs]
            ).to(self.device)
            embeddings = self.encoder(batch)  # [N, emb_dim]

        for i, key in enumerate(keys):
            self._emb_cache[key] = embeddings[i]

    def _score_pairs(
        self, entries: list[AlgorithmEntry],
    ) -> list[tuple[AlgorithmEntry, AlgorithmEntry, float]]:
        """Score all distinct (host, donor) pairs — batched for speed."""
        # Collect entries that have embeddings
        valid = [(e, self._emb_cache[e.algo_id])
                 for e in entries if e.algo_id in self._emb_cache]
        if len(valid) < 2:
            return []

        results: list[tuple[AlgorithmEntry, AlgorithmEntry, float]] = []
        n = len(valid)
        # Build all (host, donor) pair indices
        h_indices, d_indices = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    h_indices.append(i)
                    d_indices.append(j)

        if not h_indices:
            return []

        # Batch all pairs through the scorer in one forward pass
        embs = torch.stack([emb for _, emb in valid])  # [n, emb_dim]
        h_embs = embs[h_indices]  # [n_pairs, emb_dim]
        d_embs = embs[d_indices]  # [n_pairs, emb_dim]

        with torch.no_grad():
            scores = self.scorer(h_embs, d_embs).squeeze(-1).cpu().numpy()

        for k, (hi, di) in enumerate(zip(h_indices, d_indices)):
            results.append((valid[hi][0], valid[di][0], float(scores[k])))

        return results

    def _propose_graft(
        self,
        host_entry: AlgorithmEntry,
        donor_entry: AlgorithmEntry,
        pair_score: float,
    ) -> GraftProposal | None:
        """Select host region + donor sub-region using GNNs, build trimmed donor IR."""
        host_ir = host_entry.ir
        donor_ir = donor_entry.ir

        # ------------------------------------------------------------------
        # 1. Identify host entry-block non-terminator ops
        # ------------------------------------------------------------------
        host_eblock = host_ir.blocks.get(host_ir.entry_block)
        if host_eblock is None:
            return None
        host_entry_ops = [
            oid for oid in host_eblock.op_ids
            if host_ir.ops.get(oid) and host_ir.ops[oid].opcode not in _TERMINATOR_OPCODES
        ]
        if len(host_entry_ops) < 2:
            return None

        # ------------------------------------------------------------------
        # 2. Score host ops with RegionProposer → select contiguous window
        # ------------------------------------------------------------------
        host_graph = self._graph_cache.get(host_entry.algo_id)
        donor_emb = self._emb_cache.get(donor_entry.algo_id)
        if host_graph is None or donor_emb is None:
            return None

        # Build consistent op-index mapping from the *same* graph that was
        # used to build the PyG Data, so node indices are guaranteed aligned.
        # ir_to_graph() iterates list(ir.ops.values()), so we mirror that.
        all_ops_list = list(host_ir.ops.values())
        op_to_idx = {op.id: i for i, op in enumerate(all_ops_list)}

        entry_op_node_indices = [
            op_to_idx[oid] for oid in host_entry_ops
            if oid in op_to_idx
        ]
        if len(entry_op_node_indices) < 2:
            return None

        region_temp = max(0.3, 2.0 - 0.04 * self._generation)
        with torch.no_grad():
            graph_data = host_graph.clone().to(self.device)
            idx_tensor = torch.tensor(
                entry_op_node_indices, dtype=torch.long, device=self.device,
            )
            start_logits_h, length_logits_h = self.region_proposer(
                graph_data, donor_emb.unsqueeze(0), idx_tensor,
            )
            start_probs_h = torch.softmax(start_logits_h / region_temp, dim=0)
            start_probs_h = (
                (1.0 - self.region_exploration) * start_probs_h
                + self.region_exploration / max(start_probs_h.shape[0], 1)
            )
            if torch.any(torch.isnan(start_probs_h)) or torch.any(torch.isinf(start_probs_h)):
                h_start = 0
            else:
                h_start = int(torch.multinomial(start_probs_h, 1).item())

            length_probs_h = torch.softmax(length_logits_h / region_temp, dim=0)
            length_probs_h = (
                (1.0 - self.region_exploration) * length_probs_h
                + self.region_exploration / max(length_probs_h.shape[0], 1)
            )
            if torch.any(torch.isnan(length_probs_h)) or torch.any(torch.isinf(length_probs_h)):
                host_length_bucket_idx = 0
            else:
                host_length_bucket_idx = int(torch.multinomial(length_probs_h, 1).item())

        desired_host_len = RegionProposer.LENGTH_BUCKETS[host_length_bucket_idx]
        host_region_len = max(self.min_region_size, desired_host_len)
        host_region_len = min(host_region_len, self.max_region_size)
        h_end = min(len(host_entry_ops), h_start + host_region_len)
        if h_end - h_start < self.min_region_size:
            h_start = max(0, len(host_entry_ops) - self.min_region_size)
            h_end = min(len(host_entry_ops), h_start + self.min_region_size)
        host_region_ops = host_entry_ops[h_start:h_end]
        if not host_region_ops:
            return None

        # ------------------------------------------------------------------
        # 3. Identify donor entry-block non-terminator ops
        # ------------------------------------------------------------------
        donor_eblock = donor_ir.blocks.get(donor_ir.entry_block)
        if donor_eblock is None:
            return None
        donor_entry_ops = [
            oid for oid in donor_eblock.op_ids
            if donor_ir.ops.get(oid) and donor_ir.ops[oid].opcode not in _TERMINATOR_OPCODES
        ]
        if not donor_entry_ops:
            return None

        # ------------------------------------------------------------------
        # 4. DonorRegionSelectorGNN: cross-attend host→donor, sample start + length
        # ------------------------------------------------------------------
        host_region_feats_np = self._get_op_feats(host_ir, host_region_ops)
        donor_entry_feats_np = self._get_op_feats(donor_ir, donor_entry_ops)

        h_feats = torch.tensor(host_region_feats_np, dtype=torch.float32, device=self.device)
        d_feats = torch.tensor(donor_entry_feats_np, dtype=torch.float32, device=self.device)

        # Temperature annealing: high early (exploration) → low later (exploitation)
        temperature = max(0.3, 2.0 - 0.04 * self._generation)

        with torch.no_grad():
            start_logits, length_logits = self.donor_region_selector(h_feats, d_feats)
            if start_logits.shape[0] == 0:
                return None

            # Sample start position
            start_probs = F.softmax(start_logits / temperature, dim=0)
            start_probs = (
                (1.0 - self.donor_exploration) * start_probs
                + self.donor_exploration / max(start_probs.shape[0], 1)
            )
            if torch.any(torch.isnan(start_probs)) or torch.any(torch.isinf(start_probs)):
                start_idx = 0
            else:
                start_idx = int(torch.multinomial(start_probs, 1).item())

            # Sample length bucket
            length_probs = F.softmax(length_logits / temperature, dim=0)
            length_probs = (
                (1.0 - self.donor_exploration) * length_probs
                + self.donor_exploration / max(length_probs.shape[0], 1)
            )
            if torch.any(torch.isnan(length_probs)) or torch.any(torch.isinf(length_probs)):
                length_bucket_idx = 0
            else:
                length_bucket_idx = int(torch.multinomial(length_probs, 1).item())
            desired_len = DonorRegionSelectorGNN.LENGTH_BUCKETS[length_bucket_idx]

        # Clamp to available donor ops
        max_possible = len(donor_entry_ops) - start_idx
        donor_region_len = min(desired_len, max_possible)
        donor_region_len = max(donor_region_len, self.min_region_size)
        if start_idx + donor_region_len > len(donor_entry_ops):
            start_idx = max(0, len(donor_entry_ops) - donor_region_len)
        donor_region_len = min(donor_region_len, len(donor_entry_ops) - start_idx)
        if donor_region_len <= 0:
            return None

        # ------------------------------------------------------------------
        # 5. Build trimmed donor IR from selected sub-region
        # ------------------------------------------------------------------
        selected_donor_ops = donor_entry_ops[start_idx: start_idx + donor_region_len]
        trimmed_donor = build_trimmed_donor_ir(donor_ir, selected_donor_ops)
        if trimmed_donor is None:
            return None

        # ------------------------------------------------------------------
        # 6. Build host RewriteRegion + store experience for RL
        # ------------------------------------------------------------------
        region = _build_region_from_ops(host_ir, host_region_ops)
        proposal_id = _fresh_id("gnn_graft")

        self._experience.append({
            "proposal_id": proposal_id,
            "host_region_feats": host_region_feats_np,    # [H, NODE_DIM] numpy
            "donor_entry_feats": donor_entry_feats_np,    # [N, NODE_DIM] numpy
            "host_start_idx": int(h_start),
            "host_length_bucket_idx": int(host_length_bucket_idx),
            "host_region_len": int(len(host_region_ops)),
            "donor_start_idx": int(start_idx),
            "donor_length_bucket_idx": int(length_bucket_idx),
            "donor_region_len": int(donor_region_len),
            "host_algo": host_entry.algo_id,
            "donor_algo": donor_entry.algo_id,
            "predicted_graft_score": float(pair_score),
            "generation": self._generation,
            "entry_op_node_indices": entry_op_node_indices,
        })
        self._total_proposals += 1

        if len(self._experience) > self.buffer_size:
            self._experience = self._experience[-self.buffer_size:]
            live_pids = {e["proposal_id"] for e in self._experience}
            self._outcomes = {
                pid: outcome for pid, outcome in self._outcomes.items()
                if pid in live_pids
            }

        confidence = float(1.0 / (1.0 + max(pair_score, 0.0)))
        return GraftProposal(
            proposal_id=proposal_id,
            host_algo_id=host_entry.algo_id,
            region=region,
            contract=None,
            donor_algo_id=donor_entry.algo_id,
            donor_ir=trimmed_donor,   # ← GNN-selected trimmed region, not full donor
            dependency_overrides=[],
            confidence=confidence,
            rationale=(
                f"GNN graft: host_region={len(host_region_ops)} ops, "
                f"host_start={h_start}/{len(host_entry_ops)}, "
                f"donor_start={start_idx}/{len(donor_entry_ops)}, "
                f"donor_len={donor_region_len}, pred_score={pair_score:.3f}, "
                f"temp={temperature:.2f}"
            ),
        )


    # ------------------------------------------------------------------
    # RL training step
    # ------------------------------------------------------------------

    def _train_step(self, n_steps: int = 1) -> None:
        """Joint REINFORCE training for:
          1. GraftScorer (supervised BCE on pair outcomes)
          2. DonorRegionSelectorGNN — start position (REINFORCE)
          3. DonorRegionSelectorGNN — length bucket   (REINFORCE)

        Uses an EMA baseline for variance reduction.

        Parameters
        ----------
        n_steps : int
            Number of mini-batch gradient steps per call.
        """
        # Match buffered experiences with their outcomes
        matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for exp in self._experience:
            pid = exp["proposal_id"]
            if pid in self._outcomes:
                matched.append((exp, self._outcomes[pid]))

        if len(matched) < 2:
            return

        rewards = [float(outcome.get("reward", 0.0)) for _, outcome in matched]
        target_scores = [
            float(outcome.get("graft_score", 1.5)) for _, outcome in matched
        ]
        mean_reward = float(np.mean(rewards))
        mean_score = float(np.mean(target_scores))

        # Update EMA baseline
        self._reward_baseline = (
            (1 - self._baseline_alpha) * self._reward_baseline
            + self._baseline_alpha * mean_reward
        )
        baseline = self._reward_baseline

        # Run multiple gradient steps (mini-batch sampling when > 64 samples)
        batch_size = min(64, len(matched))
        total_loss_val = 0.0

        for step in range(n_steps):
            # Sample mini-batch if matched pool is large
            if len(matched) > batch_size:
                indices = np.random.choice(len(matched), batch_size, replace=False)
                batch = [matched[i] for i in indices]
            else:
                batch = matched

            loss_terms: list[torch.Tensor] = []

            # ------------------------------------------------------------------
            # Loss 1: Pair scorer + Encoder — supervised BCE (good graft = reward > 0)
            # Re-encode with gradient so the encoder is actually trained.
            # ------------------------------------------------------------------
            h_graphs, d_graphs, targets = [], [], []
            for exp, outcome in batch:
                hg = self._graph_cache.get(exp["host_algo"])
                dg = self._graph_cache.get(exp["donor_algo"])
                if hg is not None and dg is not None:
                    h_graphs.append(hg)
                    d_graphs.append(dg)
                    targets.append(float(outcome.get("graft_score", 1.5)))

            if h_graphs:
                # Re-encode with gradient (NOT no_grad) so encoder gets trained
                h_batch_data = Batch.from_data_list(
                    [g.clone() for g in h_graphs]
                ).to(self.device)
                d_batch_data = Batch.from_data_list(
                    [g.clone() for g in d_graphs]
                ).to(self.device)
                h_embs_train = self.encoder(h_batch_data)   # [B, emb_dim] — with grad
                d_embs_train = self.encoder(d_batch_data)   # [B, emb_dim] — with grad
                target = torch.tensor(targets, dtype=torch.float32, device=self.device)
                logits = self.scorer(h_embs_train, d_embs_train).squeeze(-1)
                loss_scorer = F.mse_loss(logits, target)
                loss_terms.append(loss_scorer)

            # ------------------------------------------------------------------
            # Loss 2+3: Donor region selector — REINFORCE (start + length)
            # ------------------------------------------------------------------
            for exp, outcome in batch:
                h_feats_np = exp.get("host_region_feats")
                d_feats_np = exp.get("donor_entry_feats")
                start_idx = exp.get("donor_start_idx", 0)
                length_bucket_idx = exp.get("donor_length_bucket_idx", 0)

                if h_feats_np is None or d_feats_np is None:
                    continue

                h_feats = torch.tensor(
                    h_feats_np, dtype=torch.float32, device=self.device
                )
                d_feats = torch.tensor(
                    d_feats_np, dtype=torch.float32, device=self.device
                )
                if d_feats.shape[0] == 0 or start_idx >= d_feats.shape[0]:
                    continue

                # Re-forward (differentiable)
                start_logits, length_logits = self.donor_region_selector(h_feats, d_feats)

                # Start position loss
                log_probs_start = F.log_softmax(start_logits, dim=0)
                log_prob_start = log_probs_start[start_idx]

                # Length bucket loss
                n_buckets = len(DonorRegionSelectorGNN.LENGTH_BUCKETS)
                if length_bucket_idx < n_buckets:
                    log_probs_len = F.log_softmax(length_logits, dim=0)
                    log_prob_len = log_probs_len[length_bucket_idx]
                else:
                    log_prob_len = torch.tensor(0.0, device=self.device)

                reward = float(outcome.get("reward", 0.0))
                advantage = float(reward - baseline)
                reinforce_loss = -(log_prob_start + log_prob_len) * advantage
                loss_terms.append(reinforce_loss)

            # ------------------------------------------------------------------
            # Loss 4: RegionProposer — per-node Bernoulli REINFORCE
            # Uses correct graph-node indices to avoid node ordering mismatch.
            # Supervises the full selection mask (not just start position).
            # ------------------------------------------------------------------
            for exp, outcome in batch:
                node_indices = exp.get("entry_op_node_indices")
                host_start_idx = exp.get("host_start_idx", 0)
                host_length_bucket_idx = exp.get("host_length_bucket_idx", 0)
                if node_indices is None or len(node_indices) < 2:
                    continue
                hg = self._graph_cache.get(exp["host_algo"])
                dg = self._graph_cache.get(exp["donor_algo"])
                if hg is None or dg is None:
                    continue
                dg_data = Batch.from_data_list([dg.clone()]).to(self.device)
                d_emb_rp = self.encoder(dg_data).squeeze(0)
                graph_data = hg.clone().to(self.device)
                idx_tensor = torch.tensor(
                    node_indices, dtype=torch.long, device=self.device,
                )
                start_logits_h, length_logits_h = self.region_proposer(
                    graph_data, d_emb_rp.unsqueeze(0), idx_tensor,
                )
                if host_start_idx >= start_logits_h.shape[0]:
                    continue
                log_probs_start_h = F.log_softmax(start_logits_h, dim=0)
                log_prob_start_h = log_probs_start_h[host_start_idx]
                if host_length_bucket_idx < len(RegionProposer.LENGTH_BUCKETS):
                    log_probs_len_h = F.log_softmax(length_logits_h, dim=0)
                    log_prob_len_h = log_probs_len_h[host_length_bucket_idx]
                else:
                    log_prob_len_h = torch.tensor(0.0, device=self.device)

                reward = float(outcome.get("reward", 0.0))
                advantage_rp = float(reward - baseline)
                rp_loss = -(log_prob_start_h + log_prob_len_h) * advantage_rp
                loss_terms.append(rp_loss)

            if not loss_terms:
                continue

            self.optimizer.zero_grad()
            total_loss: torch.Tensor = sum(loss_terms)  # type: ignore[assignment]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._all_params, max_norm=1.0)
            self.optimizer.step()
            total_loss_val = total_loss.item()

        # Replay buffer: keep matched samples for reuse, only prune evicted items.
        live_pids = {e["proposal_id"] for e in self._experience}
        orphaned = [pid for pid in list(self._outcomes) if pid not in live_pids]
        for pid in orphaned:
            self._outcomes.pop(pid, None)

        self._last_train_stats = {
            "matched_samples": len(matched),
            "train_steps": n_steps,
            "mean_reward": mean_reward,
            "mean_graft_score": mean_score,
            "baseline": baseline,
            "loss": total_loss_val,
        }
        logger.info(
            "GNN train: %d samples, %d steps, avg_reward=%.4f, avg_score=%.4f, baseline=%.4f, loss=%.4f",
            len(matched),
            n_steps,
            mean_reward,
            mean_score,
            baseline,
            total_loss_val,
        )


    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_op_feats(self, ir: FunctionIR, op_ids: list[str]) -> np.ndarray:
        """Return node-feature matrix [len(op_ids), NODE_DIM] for specified ops."""
        feats = []
        for oid in op_ids:
            op = ir.ops.get(oid)
            if op is None:
                continue
            one_hot = [0.0] * _N_OPCODES
            one_hot[_opcode_idx(op.opcode)] = 1.0
            callee = op.attrs.get("callee", op.attrs.get("name", ""))
            callee_feat = _hash_callee(callee) if callee else [0.0] * _CALLEE_FEATURES
            feats.append(one_hot + callee_feat)
        if not feats:
            return np.zeros((1, _NODE_DIM), dtype=np.float32)
        return np.array(feats, dtype=np.float32)

    def get_stats(self) -> dict[str, Any]:
        """Return training statistics."""
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
