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
        """Returns [N, 1] logits."""
        combined = torch.cat([host_emb, donor_emb], dim=-1)
        return self.net(combined)


# ---------------------------------------------------------------------------
# Region proposer: node-level scores for region selection
# ---------------------------------------------------------------------------

class RegionProposer(nn.Module):
    """Node-level scorer: for each op in host, predict inclusion probability
    in the graft region."""

    def __init__(self, node_dim: int = _NODE_DIM, hidden: int = 64, context_dim: int = 32):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden, heads=4, concat=False)
        self.conv2 = GATConv(hidden, hidden, heads=4, concat=False)
        # Context-conditioned scoring
        self.scorer = nn.Sequential(
            nn.Linear(hidden + context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: Data, donor_emb: torch.Tensor) -> torch.Tensor:
        """Returns [N_nodes] inclusion logits for each op in host."""
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))

        # Expand donor embedding to each node
        n_nodes = x.size(0)
        donor_exp = donor_emb.expand(n_nodes, -1)
        ctx = torch.cat([x, donor_exp], dim=-1)
        logits = self.scorer(ctx).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Donor region selector: cross-attention GNN
# ---------------------------------------------------------------------------

class DonorRegionSelectorGNN(nn.Module):
    """Cross-attention GNN that selects the best start position in the
    donor's entry block for grafting.

    Architecture:
      1. Project host region op features → query space (avg-pooled)
      2. Project donor entry block op features → key/value space
      3. Context-aware scoring: for each donor op, score = MLP([donor_proj, host_query])
      4. Output: [N] start-position logits

    Trained via REINFORCE:
      loss = -log P(start_idx | host, donor) * advantage
    where advantage = reward - moving_avg_baseline.
    """

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
        # Concat(donor_op_emb, host_query) → score
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        host_region_feats: torch.Tensor,  # [H, node_dim]
        donor_entry_feats: torch.Tensor,  # [N, node_dim]
    ) -> torch.Tensor:
        """Returns [N] logits for start position in donor entry block."""
        if host_region_feats.shape[0] == 0:
            return torch.zeros(donor_entry_feats.shape[0], device=donor_entry_feats.device)
        if donor_entry_feats.shape[0] == 0:
            return torch.zeros(1, device=host_region_feats.device)

        h = self.host_proj(host_region_feats)   # [H, hidden]
        d = self.donor_proj(donor_entry_feats)  # [N, hidden]

        # Average-pool host features → single query vector
        h_query = h.mean(0, keepdim=True)       # [1, hidden]
        h_exp = h_query.expand(d.shape[0], -1)  # [N, hidden]

        combined = torch.cat([d, h_exp], dim=-1)  # [N, hidden*2]
        logits = self.score_head(combined).squeeze(-1)  # [N]
        return logits


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
        buffer_size: int = 512,
        train_interval: int = 1,
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
        self._outcomes: dict[str, float] = {}
        # Graph cache: algo_id → Data
        self._graph_cache: dict[str, Data] = {}
        # Embedding cache (per generation)
        self._emb_cache: dict[str, torch.Tensor] = {}

        self._generation = 0
        self._total_proposals = 0
        self._total_rewards = 0.0

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

        # Train if we have enough experience
        if generation > 0 and generation % self.train_interval == 0:
            self._train_step()

        if len(entries) < 2:
            return []

        # Encode all IRs
        self._encode_entries(entries)

        # Score all (host, donor) pairs
        pair_scores = self._score_pairs(entries)

        # Select top-k pairs
        top_pairs = sorted(pair_scores, key=lambda x: x[2], reverse=True)[
            : self.top_k_pairs
        ]

        # For top pairs, propose regions
        proposals: list[GraftProposal] = []
        for host_entry, donor_entry, score in top_pairs:
            if len(proposals) >= self.max_proposals_per_gen:
                break
            proposal = self._propose_graft(host_entry, donor_entry, score)
            if proposal is not None:
                proposals.append(proposal)

        return proposals

    # ------------------------------------------------------------------
    # Reward feedback
    # ------------------------------------------------------------------

    def record_outcome(self, proposal_id: str, reward: float) -> None:
        """Record the outcome of a graft proposal for RL training."""
        # Clamp reward to avoid -inf/nan poisoning RL training
        reward = max(reward, -10.0)
        self._outcomes[proposal_id] = reward
        self._total_rewards += reward

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _encode_entries(self, entries: list[AlgorithmEntry]) -> None:
        """Encode all entries to graph embeddings (cached)."""
        self._emb_cache.clear()
        graphs: list[Data] = []
        keys: list[str] = []

        for entry in entries:
            if entry.algo_id not in self._graph_cache:
                self._graph_cache[entry.algo_id] = ir_to_graph(entry.ir)
            graphs.append(self._graph_cache[entry.algo_id])
            keys.append(entry.algo_id)

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
        """Score all distinct (host, donor) pairs."""
        results: list[tuple[AlgorithmEntry, AlgorithmEntry, float]] = []
        with torch.no_grad():
            for i, host in enumerate(entries):
                h_emb = self._emb_cache.get(host.algo_id)
                if h_emb is None:
                    continue
                for j, donor in enumerate(entries):
                    if i == j:
                        continue
                    d_emb = self._emb_cache.get(donor.algo_id)
                    if d_emb is None:
                        continue
                    score = self.scorer(
                        h_emb.unsqueeze(0), d_emb.unsqueeze(0),
                    ).item()
                    results.append((host, donor, score))
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

        all_op_ids = list(host_ir.ops.keys())
        op_to_idx = {oid: i for i, oid in enumerate(all_op_ids)}

        with torch.no_grad():
            graph_data = host_graph.clone().to(self.device)
            node_logits = self.region_proposer(graph_data, donor_emb.unsqueeze(0))
            node_scores = torch.sigmoid(node_logits).cpu().numpy()

        entry_scores = np.array([
            float(node_scores[op_to_idx[oid]])
            if oid in op_to_idx and op_to_idx[oid] < len(node_scores) else 0.0
            for oid in host_entry_ops
        ])

        host_window = min(
            self.max_region_size,
            max(self.min_region_size, len(host_entry_ops) // 4),
        )
        best_pos = int(np.argmax(entry_scores))
        h_start = max(0, best_pos - host_window // 2)
        h_end = min(len(host_entry_ops), h_start + host_window)
        h_start = max(0, h_end - host_window)
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
        # 4. DonorRegionSelectorGNN: cross-attend host→donor, sample start
        # ------------------------------------------------------------------
        host_region_feats_np = self._get_op_feats(host_ir, host_region_ops)
        donor_entry_feats_np = self._get_op_feats(donor_ir, donor_entry_ops)

        h_feats = torch.tensor(host_region_feats_np, dtype=torch.float32, device=self.device)
        d_feats = torch.tensor(donor_entry_feats_np, dtype=torch.float32, device=self.device)

        # Temperature annealing: high early (exploration) → low later (exploitation)
        temperature = max(0.3, 2.0 - 0.04 * self._generation)

        with torch.no_grad():
            start_logits = self.donor_region_selector(h_feats, d_feats)  # [N]
            if start_logits.shape[0] == 0:
                return None
            start_probs = F.softmax(start_logits / temperature, dim=0)
            if torch.any(torch.isnan(start_probs)) or torch.any(torch.isinf(start_probs)):
                start_idx = 0
            else:
                start_idx = int(torch.multinomial(start_probs, 1).item())

        # Donor region length = host window size (minimal, matched scope)
        donor_region_len = max(len(host_region_ops), self.min_region_size)
        max_possible = len(donor_entry_ops) - start_idx
        donor_region_len = min(donor_region_len, max_possible)
        if donor_region_len <= 0:
            start_idx = max(0, len(donor_entry_ops) - self.min_region_size)
            donor_region_len = len(donor_entry_ops) - start_idx
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
            "donor_start_idx": int(start_idx),
            "donor_region_len": int(donor_region_len),
            "host_algo": host_entry.algo_id,
            "donor_algo": donor_entry.algo_id,
            "pair_score": float(pair_score),
            "generation": self._generation,
        })
        self._total_proposals += 1

        if len(self._experience) > self.buffer_size:
            self._experience = self._experience[-self.buffer_size:]

        confidence = float(torch.sigmoid(torch.tensor(pair_score)).item())
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
                f"donor_start={start_idx}/{len(donor_entry_ops)}, "
                f"donor_len={donor_region_len}, pair_score={pair_score:.3f}, "
                f"temp={temperature:.2f}"
            ),
        )


    # ------------------------------------------------------------------
    # RL training step
    # ------------------------------------------------------------------

    def _train_step(self) -> None:
        """Joint REINFORCE training for:
          1. GraftScorer (supervised BCE on pair outcomes)
          2. DonorRegionSelectorGNN (REINFORCE for start-position selection)

        Uses an EMA baseline for variance reduction.
        """
        # Match buffered experiences with their outcomes
        matched: list[tuple[dict, float]] = []
        for exp in self._experience:
            pid = exp["proposal_id"]
            if pid in self._outcomes:
                matched.append((exp, self._outcomes[pid]))

        if len(matched) < 2:
            return

        rewards = [r for _, r in matched]
        mean_reward = float(np.mean(rewards))

        # Update EMA baseline
        self._reward_baseline = (
            (1 - self._baseline_alpha) * self._reward_baseline
            + self._baseline_alpha * mean_reward
        )
        baseline = self._reward_baseline

        loss_terms: list[torch.Tensor] = []

        # ------------------------------------------------------------------
        # Loss 1: Pair scorer — supervised BCE (good graft = reward > 0)
        # ------------------------------------------------------------------
        h_embs, d_embs, labels = [], [], []
        for exp, reward in matched:
            h = self._emb_cache.get(exp["host_algo"])
            d = self._emb_cache.get(exp["donor_algo"])
            if h is not None and d is not None:
                h_embs.append(h)
                d_embs.append(d)
                labels.append(1.0 if reward > 0 else 0.0)

        if h_embs:
            h_batch = torch.stack(h_embs)
            d_batch = torch.stack(d_embs)
            target = torch.tensor(labels, dtype=torch.float32, device=self.device)
            logits = self.scorer(h_batch, d_batch).squeeze(-1)
            loss_scorer = F.binary_cross_entropy_with_logits(logits, target)
            loss_terms.append(loss_scorer)

        # ------------------------------------------------------------------
        # Loss 2: Donor region selector — REINFORCE
        #   loss = -log P(start_idx | host, donor) * (reward - baseline)
        # ------------------------------------------------------------------
        for exp, reward in matched:
            h_feats_np = exp.get("host_region_feats")
            d_feats_np = exp.get("donor_entry_feats")
            start_idx = exp.get("donor_start_idx", 0)

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
            start_logits = self.donor_region_selector(h_feats, d_feats)  # [N]
            log_probs = F.log_softmax(start_logits, dim=0)               # [N]
            log_prob_taken = log_probs[start_idx]                        # scalar

            advantage = float(reward - baseline)
            reinforce_loss = -log_prob_taken * advantage
            loss_terms.append(reinforce_loss)

        if not loss_terms:
            return

        self.optimizer.zero_grad()
        total_loss: torch.Tensor = sum(loss_terms)  # type: ignore[assignment]
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._all_params, max_norm=1.0)
        self.optimizer.step()

        # Clear used outcomes from buffer
        used_pids = {exp["proposal_id"] for exp, _ in matched}
        self._experience = [
            e for e in self._experience if e["proposal_id"] not in used_pids
        ]
        for pid in used_pids:
            self._outcomes.pop(pid, None)

        # Prune orphaned outcomes (proposals already processed/expired)
        # Keep only outcomes that still have a corresponding experience entry.
        live_pids = {e["proposal_id"] for e in self._experience}
        orphaned = [pid for pid in list(self._outcomes) if pid not in live_pids]
        # Only prune if buffer is large to avoid aggressive deletion of fresh outcomes
        if len(self._outcomes) > self.buffer_size // 2:
            for pid in orphaned:
                self._outcomes.pop(pid, None)

        logger.info(
            "GNN train: %d samples, avg_reward=%.4f, baseline=%.4f, loss=%.4f",
            len(matched),
            mean_reward,
            baseline,
            total_loss.item(),
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
        }
