"""
Stack Decoder with Algorithmic Hole.
Best-First Search (Stack Algorithm) for MIMO detection.
The scoring function is delegated to the evolved Push program via the VM.
"""
import numpy as np
import heapq
from typing import List, Optional, Tuple
from stacks import TreeNode, SearchTreeGraph
from vm import MIMOPushVM, Instruction


def qpsk_constellation():
    """QPSK constellation points (normalized)."""
    s = 1.0 / np.sqrt(2)
    return np.array([s + 1j*s, s - 1j*s, -s + 1j*s, -s - 1j*s])


def qam16_constellation():
    """16-QAM constellation points (normalized)."""
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    return np.array([r + 1j*i for r in levels for i in levels])


class StackDecoder:
    """
    Best-First Stack Decoder for MIMO detection.
    Uses QR decomposition and delegates node scoring to an evolved program.
    """

    def __init__(self, Nt: int, Nr: int, constellation: np.ndarray,
                 max_nodes: int = 5000, vm: Optional[MIMOPushVM] = None):
        """
        Args:
            Nt: Number of transmit antennas
            Nr: Number of receive antennas
            constellation: Complex constellation points
            max_nodes: Maximum nodes to expand before giving up
            vm: The MIMO-Push VM instance for scoring
        """
        self.Nt = Nt
        self.Nr = Nr
        self.constellation = constellation
        self.M = len(constellation)  # constellation size
        self.max_nodes = max_nodes
        self.vm = vm or MIMOPushVM()
        self.search_tree = None
        self.last_run_stats = {}

    def detect(self, H: np.ndarray, y: np.ndarray,
               program: List[Instruction]) -> Tuple[np.ndarray, float]:
        """
        Run stack-based detection with the given evolved program as the scoring function.

        Args:
            H: Nr x Nt channel matrix
            y: Nr x 1 received signal
            program: The evolved Push program for emergent_metric

        Returns:
            x_hat: Detected Nt x 1 symbol vector
            total_flops: Total FLOPs consumed
        """
        # QR decomposition: H = Q @ R
        Q, R = np.linalg.qr(H, mode='reduced')  # Q: Nr x Nt, R: Nt x Nt
        y_tilde = Q.conj().T @ y  # Nt x 1

        # Initialize search tree
        self.search_tree = SearchTreeGraph()
        root = self.search_tree.create_root(layer=self.Nt)  # root = bottom of tree
        self.last_run_stats = {
            'nodes_expanded': 0,
            'rescore_events': 0,
            'rescore_delta_sum': 0.0,
            'rescore_samples': 0,
            'avg_rescore_delta': 0.0,
            'rank_change_sum': 0.0,
            'rank_change_events': 0,
            'avg_rank_change': 0.0,
            'max_frontier_size': 0,
        }

        # Priority queue: (score, version, counter, node)
        counter = 0
        pq = []
        total_flops = 0

        # Expand root: create children for the bottom layer (layer Nt)
        for sym in self.constellation:
            partial = np.array([sym])
            # Local Euclidean distance for layer Nt (0-indexed: Nt-1)
            k = self.Nt - 1  # 0-indexed layer
            residual = y_tilde[k] - R[k, k] * sym
            dist = float(np.abs(residual) ** 2)
            total_flops += 10

            child = self.search_tree.add_child(
                parent=root, layer=self.Nt - 1,
                symbol=sym, cumulative_distance=dist,
                partial_symbols=partial
            )

            # Score child using evolved program
            score = self._score_node(child, R, y_tilde, program)
            child.intrinsic_score = score
            child.dynamic_score = score
            child.score = score
            total_flops += self.vm.flops_count
            heapq.heappush(pq, (score, child.queue_version, counter, child))
            counter += 1

        self.search_tree.refresh_all_statistics()
        best_node = None
        best_score = float('inf')
        for queued_score, _, _, queued_node in pq:
            if queued_score < best_score:
                best_score = queued_score
                best_node = queued_node

        nodes_expanded = 0

        while pq and nodes_expanded < self.max_nodes:
            score, version, _, node = heapq.heappop(pq)
            if version != node.queue_version or abs(score - node.dynamic_score) > 1e-12:
                continue
            nodes_expanded += 1
            self.last_run_stats['nodes_expanded'] = nodes_expanded

            if score < best_score:
                best_score = score
                best_node = node

            # Check if this is a complete path (reached layer 0)
            if node.layer == 0:
                # We have a full solution
                x_hat = node.partial_symbols[::-1]  # reverse: layer 1 first
                self._finalize_run_stats(best_score)
                return x_hat, total_flops

            self.search_tree.mark_expanded(node)

            # Expand this node: create children for the next layer up
            next_layer = node.layer - 1
            k = next_layer  # 0-indexed

            for sym in self.constellation:
                # Build partial symbol vector
                new_partial = np.append(node.partial_symbols, sym)

                # Compute local distance increment
                # residual = y_tilde[k] - sum_{j=k}^{Nt-1} R[k,j] * x[j]
                # The partial symbols are in reverse order (bottom to top)
                n_decided = len(new_partial)
                interference = 0.0
                for idx in range(n_decided):
                    col = self.Nt - 1 - idx  # map partial index to column
                    if col < R.shape[1]:  # safety
                        interference += R[k, col] * new_partial[idx]
                        total_flops += 8
                residual = y_tilde[k] - interference
                local_dist = float(np.abs(residual) ** 2)
                cum_dist = node.cumulative_distance + local_dist
                total_flops += 6

                child = self.search_tree.add_child(
                    parent=node, layer=next_layer,
                    symbol=sym, cumulative_distance=cum_dist,
                    partial_symbols=new_partial
                )

                # Score using evolved program
                score = self._score_node(child, R, y_tilde, program)
                child.intrinsic_score = score
                child.dynamic_score = score
                child.score = score
                total_flops += self.vm.flops_count
                heapq.heappush(pq, (score, child.queue_version, counter, child))
                counter += 1

                if score < best_score:
                    best_score = score
                    best_node = child

            self.search_tree.refresh_all_statistics()
            counter = self._rescore_frontier_nodes(pq, R, y_tilde, program, counter, node)
            self.search_tree.refresh_all_statistics()
            self.last_run_stats['max_frontier_size'] = max(
                self.last_run_stats['max_frontier_size'],
                self.search_tree.open_node_count(),
            )

        if best_node is None:
            best_node = root
        x_hat, completion_flops = self._complete_partially_decided_path(best_node, R, y_tilde)
        self._finalize_run_stats(best_score)
        return x_hat, total_flops + completion_flops

    def _score_node(self, node: TreeNode, R: np.ndarray,
                    y_tilde: np.ndarray, program: List[Instruction]) -> float:
        """Score a candidate node using the evolved Push program."""
        # Build environment snapshot
        x_partial = node.partial_symbols
        depth_k = node.layer

        # Inject environment and execute
        node.visit_count += 1
        self.vm.inject_environment(
            R=R, y_tilde=y_tilde, x_partial=x_partial,
            graph=self.search_tree, candidate_node=node,
            depth_k=depth_k
        )
        score = self.vm.run(program)
        return score

    def _rescore_frontier_nodes(self, pq, R: np.ndarray, y_tilde: np.ndarray,
                                program: List[Instruction], counter: int,
                                focus_node: TreeNode) -> int:
        frontier = self.search_tree.frontier_nodes()
        if not frontier:
            return counter

        if len(frontier) > 96:
            frontier = sorted(
                frontier,
                key=lambda candidate: (
                    abs(candidate.layer - focus_node.layer),
                    candidate.cumulative_distance,
                )
            )[:96]

        self.last_run_stats['rescore_events'] += 1
        old_scores = {
            candidate.node_id: (
                candidate.dynamic_score if np.isfinite(candidate.dynamic_score) else candidate.cumulative_distance
            )
            for candidate in frontier
        }

        for candidate in frontier:
            old_score = old_scores[candidate.node_id]
            new_score = self._score_node(candidate, R, y_tilde, program)
            candidate.intrinsic_score = new_score
            candidate.dynamic_score = new_score
            candidate.score = new_score
            candidate.queue_version += 1
            heapq.heappush(pq, (new_score, candidate.queue_version, counter, candidate))
            counter += 1

            delta = new_score - old_score
            self.search_tree.propagate_rescore_delta(candidate, delta)
            self.last_run_stats['rescore_delta_sum'] += abs(delta)
            self.last_run_stats['rescore_samples'] += 1

        if len(frontier) > 1:
            old_order = [
                candidate.node_id for candidate in sorted(
                    frontier,
                    key=lambda item: (old_scores[item.node_id], item.node_id),
                )
            ]
            new_order = [
                candidate.node_id for candidate in sorted(
                    frontier,
                    key=lambda item: (item.dynamic_score, item.node_id),
                )
            ]
            old_rank = {node_id: rank for rank, node_id in enumerate(old_order)}
            new_rank = {node_id: rank for rank, node_id in enumerate(new_order)}
            normalizer = float(len(frontier) * max(1, len(frontier) - 1))
            rank_change = sum(abs(old_rank[node_id] - new_rank[node_id]) for node_id in old_rank) / normalizer
            self.last_run_stats['rank_change_sum'] += rank_change
            self.last_run_stats['rank_change_events'] += 1

        return counter

    def _finalize_run_stats(self, best_score: float):
        samples = max(1, self.last_run_stats['rescore_samples'])
        self.last_run_stats['avg_rescore_delta'] = self.last_run_stats['rescore_delta_sum'] / samples
        rank_events = max(1, self.last_run_stats['rank_change_events'])
        self.last_run_stats['avg_rank_change'] = self.last_run_stats['rank_change_sum'] / rank_events
        self.last_run_stats['best_score'] = float(best_score)

    def _complete_partially_decided_path(self, node: TreeNode, R: np.ndarray,
                                         y_tilde: np.ndarray) -> Tuple[np.ndarray, float]:
        """Complete a partial path with bounded greedy QR-metric decisions."""
        decided_suffix = list(node.partial_symbols)
        current_layer = node.layer - 1
        completion_flops = 0.0

        while current_layer >= 0:
            best_symbol = self.constellation[0]
            best_local_metric = float('inf')
            for sym in self.constellation:
                candidate_suffix = decided_suffix + [sym]
                interference = 0.0
                for idx, suffix_symbol in enumerate(candidate_suffix):
                    col = self.Nt - 1 - idx
                    interference += R[current_layer, col] * suffix_symbol
                    completion_flops += 8
                residual = y_tilde[current_layer] - interference
                local_metric = float(np.abs(residual) ** 2)
                completion_flops += 6
                if local_metric < best_local_metric:
                    best_local_metric = local_metric
                    best_symbol = sym
            decided_suffix.append(best_symbol)
            current_layer -= 1

        if not decided_suffix:
            return np.zeros(self.Nt, dtype=self.constellation.dtype), completion_flops
        return np.asarray(decided_suffix[::-1]), completion_flops


def lmmse_detect(H: np.ndarray, y: np.ndarray, noise_var: float,
                 constellation: np.ndarray) -> np.ndarray:
    """Standard LMMSE detector for baseline comparison."""
    Nr, Nt = H.shape
    # W_LMMSE = (H^H H + sigma^2 I)^{-1} H^H
    HH = H.conj().T @ H
    W = np.linalg.solve(HH + noise_var * np.eye(Nt), H.conj().T)
    x_hat = W @ y
    # Hard decision
    detected = np.array([
        constellation[np.argmin(np.abs(constellation - x_hat[i]))]
        for i in range(Nt)
    ])
    return detected
