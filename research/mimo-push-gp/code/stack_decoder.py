"""
Stack Decoder with Algorithmic Hole + K-Best / LMMSE baselines.
All baselines count FLOPs at runtime for fair comparison.
"""
import numpy as np
import heapq
from typing import List, Optional, Tuple
from stacks import TreeNode, SearchTreeGraph
from vm import MIMOPushVM, Instruction


# --------------------------------------------------------------------------
# Constellations
# --------------------------------------------------------------------------

def qpsk_constellation() -> np.ndarray:
    s = 1.0 / np.sqrt(2)
    return np.array([s + 1j*s, s - 1j*s, -s + 1j*s, -s - 1j*s])


def qam16_constellation() -> np.ndarray:
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    return np.array([r + 1j*i for r in levels for i in levels])


# --------------------------------------------------------------------------
# K-Best Detector (with runtime FLOPs counting)
# --------------------------------------------------------------------------

def kbest_detect(H: np.ndarray, y: np.ndarray,
                 constellation: np.ndarray, K: int) -> Tuple[np.ndarray, int]:
    """K-Best tree search detector. Returns (x_hat, flops)."""
    Nr, Nt = H.shape
    M = len(constellation)
    flops = 0

    # QR decomposition
    Q, R = np.linalg.qr(H, mode='reduced')
    flops += 2 * Nr * Nt * Nt  # Householder reflections
    y_tilde = Q.conj().T @ y
    flops += 8 * Nt * Nr  # complex mat-vec

    # Layer Nt-1 (bottom): create initial M candidates
    k = Nt - 1
    candidates = []
    for sym in constellation:
        residual = y_tilde[k] - R[k, k] * sym
        dist = float(np.abs(residual) ** 2)
        flops += 6 + 2 + 3  # cmul + csub + |.|^2
        candidates.append(([sym], dist))
    candidates.sort(key=lambda x: x[1])
    candidates = candidates[:K]
    flops += int(M * max(1, np.log2(M)))

    # Extend layer by layer (Nt-2 down to 0)
    for k in range(Nt - 2, -1, -1):
        new_cands = []
        for partial, cum_d in candidates:
            for sym in constellation:
                new_partial = partial + [sym]
                n_dec = len(new_partial)
                interference = 0.0 + 0.0j
                for j in range(n_dec):
                    col = Nt - 1 - j
                    interference += R[k, col] * new_partial[j]
                    flops += 8  # complex mul + add
                residual = y_tilde[k] - interference
                flops += 2
                ld = float(np.abs(residual) ** 2)
                flops += 3
                new_cands.append((new_partial, cum_d + ld))
                flops += 1
        new_cands.sort(key=lambda x: x[1])
        n_total = len(new_cands)
        flops += int(n_total * max(1, np.log2(max(2, n_total))))
        candidates = new_cands[:K]

    best_partial, _ = candidates[0]
    x_hat = np.array(best_partial[::-1])
    return x_hat, flops


# --------------------------------------------------------------------------
# LMMSE Detector (with runtime FLOPs counting)
# --------------------------------------------------------------------------

def lmmse_detect(H: np.ndarray, y: np.ndarray, noise_var: float,
                 constellation: np.ndarray) -> Tuple[np.ndarray, int]:
    """LMMSE detector. Returns (x_hat, flops)."""
    Nr, Nt = H.shape
    flops = 0

    # H^H H: Nt x Nt
    HH = H.conj().T @ H
    flops += 8 * Nt * Nt * Nr  # complex mat mul
    # Regularize
    HH_reg = HH + noise_var * np.eye(Nt)
    flops += Nt
    # H^H y: Nt x 1
    Hy = H.conj().T @ y
    flops += 8 * Nt * Nr
    # Solve  (H^H H + σ²I) x̃ = H^H y
    x_hat_soft = np.linalg.solve(HH_reg, Hy)
    flops += int(2 * Nt * Nt * Nt / 3)  # LU factorization

    # Slice to constellation
    detected = np.empty(Nt, dtype=constellation.dtype)
    for i in range(Nt):
        dists = np.abs(constellation - x_hat_soft[i]) ** 2
        detected[i] = constellation[np.argmin(dists)]
        flops += 6 * len(constellation)  # distance computations

    return detected, flops


# --------------------------------------------------------------------------
# Evolved Stack Decoder
# --------------------------------------------------------------------------

class StackDecoder:
    """Best-First Stack Decoder whose scoring function is an evolved program."""

    def __init__(self, Nt: int, Nr: int, constellation: np.ndarray,
                 max_nodes: int = 5000, vm: Optional[MIMOPushVM] = None,
                 rescore_interval: int = 0):
        self.Nt = Nt
        self.Nr = Nr
        self.constellation = constellation
        self.M = len(constellation)
        self.max_nodes = max_nodes
        self.vm = vm or MIMOPushVM()
        self.search_tree: Optional[SearchTreeGraph] = None
        self.rescore_interval = rescore_interval  # 0 = never rescore during search

    def detect(self, H: np.ndarray, y: np.ndarray,
               program: List[Instruction],
               noise_var: float = 1.0) -> Tuple[np.ndarray, int]:
        """Run evolved stack decoder. Returns (x_hat, total_flops)."""
        Nr, Nt = H.shape
        flops = 0

        # QR decomposition
        Q, R = np.linalg.qr(H, mode='reduced')
        flops += 2 * Nr * Nt * Nt
        y_tilde = Q.conj().T @ y
        flops += 8 * Nt * Nr

        # State for this decode call
        self._current_noise_var = noise_var

        # Initialise search tree
        self.search_tree = SearchTreeGraph()
        root = self.search_tree.create_root(layer=Nt)

        counter = 0
        pq: list = []

        # Expand root → children at bottom layer
        k0 = Nt - 1
        inf_count = 0
        for sym in self.constellation:
            residual = y_tilde[k0] - R[k0, k0] * sym
            ld = float(np.abs(residual) ** 2)
            flops += 11
            child = self.search_tree.add_child(
                parent=root, layer=Nt - 1, symbol=sym,
                local_dist=ld, cum_dist=ld,
                partial_symbols=np.array([sym]),
            )
            score = self._score_node(child, R, y_tilde, program)
            child.score = score
            flops += self.vm.flops_count
            if score == float('inf'):
                inf_count += 1
            heapq.heappush(pq, (score, child.queue_version, counter, child))
            counter += 1

        # Early termination: if ALL initial scores are inf, program is broken
        if inf_count >= self.M:
            x_hat, cf = self._complete_path(root, R, y_tilde)
            return x_hat, flops + cf

        best_node = None
        best_score = float('inf')
        nodes_expanded = 0

        while pq and nodes_expanded < self.max_nodes:
            sc, ver, _, node = heapq.heappop(pq)
            if ver != node.queue_version:
                continue
            nodes_expanded += 1

            if sc < best_score:
                best_score = sc
                best_node = node

            # Complete path?
            if node.layer == 0:
                x_hat = node.partial_symbols[::-1]
                return x_hat, flops

            self.search_tree.mark_expanded(node)
            next_layer = node.layer - 1

            # Expand children
            for sym in self.constellation:
                new_partial = np.append(node.partial_symbols, sym)
                n_dec = len(new_partial)
                interference = 0.0 + 0.0j
                for j in range(n_dec):
                    col = Nt - 1 - j
                    interference += R[next_layer, col] * new_partial[j]
                    flops += 8
                residual = y_tilde[next_layer] - interference
                flops += 2
                ld = float(np.abs(residual) ** 2)
                flops += 3
                cd = node.cum_dist + ld
                flops += 1

                child = self.search_tree.add_child(
                    parent=node, layer=next_layer, symbol=sym,
                    local_dist=ld, cum_dist=cd,
                    partial_symbols=new_partial,
                )
                score = self._score_node(child, R, y_tilde, program)
                child.score = score
                flops += self.vm.flops_count
                heapq.heappush(pq, (score, child.queue_version, counter, child))
                counter += 1

                if score < best_score:
                    best_score = score
                    best_node = child

            # Frontier rescoring (allows asynchronous message passing)
            if self.rescore_interval > 0 and nodes_expanded % self.rescore_interval == 0:
                counter, rescore_flops = self._rescore_frontier(
                    pq, R, y_tilde, program, counter, node)
                flops += rescore_flops

        # Budget exhausted — complete best partial path greedily
        if best_node is None:
            best_node = root
        x_hat, comp_flops = self._complete_path(best_node, R, y_tilde)
        flops += comp_flops
        return x_hat, flops

    # ---- scoring ----
    def _score_node(self, node: TreeNode, R: np.ndarray,
                    y_tilde: np.ndarray, program: List[Instruction]) -> float:
        self.vm.inject_environment(
            R=R, y_tilde=y_tilde,
            x_partial=node.partial_symbols,
            graph=self.search_tree, candidate_node=node,
            depth_k=node.layer,
            constellation=self.constellation,
            noise_var=getattr(self, '_current_noise_var', 1.0),
        )
        correction = self.vm.run(program)
        # Residual scoring: base = cum_dist, correction = evolved program output
        if correction == float('inf') or not np.isfinite(correction):
            return node.cum_dist  # Default to distance ordering
        return node.cum_dist + correction

    # ---- frontier rescoring ----
    def _rescore_frontier(self, pq, R, y_tilde, program, counter, focus):
        frontier = self.search_tree.frontier_nodes()
        if not frontier:
            return counter, 0
        # Limit rescoring budget
        if len(frontier) > 24:
            frontier = sorted(frontier,
                              key=lambda c: abs(c.layer - focus.layer))[:24]
        total_flops = 0
        for cand in frontier:
            new_score = self._score_node(cand, R, y_tilde, program)
            total_flops += self.vm.flops_count
            if abs(new_score - cand.score) > 1e-12:
                cand.score = new_score
                cand.queue_version += 1
                heapq.heappush(pq, (new_score, cand.queue_version, counter, cand))
                counter += 1
        return counter, total_flops

    # ---- greedy completion ----
    def _complete_path(self, node: TreeNode, R: np.ndarray,
                       y_tilde: np.ndarray) -> Tuple[np.ndarray, int]:
        decided = list(node.partial_symbols)
        cur_layer = node.layer - 1
        flops = 0
        while cur_layer >= 0:
            best_sym = self.constellation[0]
            best_ld = float('inf')
            for sym in self.constellation:
                cand = decided + [sym]
                interference = 0.0 + 0.0j
                for j in range(len(cand)):
                    col = self.Nt - 1 - j
                    interference += R[cur_layer, col] * cand[j]
                    flops += 8
                residual = y_tilde[cur_layer] - interference
                ld = float(np.abs(residual) ** 2)
                flops += 5
                if ld < best_ld:
                    best_ld = ld
                    best_sym = sym
            decided.append(best_sym)
            cur_layer -= 1
        if not decided:
            return np.zeros(self.Nt, dtype=self.constellation.dtype), flops
        return np.asarray(decided[::-1]), flops
