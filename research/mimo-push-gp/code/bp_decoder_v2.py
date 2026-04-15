"""
Structured BP Stack Decoder v2 — Full-Tree BP Sweeps.

After each expansion (creating M new children), the framework runs:
  1. Full UP-sweep: ALL nodes bottom-up (leaves→root)   [F_up]
  2. Full DOWN-sweep: ALL nodes top-down (root→leaves)   [F_down]
  3. Score-pass: ALL frontier nodes rescored              [F_belief]
  4. H_halt decides if another iteration is needed

This ensures that EVERY previously explored node (including siblings,
uncles, cousins) gets its beliefs updated when new information arrives.

The framework prescribes ZERO math. GP evolves all formulas.
"""
import numpy as np
import heapq
from typing import List, Optional, Tuple
from collections import deque

from stacks import TreeNode, SearchTreeGraph
from vm import MIMOPushVM, Instruction


def qam16_constellation() -> np.ndarray:
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    return np.array([r + 1j * i for r in levels for i in levels])


def qpsk_constellation() -> np.ndarray:
    s = 1.0 / np.sqrt(2)
    return np.array([s + 1j * s, s - 1j * s, -s + 1j * s, -s - 1j * s])


class StructuredBPDecoder:
    """Stack decoder with full-tree structured BP.

    4 evolved programs:
      prog_down:   F_down(M_parent_down, C_i) → M_i_down
      prog_up:     F_up({M_j_up, C_j}_{children}) → M_i_up
      prog_belief: F_belief(M_i_up, M_i_down, cum_dist, ...) → B_i
      prog_halt:   H_halt(old_root_m_up, new_root_m_up) → Bool (stop BP iters?)

    Per expansion cycle:
      1. Create M children, compute local_dist, cum_dist
      2. Full UP-sweep: bottom-up over ALL tree nodes
      3. Full DOWN-sweep: top-down over ALL tree nodes
      4. Score ALL frontier nodes with F_belief
      5. If H_halt says False and iterations remain, goto 2
    """

    def __init__(self, Nt: int, Nr: int, constellation: np.ndarray,
                 max_nodes: int = 500,
                 vm: Optional[MIMOPushVM] = None,
                 max_bp_iters: int = 2):
        self.Nt = Nt
        self.Nr = Nr
        self.constellation = constellation
        self.M = len(constellation)
        self.max_nodes = max_nodes
        self.vm = vm or MIMOPushVM()
        self.max_bp_iters = max_bp_iters
        self.search_tree: Optional[SearchTreeGraph] = None
        self.total_bp_calls = 0

    def detect(self, H: np.ndarray, y: np.ndarray,
               prog_down: List[Instruction],
               prog_up: List[Instruction],
               prog_belief: List[Instruction],
               prog_halt: List[Instruction],
               noise_var: float = 1.0) -> Tuple[np.ndarray, int]:
        Nr, Nt = H.shape
        flops = 0
        self.total_bp_calls = 0

        # QR decomposition
        Q, R = np.linalg.qr(H, mode='reduced')
        flops += 2 * Nr * Nt * Nt
        y_tilde = Q.conj().T @ y
        flops += 8 * Nt * Nr

        self._R = R
        self._y_tilde = y_tilde
        self._noise_var = noise_var

        # Init search tree
        self.search_tree = SearchTreeGraph()
        root = self.search_tree.create_root(layer=Nt)
        root.m_up = 0.0
        root.m_down = 0.0
        root.score = 0.0

        counter = 0
        pq: list = []

        # Expand root → children at bottom layer
        k0 = Nt - 1
        for sym in self.constellation:
            residual = y_tilde[k0] - R[k0, k0] * sym
            ld = float(np.abs(residual) ** 2)
            flops += 11
            self.search_tree.add_child(
                parent=root, layer=k0, symbol=sym,
                local_dist=ld, cum_dist=ld,
                partial_symbols=np.array([sym]),
            )

        # Full BP cycle
        flops += self._full_bp_cycle(prog_down, prog_up, prog_belief, prog_halt)

        # Build PQ from frontier
        for node in self.search_tree.nodes:
            if not node.is_expanded and node is not root:
                heapq.heappush(pq, (node.score, node.queue_version,
                                    counter, node))
                counter += 1

        best_node = None
        best_score = float('inf')

        while pq and len(self.search_tree.nodes) < self.max_nodes + 1:
            sc, ver, _, node = heapq.heappop(pq)
            if ver != node.queue_version:
                continue

            if sc < best_score:
                best_score = sc
                best_node = node

            if node.layer == 0:
                return node.partial_symbols[::-1], flops

            self.search_tree.mark_expanded(node)
            next_layer = node.layer - 1

            # Create ALL M children
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

                self.search_tree.add_child(
                    parent=node, layer=next_layer, symbol=sym,
                    local_dist=ld, cum_dist=cd,
                    partial_symbols=new_partial,
                )

            # Full BP cycle on entire tree
            flops += self._full_bp_cycle(prog_down, prog_up, prog_belief,
                                         prog_halt)

            # Rebuild PQ from ALL frontier nodes
            pq = []
            counter = 0
            for nd in self.search_tree.nodes:
                if not nd.is_expanded and nd is not root:
                    heapq.heappush(pq, (nd.score, nd.queue_version,
                                        counter, nd))
                    counter += 1

        # Fallback
        if best_node is None:
            best_node = root
        x_hat, comp_flops = self._complete_path(best_node, R, y_tilde)
        flops += comp_flops
        return x_hat, flops

    # ------------------------------------------------------------------
    def _full_bp_cycle(self, prog_down, prog_up, prog_belief, prog_halt) -> int:
        total_flops = 0

        for bp_iter in range(self.max_bp_iters):
            old_root_m_up = self.search_tree.root.m_up if self.search_tree.root else 0.0

            # 1. Full UP-sweep: leaves → root
            total_flops += self._full_up_sweep(prog_up)

            # 2. Full DOWN-sweep: root → leaves
            total_flops += self._full_down_sweep(prog_down)

            # 3. Score ALL frontier nodes
            total_flops += self._score_all_frontier(prog_belief)

            # 4. Check halt
            if bp_iter < self.max_bp_iters - 1 and self.search_tree.root:
                should_halt = self._run_halt(self.search_tree.root,
                                             old_root_m_up, prog_halt)
                total_flops += self.vm.flops_count
                if should_halt:
                    break

        return total_flops

    # ------------------------------------------------------------------
    def _bfs_order(self) -> List[TreeNode]:
        if not self.search_tree.root:
            return []
        order = []
        q = deque([self.search_tree.root])
        while q:
            node = q.popleft()
            order.append(node)
            for c in node.children:
                q.append(c)
        return order

    # ------------------------------------------------------------------
    def _full_up_sweep(self, prog_up) -> int:
        total_flops = 0
        bfs_order = self._bfs_order()
        for node in reversed(bfs_order):
            if node.children:
                total_flops += self._run_f_up(node, prog_up)
                self.total_bp_calls += 1
            else:
                # Leaf: m_up = local_dist (provides non-zero values for
                # F_up to aggregate; 0 creates a bootstrap trap where
                # all m_up values stay at 0 forever)
                node.m_up = node.local_dist
        return total_flops

    # ------------------------------------------------------------------
    def _full_down_sweep(self, prog_down) -> int:
        total_flops = 0
        bfs_order = self._bfs_order()
        for node in bfs_order:
            if node.parent is not None:
                total_flops += self._run_down_pass(node, prog_down)
                self.total_bp_calls += 1
        return total_flops

    # ------------------------------------------------------------------
    def _score_all_frontier(self, prog_belief) -> int:
        total_flops = 0
        for node in self.search_tree.nodes:
            if not node.is_expanded and node is not self.search_tree.root:
                total_flops += self._run_belief(node, prog_belief)
        return total_flops

    # ------------------------------------------------------------------
    # F_down: M_parent_down + C_i → M_i_down
    # ------------------------------------------------------------------
    def _run_down_pass(self, child: TreeNode, prog_down: List[Instruction]) -> int:
        parent = child.parent
        if parent is None:
            return 0

        self.vm.reset()
        self.vm.candidate_node = child
        self.vm.constellation = self.constellation
        self.vm.noise_var = self._noise_var

        self.vm.matrix_stack.push(self._R.copy())
        self.vm.vector_stack.push(self._y_tilde.copy())
        self.vm.vector_stack.push(child.partial_symbols.copy())
        self.vm.graph_stack.push(self.search_tree)
        self.vm.node_stack.push(child)
        self.vm.int_stack.push(child.layer)

        # Inputs: M_parent_down (bottom), C_i=local_dist (top)
        self.vm.float_stack.push(float(parent.m_down))
        self.vm.float_stack.push(float(child.local_dist))

        try:
            self.vm._execute_block(prog_down)
        except Exception:
            pass

        result = self.vm.float_stack.peek()
        if result is not None and np.isfinite(result):
            child.m_down = float(result)
        else:
            child.m_down = parent.m_down + child.local_dist

        return self.vm.flops_count

    # ------------------------------------------------------------------
    # F_up: {M_j_up, C_j} → M_i_up
    # ------------------------------------------------------------------
    def _run_f_up(self, node: TreeNode, prog_up: List[Instruction]) -> int:
        self.vm.reset()
        self.vm.candidate_node = node
        self.vm.constellation = self.constellation
        self.vm.noise_var = self._noise_var

        self.vm.matrix_stack.push(self._R.copy())
        self.vm.vector_stack.push(self._y_tilde.copy())
        self.vm.vector_stack.push(node.partial_symbols.copy())
        self.vm.graph_stack.push(self.search_tree)
        self.vm.node_stack.push(node)
        self.vm.int_stack.push(node.layer)

        # Push children's (C_j, M_j_up) pairs
        for child in node.children:
            self.vm.float_stack.push(float(child.local_dist))
            self.vm.float_stack.push(float(child.m_up))

        self.vm.int_stack.push(len(node.children))

        try:
            self.vm._execute_block(prog_up)
        except Exception:
            pass

        result = self.vm.float_stack.peek()
        if result is not None and np.isfinite(result):
            node.m_up = float(result)

        return self.vm.flops_count

    # ------------------------------------------------------------------
    # F_belief: (cum_dist, M_down, M_up) → score
    # ------------------------------------------------------------------
    def _run_belief(self, node: TreeNode, prog_belief: List[Instruction]) -> int:
        self.vm.reset()
        self.vm.candidate_node = node
        self.vm.constellation = self.constellation
        self.vm.noise_var = self._noise_var

        self.vm.matrix_stack.push(self._R.copy())
        self.vm.vector_stack.push(self._y_tilde.copy())
        self.vm.vector_stack.push(node.partial_symbols.copy())
        self.vm.graph_stack.push(self.search_tree)
        self.vm.node_stack.push(node)
        self.vm.int_stack.push(node.layer)

        # Inputs: cum_dist (bottom), M_down, M_up (top)
        self.vm.float_stack.push(float(node.cum_dist))
        self.vm.float_stack.push(float(node.m_down))
        self.vm.float_stack.push(float(node.m_up))

        try:
            self.vm._execute_block(prog_belief)
        except Exception:
            pass

        result = self.vm.float_stack.peek()
        if result is not None and np.isfinite(result):
            node.score = float(result)
            node.queue_version += 1
        else:
            node.score = node.cum_dist

        return self.vm.flops_count

    # ------------------------------------------------------------------
    # H_halt
    # ------------------------------------------------------------------
    def _run_halt(self, node: TreeNode, old_m_up: float,
                  prog_halt: List[Instruction]) -> bool:
        self.vm.reset()
        self.vm.candidate_node = node
        self.vm.constellation = self.constellation
        self.vm.noise_var = self._noise_var

        self.vm.graph_stack.push(self.search_tree)
        self.vm.node_stack.push(node)
        self.vm.int_stack.push(node.layer)

        self.vm.float_stack.push(float(old_m_up))
        self.vm.float_stack.push(float(node.m_up))

        try:
            self.vm._execute_block(prog_halt)
        except Exception:
            return True

        result = self.vm.bool_stack.peek()
        if result is not None:
            return bool(result)
        return True

    # ------------------------------------------------------------------
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
                flops += 2
                ld = float(np.abs(residual) ** 2)
                flops += 3
                if ld < best_ld:
                    best_ld = ld
                    best_sym = sym
            decided.append(best_sym)
            cur_layer -= 1
        x_hat = np.array(decided[::-1])
        return x_hat, flops
