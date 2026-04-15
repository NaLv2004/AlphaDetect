"""
Program-Controlled Stack Decoder for Automated Algorithm Discovery.

Architecture:
  The outer loop is a standard Best-First Search (stack decoder) that builds
  a search tree incrementally.  The inner scoring function is an evolved Push
  program that runs on EVERY new node.

Program-Controlled BP:
  The evolved program has full access to the search tree graph and can:
    1. Read any node's data (score, memory, layer, symbol, distances)
    2. Write any node's memory via Node.WriteMem
    3. SET any node's score via Node.SetScore — THIS is how BP happens
    4. Traverse the tree via Node.GetParent, Node.ChildAt,
       Node.ForEachChild, Node.ForEachSibling, Node.ForEachAncestor

  When the program calls Node.SetScore on ANY node, that node is marked
  dirty and its priority-queue entry is automatically updated.  This means
  the program can propagate information to ancestors, siblings, or descendants
  during its execution — implementing BP, message passing, or any other
  correction scheme it discovers.

  The framework provides NO hardcoded BP sweep schedule.  The program runs
  once per new node.  How much tree traversal it does (and therefore how much
  BP it performs) is determined entirely by its own control flow and the
  step/flops budget.

Complexity Control:
  The VM has hard step_max and flops_max limits.  A program that tries to
  traverse the entire tree will be truncated.  Evolution naturally selects
  programs that extract maximum information within the budget — this is how
  the program learns to control BP complexity.
"""

import numpy as np
import heapq
from typing import List, Optional, Tuple

from stacks import TreeNode, SearchTreeGraph
from vm import MIMOPushVM, Instruction


# --------------------------------------------------------------------------
# Constellations  (duplicated here for standalone use)
# --------------------------------------------------------------------------

def qam16_constellation() -> np.ndarray:
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    return np.array([r + 1j * i for r in levels for i in levels])


def qpsk_constellation() -> np.ndarray:
    s = 1.0 / np.sqrt(2)
    return np.array([s + 1j * s, s - 1j * s, -s + 1j * s, -s - 1j * s])


# --------------------------------------------------------------------------
# BP-Enabled Stack Decoder
# --------------------------------------------------------------------------

class BPStackDecoder:
    """Best-First Stack Decoder where BP is fully program-controlled.

    The scoring function is an evolved Push program that runs on every
    new node and can traverse the tree to update other nodes' scores.
    No hardcoded BP sweep schedule — the program IS the BP algorithm.
    """

    def __init__(self, Nt: int, Nr: int, constellation: np.ndarray,
                 max_nodes: int = 1000,
                 vm: Optional[MIMOPushVM] = None,
                 allow_score_writes: bool = True):
        self.Nt = Nt
        self.Nr = Nr
        self.constellation = constellation
        self.M = len(constellation)
        self.max_nodes = max_nodes
        self.vm = vm or MIMOPushVM()
        self.allow_score_writes = allow_score_writes
        self.search_tree: Optional[SearchTreeGraph] = None
        self.bp_updates: int = 0  # count of dirty-node updates per detect()
        self.nonlocal_bp_updates: int = 0  # non-candidate score writes per detect()

    # ------------------------------------------------------------------
    # Main detect
    # ------------------------------------------------------------------

    def detect(self, H: np.ndarray, y: np.ndarray,
               program: List[Instruction],
               noise_var: float = 1.0) -> Tuple[np.ndarray, int]:
        """Run stack decoder with program-controlled scoring/BP."""
        Nr, Nt = H.shape
        flops = 0
        self.vm.allow_score_writes = self.allow_score_writes

        # QR decomposition
        Q, R = np.linalg.qr(H, mode='reduced')
        flops += 2 * Nr * Nt * Nt
        y_tilde = Q.conj().T @ y
        flops += 8 * Nt * Nr

        self._noise_var = noise_var

        # Initialise search tree
        self.search_tree = SearchTreeGraph()
        self.bp_updates = 0
        self.nonlocal_bp_updates = 0
        root = self.search_tree.create_root(layer=Nt)

        counter = 0
        pq: list = []

        # Expand root → children at bottom layer
        k0 = Nt - 1
        for sym in self.constellation:
            residual = y_tilde[k0] - R[k0, k0] * sym
            ld = float(np.abs(residual) ** 2)
            flops += 11
            child = self.search_tree.add_child(
                parent=root, layer=k0, symbol=sym,
                local_dist=ld, cum_dist=ld,
                partial_symbols=np.array([sym]),
            )
            score = self._score_node(child, R, y_tilde, program)
            self.nonlocal_bp_updates += self.vm.nonlocal_score_write_count
            child.score = score
            flops += self.vm.flops_count
            heapq.heappush(pq, (score, child.queue_version, counter, child))
            counter += 1

        # Process any dirty nodes from initial scoring (program may SetScore)
        counter = self._process_dirty(pq, counter)

        best_node = None
        best_score = float('inf')

        while pq and len(self.search_tree.nodes) < self.max_nodes + 1:
            sc, ver, _, node = heapq.heappop(pq)
            if ver != node.queue_version:
                continue  # stale entry

            if sc < best_score:
                best_score = sc
                best_node = node

            # Complete path?
            if node.layer == 0:
                return node.partial_symbols[::-1], flops

            self.search_tree.mark_expanded(node)
            next_layer = node.layer - 1

            # Expand children — program runs on EACH new child
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

                child_node = self.search_tree.add_child(
                    parent=node, layer=next_layer, symbol=sym,
                    local_dist=ld, cum_dist=cd,
                    partial_symbols=new_partial,
                )
                score = self._score_node(child_node, R, y_tilde, program)
                self.nonlocal_bp_updates += self.vm.nonlocal_score_write_count
                child_node.score = score
                flops += self.vm.flops_count
                heapq.heappush(pq, (score, child_node.queue_version,
                                    counter, child_node))
                counter += 1

            # Process nodes the program modified via SetScore (BP)
            n_dirty_before = self.bp_updates
            counter = self._process_dirty(pq, counter)

            # --- BP propagation pass ---
            # Run program once more on the PARENT to let it propagate
            # info upward (parent sees its children now exist).
            # This gives the program a chance to aggregate child
            # information and update ancestor scores.
            if node.children:
                self._bp_propagate(node, R, y_tilde, program)
                self.nonlocal_bp_updates += self.vm.nonlocal_score_write_count
                flops += self.vm.flops_count
                counter = self._process_dirty(pq, counter)

        # Complete best partial path greedily
        if best_node is None:
            best_node = root
        x_hat, comp_flops = self._complete_path(best_node, R, y_tilde)
        flops += comp_flops
        return x_hat, flops

    # ------------------------------------------------------------------
    # Scoring (runs Push program on a node)
    # ------------------------------------------------------------------

    def _score_node(self, node: TreeNode, R: np.ndarray,
                    y_tilde: np.ndarray, program: List[Instruction]) -> float:
        """Run the evolved program on a node.

        The program can:
        - Return a correction value (score = cum_dist + correction)
        - Traverse the tree and call SetScore on other nodes (BP)
        - Read/write memory on any visited node (message passing)
        """
        self.vm.inject_environment(
            R=R, y_tilde=y_tilde,
            x_partial=node.partial_symbols,
            graph=self.search_tree, candidate_node=node,
            depth_k=node.layer,
            constellation=self.constellation,
            noise_var=self._noise_var,
        )
        correction = self.vm.run(program)
        if correction == float('inf') or not np.isfinite(correction):
            return node.cum_dist
        return node.cum_dist + correction

    # ------------------------------------------------------------------
    # Dirty node processing (update PQ after program-driven BP)
    # ------------------------------------------------------------------

    def _process_dirty(self, pq: list, counter: int) -> int:
        """Update priority queue for nodes whose scores were modified.

        When the program calls Node.SetScore during tree traversal,
        the affected nodes are marked dirty.  We re-insert them into
        the PQ with their new scores (using queue_version for lazy deletion).
        """
        if not self.search_tree.dirty_nodes:
            return counter

        for nd in self.search_tree.dirty_nodes:
            self.bp_updates += 1
            if not nd.is_expanded:  # only frontier nodes need PQ update
                nd.queue_version += 1
                heapq.heappush(pq, (nd.score, nd.queue_version,
                                    counter, nd))
                counter += 1
        self.search_tree.dirty_nodes.clear()
        return counter

    # ------------------------------------------------------------------
    # BP propagation (run program on parent after expansion)
    # ------------------------------------------------------------------

    def _bp_propagate(self, parent_node: TreeNode, R: np.ndarray,
                      y_tilde: np.ndarray, program: List[Instruction]):
        """Run the program on the parent node after its children are created.

        This gives the program an opportunity to:
        - Iterate over the parent's children (ForEachChild now works!)
        - Aggregate child scores/distances
        - SetScore on the parent's siblings, ancestors, or other frontier nodes
        - Propagate information upward through the tree

        The parent's own score doesn't matter (it's already expanded and removed
        from PQ), but the program can update FRONTIER nodes via SetScore.
        """
        self.vm.inject_environment(
            R=R, y_tilde=y_tilde,
            x_partial=parent_node.partial_symbols,
            graph=self.search_tree, candidate_node=parent_node,
            depth_k=parent_node.layer,
            constellation=self.constellation,
            noise_var=self._noise_var,
        )
        self.vm.run(program)
        # Don't use the return value — we only care about side effects
        # (SetScore calls that update other nodes)

    # ------------------------------------------------------------------
    # Greedy completion (fallback)
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
