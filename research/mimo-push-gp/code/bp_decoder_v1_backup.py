"""
BP-Enabled Stack Decoder for Automated Algorithm Discovery.

Architecture:
  The outer loop is a standard Best-First Search (stack decoder) that builds
  a search tree incrementally.  After every `bp_interval` node expansions the
  framework runs a **Belief-Propagation phase** on the current tree.

BP Phase — the computation-graph pattern provided by the framework:
  1. BOTTOM-UP sweep  (leaves → root):
       For every node n in ascending layer order, run the evolved Push
       program with n as the candidate.  The program can
         - read n's children via  Node.ForEachChild / Node.ChildAt
         - read/write any node's memory  (Node.ReadMem / Node.WriteMem)
         - read any node's current score (Node.GetScore)
         - SET any node's score          (Node.SetScore)   ← KEY for BP
       After the program returns, n.score is also updated to
       cum_dist + program_output (unless the program used SetScore directly).

  2. TOP-DOWN sweep (root → leaves):
       Same as above but in descending layer order.  This lets parent
       information flow downward.

  3. Repeat for `bp_sweeps` full bottom-up + top-down pairs.

  4. Rebuild the priority queue with the updated scores.

What the GP discovers:
  - The message-update formula  (what to write into mem[])
  - The belief-computation rule (how to combine cum_dist, local_dist,
    and neighbor messages into a score)
  - Convergence patterns emerge implicitly from the fixed sweep count
    and the evolved update logic.

What the framework hard-codes (the "BP pattern"):
  - The sweep schedule  (bottom-up then top-down)
  - The adjacency structure  (parent ↔ child edges of the search tree)
  - The message storage locations  (node.mem[0..15])
  - When BP runs  (every bp_interval expansions)

Node budgets:  hundreds to thousands (default 1000).
"""

import numpy as np
import heapq
from collections import defaultdict
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
    """Best-First Stack Decoder with Belief-Propagation sweeps.

    The scoring function is an evolved Push program.  After every
    `bp_interval` expansions the decoder performs BP sweeps on the
    full tree, allowing later discoveries to update earlier nodes'
    scores via message passing through node memory slots.
    """

    def __init__(self, Nt: int, Nr: int, constellation: np.ndarray,
                 max_nodes: int = 1000,
                 vm: Optional[MIMOPushVM] = None,
                 bp_interval: int = 32,
                 bp_sweeps: int = 2,
                 bp_step_limit: int = 300,
                 bp_flops_limit: int = 50000,
                 max_bp_nodes: int = 200):
        self.Nt = Nt
        self.Nr = Nr
        self.constellation = constellation
        self.M = len(constellation)
        self.max_nodes = max_nodes
        self.vm = vm or MIMOPushVM()
        self.bp_interval = bp_interval
        self.bp_sweeps = bp_sweeps
        self.bp_step_limit = bp_step_limit
        self.bp_flops_limit = bp_flops_limit
        self.max_bp_nodes = max_bp_nodes  # max nodes swept during each BP phase
        self.search_tree: Optional[SearchTreeGraph] = None

    # ------------------------------------------------------------------
    # Main detect
    # ------------------------------------------------------------------

    def detect(self, H: np.ndarray, y: np.ndarray,
               program: List[Instruction],
               noise_var: float = 1.0) -> Tuple[np.ndarray, int]:
        """Run BP-enabled stack decoder.  Returns (x_hat, total_flops)."""
        Nr, Nt = H.shape
        flops = 0

        # QR decomposition
        Q, R = np.linalg.qr(H, mode='reduced')
        flops += 2 * Nr * Nt * Nt
        y_tilde = Q.conj().T @ y
        flops += 8 * Nt * Nr

        self._noise_var = noise_var

        # Initialise search tree
        self.search_tree = SearchTreeGraph()
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
            child.score = score
            flops += self.vm.flops_count
            heapq.heappush(pq, (score, child.queue_version, counter, child))
            counter += 1

        best_node = None
        best_score = float('inf')
        nodes_expanded = 0
        expansions_since_bp = 0

        while pq and nodes_expanded < self.max_nodes:
            sc, ver, _, node = heapq.heappop(pq)
            if ver != node.queue_version:
                continue  # stale entry
            nodes_expanded += 1

            if sc < best_score:
                best_score = sc
                best_node = node

            # Complete path?
            if node.layer == 0:
                return node.partial_symbols[::-1], flops

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

                child_node = self.search_tree.add_child(
                    parent=node, layer=next_layer, symbol=sym,
                    local_dist=ld, cum_dist=cd,
                    partial_symbols=new_partial,
                )
                score = self._score_node(child_node, R, y_tilde, program)
                child_node.score = score
                flops += self.vm.flops_count
                heapq.heappush(pq, (score, child_node.queue_version, counter, child_node))
                counter += 1

            expansions_since_bp += 1

            # ============ BP PHASE ============
            if expansions_since_bp >= self.bp_interval:
                bp_flops = self._run_bp(R, y_tilde, program)
                flops += bp_flops
                expansions_since_bp = 0

                # Rebuild priority queue with updated scores
                counter = self._rebuild_pq(pq, counter)

                # Update best after BP
                for nd in self.search_tree.nodes:
                    if not nd.is_expanded and nd.score < best_score:
                        best_score = nd.score
                        best_node = nd

        # Budget exhausted — run final BP sweep if not just done
        if expansions_since_bp > 0 and len(self.search_tree.nodes) > 1:
            bp_flops = self._run_bp(R, y_tilde, program)
            flops += bp_flops
            # Update best_node/best_score after final BP
            for nd in self.search_tree.nodes:
                if not nd.is_expanded and nd.score < best_score:
                    best_score = nd.score
                    best_node = nd

        # Complete best partial path greedily
        if best_node is None:
            best_node = root
        x_hat, comp_flops = self._complete_path(best_node, R, y_tilde)
        flops += comp_flops
        return x_hat, flops

    # ------------------------------------------------------------------
    # Scoring (runs Push program, returns score)
    # ------------------------------------------------------------------

    def _score_node(self, node: TreeNode, R: np.ndarray,
                    y_tilde: np.ndarray, program: List[Instruction]) -> float:
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
    # BP sweep
    # ------------------------------------------------------------------

    def _run_bp(self, R: np.ndarray, y_tilde: np.ndarray,
                program: List[Instruction]) -> int:
        """Run bp_sweeps rounds of bottom-up + top-down BP on the tree.

        Only sweeps the top max_bp_nodes nodes (by score) for efficiency.
        The selection includes both expanded nodes (parents that aggregate
        child info upward) and frontier nodes (whose scores drive the search).

        Returns total FLOPs consumed by BP.
        """
        total_flops = 0

        # Select nodes to participate in BP:
        # - All expanded nodes (they have children → aggregate messages)
        # - Top-scoring frontier nodes (their scores drive exploration)
        all_nodes = [nd for nd in self.search_tree.nodes
                     if nd is not self.search_tree.root]

        if len(all_nodes) > self.max_bp_nodes:
            # Prioritize: expanded nodes first, then best frontier nodes
            expanded = [nd for nd in all_nodes if nd.is_expanded]
            frontier = [nd for nd in all_nodes if not nd.is_expanded]
            frontier.sort(key=lambda n: n.score)
            remaining = self.max_bp_nodes - len(expanded)
            if remaining > 0:
                bp_nodes = expanded + frontier[:remaining]
            else:
                bp_nodes = expanded[:self.max_bp_nodes]
        else:
            bp_nodes = all_nodes

        max_bp_flops = self.bp_flops_limit * len(bp_nodes) * 2

        # Group selected nodes by layer
        by_layer: dict = defaultdict(list)
        for nd in bp_nodes:
            by_layer[nd.layer].append(nd)

        layers_bottom_up = sorted(by_layer.keys())
        layers_top_down = sorted(by_layer.keys(), reverse=True)

        for _sweep in range(self.bp_sweeps):
            for layer in layers_bottom_up:
                for nd in by_layer[layer]:
                    total_flops += self._bp_update_node(nd, R, y_tilde, program)
                    if total_flops >= max_bp_flops:
                        return total_flops

            for layer in layers_top_down:
                for nd in by_layer[layer]:
                    total_flops += self._bp_update_node(nd, R, y_tilde, program)
                    if total_flops >= max_bp_flops:
                        return total_flops

        return total_flops

    def _bp_update_node(self, node: TreeNode, R: np.ndarray,
                        y_tilde: np.ndarray,
                        program: List[Instruction]) -> int:
        """Run the Push program on a node during a BP sweep.

        Uses reduced step/flops budget compared to initial scoring,
        because BP updates should be lightweight message operations.
        """
        old_score = node.score

        # Save and apply reduced budgets for BP
        orig_step_max = self.vm.step_max
        orig_flops_max = self.vm.flops_max
        self.vm.step_max = self.bp_step_limit
        self.vm.flops_max = self.bp_flops_limit

        self.vm.inject_environment(
            R=R, y_tilde=y_tilde,
            x_partial=node.partial_symbols,
            graph=self.search_tree, candidate_node=node,
            depth_k=node.layer,
            constellation=self.constellation,
            noise_var=self._noise_var,
        )
        correction = self.vm.run(program)
        fl = self.vm.flops_count

        # Restore original budgets
        self.vm.step_max = orig_step_max
        self.vm.flops_max = orig_flops_max

        # If the program used Node.SetScore, the score was already updated
        # inside the VM.  Otherwise, use the program output as correction.
        if node.score == old_score:
            if correction != float('inf') and np.isfinite(correction):
                node.score = node.cum_dist + correction

        return fl

    # ------------------------------------------------------------------
    # Priority queue rebuild after BP
    # ------------------------------------------------------------------

    def _rebuild_pq(self, pq: list, counter: int) -> int:
        """Replace the priority queue with current scores of all frontier nodes."""
        pq.clear()
        for nd in self.search_tree.nodes:
            if not nd.is_expanded and nd.layer > 0:
                # Only frontier (unexpanded, non-complete) nodes go in PQ
                nd.queue_version += 1
                heapq.heappush(pq, (nd.score, nd.queue_version, counter, nd))
                counter += 1
            elif nd.layer == 0 and not nd.is_expanded:
                # Complete paths (layer 0) also go in PQ
                nd.queue_version += 1
                heapq.heappush(pq, (nd.score, nd.queue_version, counter, nd))
                counter += 1
        return counter

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
