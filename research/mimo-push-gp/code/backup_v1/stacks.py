"""
Typed Stack Data Structures for MIMO-Push VM.
Each stack is a LIFO structure holding elements of a specific type.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional


class TypedStack:
    """Generic typed stack with max depth to prevent memory explosion."""
    def __init__(self, max_depth: int = 200):
        self._items: list = []
        self.max_depth = max_depth

    def push(self, item):
        if len(self._items) < self.max_depth:
            self._items.append(item)

    def pop(self):
        if self._items:
            return self._items.pop()
        return None

    def peek(self):
        if self._items:
            return self._items[-1]
        return None

    def dup(self):
        """Duplicate top element."""
        if self._items:
            top = self._items[-1]
            if isinstance(top, np.ndarray):
                self.push(top.copy())
            elif isinstance(top, SearchTreeGraph):
                self.push(top)  # graphs are shared references
            elif isinstance(top, TreeNode):
                self.push(top)  # nodes are shared references
            else:
                self.push(top)

    def swap(self):
        """Swap top two elements."""
        if len(self._items) >= 2:
            self._items[-1], self._items[-2] = self._items[-2], self._items[-1]

    def flush(self):
        """Clear the stack."""
        self._items.clear()

    def depth(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def __repr__(self):
        return f"Stack(depth={self.depth()})"


@dataclass
class TreeNode:
    """Node in the dynamic search tree."""
    node_id: int
    layer: int  # depth k in the search tree (Nt down to 1)
    symbol: complex = 0.0  # decided symbol at this layer
    cumulative_distance: float = 0.0  # accumulated Euclidean distance
    partial_symbols: np.ndarray = field(default_factory=lambda: np.array([]))
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    score: float = float('inf')  # compatibility alias for dynamic_score
    intrinsic_score: float = float('inf')
    dynamic_score: float = float('inf')
    queue_version: int = 0
    visit_count: int = 0
    expansion_count: int = 0
    subtree_size: int = 1
    open_descendants: int = 0
    complete_descendants: int = 0
    best_complete_distance: float = float('inf')
    best_open_score: float = float('inf')
    innovation_signal: float = 0.0
    innovation_energy: float = 0.0
    update_count: int = 0
    state_slots: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=float))

    def get_data_vector(self) -> np.ndarray:
        """Return node data as a float vector for VM consumption."""
        return np.array([
            float(self.layer),
            float(np.real(self.symbol)),
            float(np.imag(self.symbol)),
            float(self.cumulative_distance),
            float(self.dynamic_score if np.isfinite(self.dynamic_score) else self.score),
            float(self.visit_count),
            float(self.expansion_count),
            float(self.subtree_size),
            float(self.open_descendants),
            float(self.complete_descendants),
            float(self.best_complete_distance if np.isfinite(self.best_complete_distance) else self.cumulative_distance),
            float(self.best_open_score if np.isfinite(self.best_open_score) else self.cumulative_distance),
            float(self.innovation_signal),
            float(self.innovation_energy),
            float(self.update_count),
            *self.state_slots.tolist(),
        ])


class SearchTreeGraph:
    """Dynamic search tree used by the Stack Decoder."""
    def __init__(self):
        self.nodes: List[TreeNode] = []
        self.root: Optional[TreeNode] = None
        self._next_id = 0
        self._open_node_ids = set()
        self._closed_node_ids = set()
        self._last_expanded: Optional[TreeNode] = None

    def create_root(self, layer: int) -> TreeNode:
        node = TreeNode(node_id=self._next_id, layer=layer)
        self._next_id += 1
        self.nodes.append(node)
        self.root = node
        self._closed_node_ids.add(node.node_id)
        return node

    def add_child(self, parent: TreeNode, layer: int, symbol: complex,
                  cumulative_distance: float, partial_symbols: np.ndarray) -> TreeNode:
        node = TreeNode(
            node_id=self._next_id,
            layer=layer,
            symbol=symbol,
            cumulative_distance=cumulative_distance,
            partial_symbols=partial_symbols.copy(),
            parent=parent
        )
        self._next_id += 1
        parent.children.append(node)
        self.nodes.append(node)
        self._open_node_ids.add(node.node_id)
        return node

    def get_root(self) -> Optional[TreeNode]:
        return self.root

    def get_last_expanded(self) -> Optional[TreeNode]:
        return self._last_expanded

    def node_count(self) -> int:
        return len(self.nodes)

    def frontier_nodes(self) -> List[TreeNode]:
        return [node for node in self.nodes if node.node_id in self._open_node_ids]

    def open_node_count(self) -> int:
        return len(self._open_node_ids)

    def open_node_at(self, index: int) -> Optional[TreeNode]:
        frontier = self.frontier_nodes()
        if not frontier:
            return None
        return frontier[index % len(frontier)]

    def siblings(self, node: TreeNode) -> List[TreeNode]:
        if node.parent is None:
            return []
        return [child for child in node.parent.children if child.node_id != node.node_id]

    def ancestor_chain(self, node: TreeNode) -> List[TreeNode]:
        chain = [node]
        current = node.parent
        while current is not None:
            chain.append(current)
            current = current.parent
        return chain

    def mark_expanded(self, node: TreeNode):
        self._open_node_ids.discard(node.node_id)
        self._closed_node_ids.add(node.node_id)
        node.expansion_count += 1
        self._last_expanded = node

    def propagate_rescore_delta(self, node: TreeNode, delta: float):
        current = node
        attenuation = 1.0
        while current is not None:
            current.innovation_signal = 0.7 * current.innovation_signal + attenuation * float(delta)
            current.innovation_energy = 0.7 * current.innovation_energy + attenuation * float(delta * delta)
            current.update_count += 1
            attenuation *= 0.5
            current = current.parent

    def refresh_all_statistics(self):
        if self.root is None:
            return
        self._refresh_recursive(self.root)

    def _refresh_recursive(self, node: TreeNode):
        for child in node.children:
            self._refresh_recursive(child)

        subtree_size = 1
        open_descendants = 1 if node.node_id in self._open_node_ids else 0
        complete_descendants = 1 if node.layer == 0 else 0
        best_complete_distance = node.cumulative_distance if node.layer == 0 else float('inf')
        best_open_score = float('inf')
        child_distances = []
        child_scores = []

        for child in node.children:
            subtree_size += child.subtree_size
            open_descendants += child.open_descendants
            complete_descendants += child.complete_descendants
            best_complete_distance = min(best_complete_distance, child.best_complete_distance)
            best_open_score = min(best_open_score, child.best_open_score)
            child_distances.append(child.cumulative_distance)
            if np.isfinite(child.dynamic_score):
                child_scores.append(child.dynamic_score)

        if node.node_id in self._open_node_ids:
            node_score = node.dynamic_score if np.isfinite(node.dynamic_score) else node.cumulative_distance
            best_open_score = min(best_open_score, node_score)

        if not np.isfinite(best_complete_distance):
            best_complete_distance = node.cumulative_distance
        if not np.isfinite(best_open_score):
            best_open_score = node.dynamic_score if np.isfinite(node.dynamic_score) else node.cumulative_distance

        child_distance_mean = float(np.mean(child_distances)) if child_distances else float(node.cumulative_distance)
        child_distance_var = float(np.var(child_distances)) if len(child_distances) > 1 else 0.0

        node.subtree_size = subtree_size
        node.open_descendants = open_descendants
        node.complete_descendants = complete_descendants
        node.best_complete_distance = best_complete_distance
        node.best_open_score = best_open_score
        node.state_slots = np.array([
            best_complete_distance,
            best_open_score,
            float(open_descendants),
            float(complete_descendants),
            float(subtree_size),
            child_distance_mean,
            float(node.innovation_signal + child_distance_var),
            float(node.update_count),
        ], dtype=float)
