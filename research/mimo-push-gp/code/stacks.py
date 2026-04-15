"""
Typed Stack Data Structures for MIMO-Push VM v2.
Minimal node memory model — no pre-computed statistics.
All aggregate/relational information must be computed by the evolved program.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

N_MEM = 16  # writable memory slots per node (used as BP message channels)


class TypedStack:
    """Generic typed stack with max depth."""

    def __init__(self, max_depth: int = 200):
        self._items: list = []
        self.max_depth = max_depth

    def push(self, item):
        if len(self._items) < self.max_depth:
            self._items.append(item)

    def pop(self):
        return self._items.pop() if self._items else None

    def peek(self):
        return self._items[-1] if self._items else None

    def dup(self):
        if self._items:
            top = self._items[-1]
            if isinstance(top, np.ndarray):
                self.push(top.copy())
            else:
                self.push(top)

    def swap(self):
        if len(self._items) >= 2:
            self._items[-1], self._items[-2] = self._items[-2], self._items[-1]

    def rot(self):
        """Forth-style ROT: move 3rd item to top. (a b c) -> (b c a)"""
        if len(self._items) >= 3:
            self._items[-3], self._items[-2], self._items[-1] = \
                self._items[-2], self._items[-1], self._items[-3]

    def depth(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return not self._items


@dataclass
class TreeNode:
    """Minimal search tree node with writable memory bank.
    
    Physical data (layer, symbol, distances) accessible via dedicated ops.
    Memory slots mem[0..N_MEM-1] initialized to 0, freely read/written by programs.
    No pre-computed statistics — the evolved program must derive everything.
    """
    node_id: int
    layer: int
    symbol: complex = 0.0
    local_dist: float = 0.0
    cum_dist: float = 0.0
    partial_symbols: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    mem: np.ndarray = field(default_factory=lambda: np.zeros(N_MEM))
    is_expanded: bool = False
    score: float = float('inf')
    queue_version: int = 0
    m_up: float = 0.0      # upward memory (BP message to parent)
    m_down: float = 0.0    # downward memory (context from parent)

    def __hash__(self):
        # Hash by node_id for set membership (needed for dirty_nodes set)
        return hash(self.node_id)

    def __eq__(self, other):
        if isinstance(other, TreeNode):
            return self.node_id == other.node_id
        return NotImplemented


class SearchTreeGraph:
    """Dynamic search tree built during stack decoding."""

    def __init__(self):
        self.nodes: List[TreeNode] = []
        self.root: Optional[TreeNode] = None
        self._next_id = 0
        self.dirty_nodes: set = set()  # nodes whose scores were modified by program

    def create_root(self, layer: int) -> TreeNode:
        node = TreeNode(node_id=self._next_id, layer=layer)
        self._next_id += 1
        self.nodes.append(node)
        self.root = node
        node.is_expanded = True
        return node

    def add_child(self, parent: TreeNode, layer: int, symbol: complex,
                  local_dist: float, cum_dist: float,
                  partial_symbols: np.ndarray) -> TreeNode:
        node = TreeNode(
            node_id=self._next_id, layer=layer, symbol=symbol,
            local_dist=local_dist, cum_dist=cum_dist,
            partial_symbols=partial_symbols.copy(), parent=parent,
        )
        self._next_id += 1
        parent.children.append(node)
        self.nodes.append(node)
        return node

    def mark_expanded(self, node: TreeNode):
        node.is_expanded = True

    def frontier_nodes(self) -> List[TreeNode]:
        return [n for n in self.nodes if not n.is_expanded]

    def node_count(self) -> int:
        return len(self.nodes)

    def frontier_count(self) -> int:
        return sum(1 for n in self.nodes if not n.is_expanded)

    def siblings(self, node: TreeNode) -> List[TreeNode]:
        if node.parent is None:
            return []
        return [c for c in node.parent.children if c.node_id != node.node_id]

    def ancestors(self, node: TreeNode) -> List[TreeNode]:
        chain = []
        cur = node.parent
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        return chain
