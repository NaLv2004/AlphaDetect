"""
MIMO-Push Virtual Machine.
A typed stack-based VM with FLOPs counting and hardware fault interrupts.
Executes evolved programs to compute emergent_metric scores.
"""
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from stacks import TypedStack, TreeNode, SearchTreeGraph

# ============================================================
# FLOPs Fault Exception
# ============================================================
class HardwareFault(Exception):
    """Raised when FLOPs budget is exceeded."""
    pass


class StepLimitFault(Exception):
    """Raised when instruction step limit is exceeded."""
    pass


# ============================================================
# Instruction representation
# ============================================================
class Instruction:
    """A single instruction in a Push program."""
    def __init__(self, name: str, code_block: Optional[List['Instruction']] = None,
                 code_block2: Optional[List['Instruction']] = None):
        self.name = name
        # For Exec.If: code_block = true branch, code_block2 = false branch
        # For Exec.While / Graph.MapReduce: code_block = body
        self.code_block = code_block or []
        self.code_block2 = code_block2 or []

    def __repr__(self):
        if self.name == 'Exec.If':
            return f"IF({_fmt_block(self.code_block)}, {_fmt_block(self.code_block2)})"
        elif self.name == 'Exec.While':
            return f"WHILE({_fmt_block(self.code_block)})"
        elif self.name == 'Graph.MapReduce':
            return f"MAP_REDUCE({_fmt_block(self.code_block)})"
        return self.name

    def human_readable(self, indent=0) -> str:
        """Return a human-readable multi-line representation."""
        prefix = "  " * indent
        if self.name == 'Exec.If':
            lines = [f"{prefix}IF (Bool.top):"]
            lines.append(f"{prefix}  THEN:")
            for ins in self.code_block:
                lines.append(ins.human_readable(indent + 2))
            lines.append(f"{prefix}  ELSE:")
            for ins in self.code_block2:
                lines.append(ins.human_readable(indent + 2))
            return "\n".join(lines)
        elif self.name == 'Exec.While':
            lines = [f"{prefix}WHILE (Bool.top):"]
            for ins in self.code_block:
                lines.append(ins.human_readable(indent + 1))
            return "\n".join(lines)
        elif self.name == 'Graph.MapReduce':
            lines = [f"{prefix}MAP_REDUCE over children:"]
            for ins in self.code_block:
                lines.append(ins.human_readable(indent + 1))
            return "\n".join(lines)
        return f"{prefix}{self.name}"


def _fmt_block(block):
    return "[" + ", ".join(str(i) for i in block) + "]"


def program_to_string(program: List[Instruction]) -> str:
    """Convert a program to human-readable string."""
    lines = []
    for ins in program:
        lines.append(ins.human_readable(0))
    return "\n".join(lines)


def program_to_oneliner(program: List[Instruction]) -> str:
    """Compact one-line representation."""
    return " ; ".join(str(ins) for ins in program)


# ============================================================
# MIMO-Push VM
# ============================================================

# All primitive instruction names (flat, no control flow)
PRIMITIVE_INSTRUCTIONS = [
    # Stack manipulation
    'Float.Pop', 'Float.Dup', 'Float.Swap',
    'Int.Pop', 'Int.Dup', 'Int.Swap',
    'Bool.Pop', 'Bool.Dup', 'Bool.Swap',
    'Vector.Pop', 'Vector.Dup', 'Vector.Swap',
    'Matrix.Pop', 'Matrix.Dup', 'Matrix.Swap',
    'Node.Pop', 'Node.Dup', 'Node.Swap',
    'Graph.Pop',
    # Scalar arithmetic
    'Float.Add', 'Float.Sub', 'Float.Mul', 'Float.Div',
    'Float.Abs', 'Float.Neg', 'Float.Sqrt', 'Float.Square',
    'Float.Min', 'Float.Max',
    # Comparisons -> Bool
    'Float.LT', 'Float.GT', 'Float.EQ',
    'Int.LT', 'Int.GT', 'Int.EQ',
    # Int arithmetic
    'Int.Add', 'Int.Sub', 'Int.Mul',
    'Int.Inc', 'Int.Dec',
    # Bool logic
    'Bool.And', 'Bool.Or', 'Bool.Not',
    # Tensor algebra
    'Mat.Mul', 'Mat.VecMul', 'Mat.Add', 'Mat.Sub', 'Mat.Transpose',
    'Mat.InvertSafe',
    'Mat.ExtractTopLeft',
    'Vec.Add', 'Vec.Sub', 'Vec.Dot', 'Vec.Norm2',
    'Vec.Scale',  # Float * Vec
    'Vec.ExtractSlice',
    'Vec.Len',  # push length as Int
    # Graph / Node operations
    'Graph.GetRoot',
    'Graph.GetNodeCount', 'Graph.GetOpenCount', 'Graph.GetLastExpanded', 'Graph.OpenAt',
    'Node.GetParent', 'Node.GetChildren',  # pushes children count to Int
    'Node.GetData',  # push node data vector to Vector, distance to Float
    'Node.GetLayer',  # push layer to Int
    'Node.GetDistance',  # push cumulative distance to Float
    'Node.GetScore', 'Node.GetVisitCount', 'Node.GetExpansionCount',
    'Node.GetSubtreeSize', 'Node.GetOpenDescendants', 'Node.GetCompleteDescendants',
    'Node.GetSiblingCount',
    'Node.GetState0', 'Node.GetState1', 'Node.GetState2', 'Node.GetState3',
    'Node.GetState4', 'Node.GetState5', 'Node.GetState6', 'Node.GetState7',
    'Node.ChildAt',  # pop Int index, push child node
    # Type conversion / constants
    'Float.FromInt',
    'Int.FromFloat',
    'Float.Const0', 'Float.Const1', 'Float.Const2', 'Float.ConstHalf',
    'Float.ConstInf',
    'Int.Const0', 'Int.Const1',
    'Bool.True', 'Bool.False',
]

# Control flow instructions (have code blocks)
CONTROL_INSTRUCTIONS = [
    'Exec.If', 'Exec.While', 'Graph.MapReduce',
    'Graph.MapReduceOpen', 'Graph.MapReduceSiblings', 'Graph.MapReduceAncestors',
]

ALL_INSTRUCTIONS = PRIMITIVE_INSTRUCTIONS + CONTROL_INSTRUCTIONS


class MIMOPushVM:
    """
    The MIMO-Push Virtual Machine.
    Executes typed stack-based programs with FLOPs budget enforcement.
    """

    def __init__(self, flops_max: int = 50000, step_max: int = 2000):
        self.flops_max = flops_max
        self.step_max = step_max
        self.reset()

    def reset(self):
        """Reset all stacks and counters."""
        self.float_stack = TypedStack(max_depth=100)
        self.int_stack = TypedStack(max_depth=100)
        self.bool_stack = TypedStack(max_depth=100)
        self.vector_stack = TypedStack(max_depth=50)
        self.matrix_stack = TypedStack(max_depth=20)
        self.graph_stack = TypedStack(max_depth=5)
        self.node_stack = TypedStack(max_depth=100)
        self.flops_count = 0
        self.step_count = 0

    def _charge_flops(self, cost: int):
        """Charge FLOPs and raise HardwareFault if exceeded."""
        self.flops_count += cost
        if self.flops_count > self.flops_max:
            raise HardwareFault(f"FLOPs exceeded: {self.flops_count} > {self.flops_max}")

    def _charge_step(self):
        """Charge one instruction step."""
        self.step_count += 1
        if self.step_count > self.step_max:
            raise StepLimitFault(f"Step limit exceeded: {self.step_count}")

    def inject_environment(self, R: np.ndarray, y_tilde: np.ndarray,
                           x_partial: np.ndarray, graph: SearchTreeGraph,
                           candidate_node: TreeNode, depth_k: int):
        """
        Inject environment snapshot into stacks before running program.
        Follows the spec: R -> Matrix top, y_tilde and x_partial -> Vector,
        graph -> Graph, candidate_node -> Node, k -> Int.
        """
        self.reset()
        # Matrix stack: R
        self.matrix_stack.push(R.copy())
        # Vector stack: y_tilde (bottom), x_partial (top)
        self.vector_stack.push(y_tilde.copy())
        self.vector_stack.push(x_partial.copy())
        # Graph stack: search tree
        self.graph_stack.push(graph)
        # Node stack: candidate node
        self.node_stack.push(candidate_node)
        # Int stack: depth k
        self.int_stack.push(depth_k)

    def run(self, program: List[Instruction]) -> float:
        """
        Execute a program and return the Float stack top as the score.
        Returns inf on fault or empty Float stack.
        """
        try:
            self._execute_block(program)
        except (HardwareFault, StepLimitFault):
            return float('inf')
        except Exception:
            return float('inf')

        result = self.float_stack.peek()
        if result is None:
            return float('inf')
        if np.isnan(result) or np.isinf(result):
            return float('inf')
        return float(result)

    def _execute_block(self, block: List[Instruction]):
        """Execute a list of instructions sequentially."""
        for ins in block:
            self._execute_instruction(ins)

    def _execute_instruction(self, ins: Instruction):
        """Execute a single instruction."""
        self._charge_step()
        name = ins.name

        # ---- Stack manipulation ----
        if name == 'Float.Pop': self.float_stack.pop()
        elif name == 'Float.Dup': self.float_stack.dup()
        elif name == 'Float.Swap': self.float_stack.swap()
        elif name == 'Int.Pop': self.int_stack.pop()
        elif name == 'Int.Dup': self.int_stack.dup()
        elif name == 'Int.Swap': self.int_stack.swap()
        elif name == 'Bool.Pop': self.bool_stack.pop()
        elif name == 'Bool.Dup': self.bool_stack.dup()
        elif name == 'Bool.Swap': self.bool_stack.swap()
        elif name == 'Vector.Pop': self.vector_stack.pop()
        elif name == 'Vector.Dup': self.vector_stack.dup()
        elif name == 'Vector.Swap': self.vector_stack.swap()
        elif name == 'Matrix.Pop': self.matrix_stack.pop()
        elif name == 'Matrix.Dup': self.matrix_stack.dup()
        elif name == 'Matrix.Swap': self.matrix_stack.swap()
        elif name == 'Node.Pop': self.node_stack.pop()
        elif name == 'Node.Dup': self.node_stack.dup()
        elif name == 'Node.Swap': self.node_stack.swap()
        elif name == 'Graph.Pop': self.graph_stack.pop()

        # ---- Float arithmetic ----
        elif name == 'Float.Add': self._float_binop(lambda a, b: a + b, 1)
        elif name == 'Float.Sub': self._float_binop(lambda a, b: b - a, 1)
        elif name == 'Float.Mul': self._float_binop(lambda a, b: a * b, 1)
        elif name == 'Float.Div':
            a = self.float_stack.pop()
            b = self.float_stack.pop()
            if a is not None and b is not None:
                self._charge_flops(1)
                if abs(a) < 1e-30:
                    self.float_stack.push(0.0)
                else:
                    self.float_stack.push(b / a)
        elif name == 'Float.Abs':
            a = self.float_stack.pop()
            if a is not None:
                self._charge_flops(1)
                self.float_stack.push(abs(a))
        elif name == 'Float.Neg':
            a = self.float_stack.pop()
            if a is not None:
                self.float_stack.push(-a)
        elif name == 'Float.Sqrt':
            a = self.float_stack.pop()
            if a is not None:
                self._charge_flops(1)
                self.float_stack.push(np.sqrt(max(a, 0.0)))
        elif name == 'Float.Square':
            a = self.float_stack.pop()
            if a is not None:
                self._charge_flops(1)
                self.float_stack.push(a * a)
        elif name == 'Float.Min':
            self._float_binop(lambda a, b: min(a, b), 1)
        elif name == 'Float.Max':
            self._float_binop(lambda a, b: max(a, b), 1)

        # ---- Comparisons ----
        elif name == 'Float.LT':
            a = self.float_stack.pop()
            b = self.float_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(b < a)
        elif name == 'Float.GT':
            a = self.float_stack.pop()
            b = self.float_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(b > a)
        elif name == 'Float.EQ':
            a = self.float_stack.pop()
            b = self.float_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(abs(a - b) < 1e-10)
        elif name == 'Int.LT':
            a = self.int_stack.pop()
            b = self.int_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(b < a)
        elif name == 'Int.GT':
            a = self.int_stack.pop()
            b = self.int_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(b > a)
        elif name == 'Int.EQ':
            a = self.int_stack.pop()
            b = self.int_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(a == b)

        # ---- Int arithmetic ----
        elif name == 'Int.Add':
            a = self.int_stack.pop()
            b = self.int_stack.pop()
            if a is not None and b is not None:
                self.int_stack.push(a + b)
        elif name == 'Int.Sub':
            a = self.int_stack.pop()
            b = self.int_stack.pop()
            if a is not None and b is not None:
                self.int_stack.push(b - a)
        elif name == 'Int.Mul':
            a = self.int_stack.pop()
            b = self.int_stack.pop()
            if a is not None and b is not None:
                self.int_stack.push(a * b)
        elif name == 'Int.Inc':
            a = self.int_stack.pop()
            if a is not None:
                self.int_stack.push(a + 1)
        elif name == 'Int.Dec':
            a = self.int_stack.pop()
            if a is not None:
                self.int_stack.push(a - 1)

        # ---- Bool logic ----
        elif name == 'Bool.And':
            a = self.bool_stack.pop()
            b = self.bool_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(a and b)
        elif name == 'Bool.Or':
            a = self.bool_stack.pop()
            b = self.bool_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(a or b)
        elif name == 'Bool.Not':
            a = self.bool_stack.pop()
            if a is not None:
                self.bool_stack.push(not a)

        # ---- Tensor algebra ----
        elif name == 'Mat.Mul':
            A = self.matrix_stack.pop()
            B = self.matrix_stack.pop()
            if A is not None and B is not None:
                if A.shape[1] == B.shape[0]:
                    n = A.shape[0]
                    self._charge_flops(n * n * A.shape[1])
                    self.matrix_stack.push(B @ A)
                # else: type mismatch, silently ignore (PushGP convention)

        elif name == 'Mat.VecMul':
            v = self.vector_stack.pop()
            M = self.matrix_stack.pop()
            if v is not None and M is not None:
                if M.shape[1] == v.shape[0]:
                    self._charge_flops(M.shape[0] * M.shape[1])
                    self.vector_stack.push(M @ v)

        elif name == 'Mat.Add':
            A = self.matrix_stack.pop()
            B = self.matrix_stack.pop()
            if A is not None and B is not None and A.shape == B.shape:
                self._charge_flops(A.size)
                self.matrix_stack.push(A + B)

        elif name == 'Mat.Sub':
            A = self.matrix_stack.pop()
            B = self.matrix_stack.pop()
            if A is not None and B is not None and A.shape == B.shape:
                self._charge_flops(A.size)
                self.matrix_stack.push(B - A)

        elif name == 'Mat.Transpose':
            M = self.matrix_stack.pop()
            if M is not None:
                self.matrix_stack.push(M.T.copy())

        elif name == 'Mat.InvertSafe':
            M = self.matrix_stack.pop()
            if M is not None:
                n = M.shape[0]
                self._charge_flops(n * n * n)
                try:
                    if M.shape[0] == M.shape[1]:
                        inv = np.linalg.inv(M)
                        if np.any(np.isnan(inv)) or np.any(np.isinf(inv)):
                            self.matrix_stack.push(np.eye(n))
                        else:
                            self.matrix_stack.push(inv)
                    else:
                        # pseudo-inverse
                        inv = np.linalg.pinv(M)
                        self.matrix_stack.push(inv)
                except np.linalg.LinAlgError:
                    self.matrix_stack.push(np.eye(n))

        elif name == 'Mat.ExtractTopLeft':
            k = self.int_stack.pop()
            M = self.matrix_stack.pop()
            if k is not None and M is not None:
                k = max(1, min(k, min(M.shape)))
                self.matrix_stack.push(M[:k, :k].copy())

        elif name == 'Vec.Add':
            a = self.vector_stack.pop()
            b = self.vector_stack.pop()
            if a is not None and b is not None and a.shape == b.shape:
                self._charge_flops(a.size)
                self.vector_stack.push(a + b)

        elif name == 'Vec.Sub':
            a = self.vector_stack.pop()
            b = self.vector_stack.pop()
            if a is not None and b is not None and a.shape == b.shape:
                self._charge_flops(a.size)
                self.vector_stack.push(b - a)

        elif name == 'Vec.Dot':
            a = self.vector_stack.pop()
            b = self.vector_stack.pop()
            if a is not None and b is not None and a.shape == b.shape:
                self._charge_flops(2 * a.size)
                self.float_stack.push(float(np.real(np.dot(a.conj(), b))))

        elif name == 'Vec.Norm2':
            a = self.vector_stack.pop()
            if a is not None:
                self._charge_flops(2 * a.size)
                self.float_stack.push(float(np.real(np.dot(a.conj(), a))))

        elif name == 'Vec.Scale':
            v = self.vector_stack.pop()
            s = self.float_stack.pop()
            if v is not None and s is not None:
                self._charge_flops(v.size)
                self.vector_stack.push(s * v)

        elif name == 'Vec.ExtractSlice':
            k = self.int_stack.pop()
            v = self.vector_stack.pop()
            if k is not None and v is not None:
                k = max(0, min(k, v.size))
                self.vector_stack.push(v[:k].copy())

        elif name == 'Vec.Len':
            v = self.vector_stack.peek()
            if v is not None:
                self.int_stack.push(v.size)

        # ---- Graph / Node operations ----
        elif name == 'Graph.GetRoot':
            g = self.graph_stack.peek()
            if g is not None:
                root = g.get_root()
                if root is not None:
                    self.node_stack.push(root)

        elif name == 'Graph.GetNodeCount':
            g = self.graph_stack.peek()
            if g is not None:
                self.int_stack.push(g.node_count())

        elif name == 'Graph.GetOpenCount':
            g = self.graph_stack.peek()
            if g is not None:
                self.int_stack.push(g.open_node_count())

        elif name == 'Graph.GetLastExpanded':
            g = self.graph_stack.peek()
            if g is not None:
                node = g.get_last_expanded()
                if node is not None:
                    self.node_stack.push(node)

        elif name == 'Graph.OpenAt':
            idx = self.int_stack.pop()
            g = self.graph_stack.peek()
            if idx is not None and g is not None:
                node = g.open_node_at(idx)
                if node is not None:
                    self.node_stack.push(node)

        elif name == 'Node.GetParent':
            n = self.node_stack.pop()
            if n is not None and n.parent is not None:
                self.node_stack.push(n.parent)

        elif name == 'Node.GetChildren':
            n = self.node_stack.peek()
            if n is not None:
                self.int_stack.push(len(n.children))

        elif name == 'Node.GetData':
            n = self.node_stack.peek()
            if n is not None:
                dv = n.get_data_vector()
                self.vector_stack.push(dv)
                self.float_stack.push(n.cumulative_distance)

        elif name == 'Node.GetLayer':
            n = self.node_stack.peek()
            if n is not None:
                self.int_stack.push(n.layer)

        elif name == 'Node.GetDistance':
            n = self.node_stack.peek()
            if n is not None:
                self.float_stack.push(n.cumulative_distance)

        elif name == 'Node.GetScore':
            n = self.node_stack.peek()
            if n is not None:
                score = n.dynamic_score if np.isfinite(n.dynamic_score) else n.score
                self.float_stack.push(float(score))

        elif name == 'Node.GetVisitCount':
            n = self.node_stack.peek()
            if n is not None:
                self.int_stack.push(n.visit_count)

        elif name == 'Node.GetExpansionCount':
            n = self.node_stack.peek()
            if n is not None:
                self.int_stack.push(n.expansion_count)

        elif name == 'Node.GetSubtreeSize':
            n = self.node_stack.peek()
            if n is not None:
                self.int_stack.push(n.subtree_size)

        elif name == 'Node.GetOpenDescendants':
            n = self.node_stack.peek()
            if n is not None:
                self.int_stack.push(n.open_descendants)

        elif name == 'Node.GetCompleteDescendants':
            n = self.node_stack.peek()
            if n is not None:
                self.int_stack.push(n.complete_descendants)

        elif name == 'Node.GetSiblingCount':
            n = self.node_stack.peek()
            if n is not None and n.parent is not None:
                self.int_stack.push(max(0, len(n.parent.children) - 1))

        elif name.startswith('Node.GetState'):
            index = int(name.replace('Node.GetState', ''))
            n = self.node_stack.peek()
            if n is not None and 0 <= index < n.state_slots.size:
                self.float_stack.push(float(n.state_slots[index]))

        elif name == 'Node.ChildAt':
            idx = self.int_stack.pop()
            n = self.node_stack.peek()
            if idx is not None and n is not None and n.children:
                idx = idx % len(n.children)
                self.node_stack.push(n.children[idx])

        elif name == 'Node.SiblingAt':
            idx = self.int_stack.pop()
            n = self.node_stack.peek()
            if idx is not None and n is not None and n.parent is not None:
                siblings = [child for child in n.parent.children if child.node_id != n.node_id]
                if siblings:
                    self.node_stack.push(siblings[idx % len(siblings)])

        # ---- Type conversion / constants ----
        elif name == 'Float.FromInt':
            a = self.int_stack.pop()
            if a is not None:
                self.float_stack.push(float(a))
        elif name == 'Int.FromFloat':
            a = self.float_stack.pop()
            if a is not None:
                self.int_stack.push(int(a))
        elif name == 'Float.Const0': self.float_stack.push(0.0)
        elif name == 'Float.Const1': self.float_stack.push(1.0)
        elif name == 'Float.Const2': self.float_stack.push(2.0)
        elif name == 'Float.ConstHalf': self.float_stack.push(0.5)
        elif name == 'Float.ConstInf': self.float_stack.push(float('inf'))
        elif name == 'Int.Const0': self.int_stack.push(0)
        elif name == 'Int.Const1': self.int_stack.push(1)
        elif name == 'Bool.True': self.bool_stack.push(True)
        elif name == 'Bool.False': self.bool_stack.push(False)

        # ---- Control flow ----
        elif name == 'Exec.If':
            cond = self.bool_stack.pop()
            if cond is not None:
                if cond:
                    self._execute_block(ins.code_block)
                else:
                    self._execute_block(ins.code_block2)

        elif name == 'Exec.While':
            max_iterations = 50  # prevent infinite loops
            iteration = 0
            while iteration < max_iterations:
                cond = self.bool_stack.pop()
                if cond is None or not cond:
                    break
                self._execute_block(ins.code_block)
                iteration += 1

        elif name == 'Graph.MapReduce':
            node = self.node_stack.peek()
            if node is not None and node.children:
                self._execute_mapreduce(node.children, ins.code_block)

        elif name == 'Graph.MapReduceOpen':
            graph = self.graph_stack.peek()
            if graph is not None:
                self._execute_mapreduce(graph.frontier_nodes(), ins.code_block)

        elif name == 'Graph.MapReduceSiblings':
            node = self.node_stack.peek()
            if node is not None and node.parent is not None:
                siblings = [child for child in node.parent.children if child.node_id != node.node_id]
                self._execute_mapreduce(siblings, ins.code_block)

        elif name == 'Graph.MapReduceAncestors':
            graph = self.graph_stack.peek()
            node = self.node_stack.peek()
            if graph is not None and node is not None:
                ancestors = graph.ancestor_chain(node)[1:]
                self._execute_mapreduce(ancestors, ins.code_block)

    def _float_binop(self, op, flops_cost):
        a = self.float_stack.pop()
        b = self.float_stack.pop()
        if a is not None and b is not None:
            self._charge_flops(flops_cost)
            result = op(a, b)
            if np.isnan(result) or np.isinf(result):
                self.float_stack.push(0.0)
            else:
                self.float_stack.push(float(result))

    def _execute_mapreduce(self, nodes, block: List[Instruction]):
        accumulator = 0.0
        nodes_to_visit = list(nodes)[:20]
        for mapped_node in nodes_to_visit:
            original_depth = self.node_stack.depth()
            self.node_stack.push(mapped_node)
            self._execute_block(block)
            result = self.float_stack.pop()
            if result is not None and np.isfinite(result):
                accumulator += float(result)
            while self.node_stack.depth() > original_depth:
                self.node_stack.pop()
        self.float_stack.push(accumulator)
