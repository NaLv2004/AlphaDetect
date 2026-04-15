"""
MIMO-Push Virtual Machine v2.
Fine-grained typed stack VM — no coarse-grained domain oracles.
All pre-computed tree statistics removed; nodes expose only raw physical data
and a writable memory bank.  The evolved program must compose everything
from micro-level algebra, memory R/W, and local graph traversal.
"""
import numpy as np
from typing import List, Optional
from stacks import TypedStack, TreeNode, SearchTreeGraph, N_MEM


class HardwareFault(Exception):
    pass


class StepLimitFault(Exception):
    pass


class Instruction:
    def __init__(self, name: str,
                 code_block: Optional[List['Instruction']] = None,
                 code_block2: Optional[List['Instruction']] = None):
        self.name = name
        self.code_block = code_block or []
        self.code_block2 = code_block2 or []

    def __repr__(self):
        if self.name == 'Exec.If':
            return f"IF([{_fmt(self.code_block)}], [{_fmt(self.code_block2)}])"
        elif self.name in CONTROL_INSTRUCTIONS:
            return f"{self.name}([{_fmt(self.code_block)}])"
        return self.name

    def human_readable(self, indent=0) -> str:
        p = "  " * indent
        if self.name == 'Exec.If':
            lines = [f"{p}IF:"]
            lines += [f"{p}  THEN:"] + [i.human_readable(indent+2) for i in self.code_block]
            lines += [f"{p}  ELSE:"] + [i.human_readable(indent+2) for i in self.code_block2]
            return "\n".join(lines)
        if self.name in CONTROL_INSTRUCTIONS:
            lines = [f"{p}{self.name}:"]
            lines += [i.human_readable(indent+1) for i in self.code_block]
            return "\n".join(lines)
        return f"{p}{self.name}"


def _fmt(block):
    return ", ".join(str(i) for i in block)


def program_to_string(prog: List[Instruction]) -> str:
    return "\n".join(i.human_readable(0) for i in prog)


def program_to_oneliner(prog: List[Instruction]) -> str:
    return " ; ".join(str(i) for i in prog)


# ---------------------------------------------------------------------------
# Instruction catalogue
# ---------------------------------------------------------------------------

PRIMITIVE_INSTRUCTIONS = [
    # -- stack manipulation (16) --
    'Float.Pop', 'Float.Dup', 'Float.Swap', 'Float.Rot',
    'Int.Pop', 'Int.Dup', 'Int.Swap',
    'Bool.Pop', 'Bool.Dup',
    'Vector.Pop', 'Vector.Dup', 'Vector.Swap',
    'Matrix.Pop', 'Matrix.Dup',
    'Node.Pop', 'Node.Dup', 'Node.Swap',
    # -- float arithmetic (14) --
    'Float.Add', 'Float.Sub', 'Float.Mul', 'Float.Div',
    'Float.Abs', 'Float.Neg', 'Float.Inv', 'Float.Sqrt', 'Float.Square',
    'Float.Min', 'Float.Max', 'Float.Exp', 'Float.Log', 'Float.Tanh',
    # -- comparisons (4) --
    'Float.LT', 'Float.GT', 'Int.LT', 'Int.GT',
    # -- int arithmetic (4) --
    'Int.Add', 'Int.Sub', 'Int.Inc', 'Int.Dec',
    # -- bool logic (3) --
    'Bool.And', 'Bool.Or', 'Bool.Not',
    # -- tensor ops (11) --
    'Mat.VecMul',
    'Vec.Add', 'Vec.Sub', 'Vec.Dot', 'Vec.Norm2', 'Vec.Scale',
    'Vec.ElementAt', 'Mat.ElementAt', 'Mat.Row',
    'Vec.Len', 'Mat.Rows',
    # -- peek-based element access (preserve data) --
    'Mat.PeekAt', 'Vec.PeekAt',
    'Mat.PeekAtIm', 'Vec.PeekAtIm',
    # -- second-vector element access (for y_tilde under x_partial) --
    'Vec.SecondPeekAt',    # Re(y_tilde[idx]) — access second vector on stack
    'Vec.SecondPeekAtIm',  # Im(y_tilde[idx])
    # -- high-level decoding primitives (DISABLED: too much domain prior) --
    # 'Vec.GetResidue',    # REMOVED — embeds MIMO-specific interference cancellation
    # 'Float.GetMMSELB',   # REMOVED — embeds MMSE lower bound (O(k³) solve)
    # -- node memory (7) --
    'Node.ReadMem', 'Node.WriteMem',
    'Node.GetCumDist', 'Node.GetLocalDist',
    'Node.GetSymRe', 'Node.GetSymIm', 'Node.GetLayer',
    # -- graph navigation (6) --
    'Graph.GetRoot', 'Node.GetParent',
    'Node.NumChildren', 'Node.ChildAt',
    'Graph.NodeCount', 'Graph.FrontierCount',
    # -- BP-essential (3) --
    'Node.GetScore',       # push current score of node (allows reading updated beliefs)
    'Node.SetScore',       # pop float, set node's score (enables backward propagation)
    'Node.IsExpanded',     # push bool: has this node been expanded (has children)?
    # -- structured BP memory (4) --
    'Node.GetMUp',         # push m_up of node on node stack
    'Node.SetMUp',         # pop float, set m_up of node on node stack
    'Node.GetMDown',       # push m_down of node on node stack
    'Node.SetMDown',       # pop float, set m_down of node on node stack
    # -- constants (11) --
    'Float.Const0', 'Float.Const1', 'Float.ConstHalf',
    'Float.ConstNeg1', 'Float.Const2', 'Float.Const0_1',
    'Int.Const0', 'Int.Const1', 'Int.Const2',
    'Bool.True', 'Bool.False',
    # -- environment scalars (2) --
    'Float.GetNoiseVar',  # push noise variance (sigma^2 per real component)
    'Int.GetNumSymbols',  # push constellation size M
    # -- type conversion (2) --
    'Float.FromInt', 'Int.FromFloat',
    # -- evolvable constants (4) --
    'Float.EvoConst0', 'Float.EvoConst1', 'Float.EvoConst2', 'Float.EvoConst3',
]

CONTROL_INSTRUCTIONS = [
    'Exec.If',                # Bool → branch
    'Exec.While',             # while Bool.top (max 20 iter)
    'Exec.DoTimes',           # pop Int n, loop min(n,15) times
    'Node.ForEachChild',      # MapReduce over children → Float sum
    'Node.ForEachChildMin',   # MinReduce over children → Float min (A* heuristic)
    'Node.ForEachSibling',    # MapReduce over siblings → Float sum
    'Node.ForEachAncestor',   # MapReduce over ancestor chain → Float sum
    'Exec.ForEachSymbol',     # SumReduce over QAM constellation → Float sum
                              #   pushes Re(s), Im(s) before each block exec
    'Exec.MinOverSymbols',    # MinReduce over QAM constellation → Float min
                              #   pushes Re(s), Im(s) before each block exec
]

ALL_INSTRUCTIONS = PRIMITIVE_INSTRUCTIONS + CONTROL_INSTRUCTIONS


# ---------------------------------------------------------------------------
# VM
# ---------------------------------------------------------------------------

class MIMOPushVM:
    def __init__(self, flops_max: int = 50000, step_max: int = 2000):
        self.flops_max = flops_max
        self.step_max = step_max
        self.constellation: Optional[np.ndarray] = None
        self.noise_var: float = 1.0  # noise variance per real component
        self.allow_score_writes: bool = True
        self.candidate_node: Optional[TreeNode] = None
        self.reset()

    def reset(self):
        self.float_stack = TypedStack(100)
        self.int_stack = TypedStack(100)
        self.bool_stack = TypedStack(100)
        self.vector_stack = TypedStack(50)
        self.matrix_stack = TypedStack(20)
        self.graph_stack = TypedStack(5)
        self.node_stack = TypedStack(100)
        self.evolved_constants = np.zeros(4)  # set before running each genome
        self.flops_count = 0
        self.step_count = 0
        self.score_write_count = 0
        self.nonlocal_score_write_count = 0

    def _charge_flops(self, cost: int):
        self.flops_count += cost
        if self.flops_count > self.flops_max:
            raise HardwareFault()

    def _charge_step(self):
        self.step_count += 1
        if self.step_count > self.step_max:
            raise StepLimitFault()

    # ---- environment injection ----
    def inject_environment(self, R: np.ndarray, y_tilde: np.ndarray,
                           x_partial: np.ndarray, graph: SearchTreeGraph,
                           candidate_node: TreeNode, depth_k: int,
                           constellation: Optional[np.ndarray] = None,
                           noise_var: float = 1.0):
        self.reset()
        self.candidate_node = candidate_node
        self.matrix_stack.push(R.copy())
        self.vector_stack.push(y_tilde.copy())
        self.vector_stack.push(x_partial.copy())
        self.graph_stack.push(graph)
        self.node_stack.push(candidate_node)
        self.int_stack.push(depth_k)
        if constellation is not None:
            self.constellation = constellation  # set once, survives reset() within run
        self.noise_var = noise_var

    # ---- run ----
    def run(self, program: List[Instruction]) -> float:
        try:
            self._execute_block(program)
        except (HardwareFault, StepLimitFault):
            return float('inf')
        except Exception:
            return float('inf')
        result = self.float_stack.peek()
        if result is None:
            return float('inf')
        if not np.isfinite(result):
            return float('inf')
        return float(result)

    def _execute_block(self, block: List[Instruction]):
        for ins in block:
            self._execute_instruction(ins)

    # ---- helpers ----
    def _safe_float(self, v: float) -> float:
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return float(v)

    def _float_binop(self, op, cost: int):
        a = self.float_stack.pop()
        b = self.float_stack.pop()
        if a is not None and b is not None:
            self._charge_flops(cost)
            r = op(a, b)
            self.float_stack.push(self._safe_float(r))

    def _mapreduce(self, nodes: list, block: List[Instruction]):
        acc = 0.0
        for nd in nodes[:20]:
            orig_depth = self.node_stack.depth()
            self.node_stack.push(nd)
            self._execute_block(block)
            v = self.float_stack.pop()
            if v is not None and np.isfinite(v):
                acc += float(v)
            while self.node_stack.depth() > orig_depth:
                self.node_stack.pop()
        self.float_stack.push(acc)

    def _minreduce(self, nodes: list, block: List[Instruction]):
        """MinReduce over nodes: execute block for each node, take the minimum result.
        Enables A*-style lower-bound heuristics in F_up."""
        acc = float('inf')
        for nd in nodes[:20]:
            orig_depth = self.node_stack.depth()
            self.node_stack.push(nd)
            self._execute_block(block)
            v = self.float_stack.pop()
            if v is not None and np.isfinite(v):
                acc = min(acc, float(v))
            while self.node_stack.depth() > orig_depth:
                self.node_stack.pop()
        if not np.isfinite(acc):
            acc = 0.0
        self.float_stack.push(acc)

    def _constellation_mapreduce(self, constellation: np.ndarray,
                                  block: List[Instruction],
                                  mode: str = 'sum'):
        """Execute block for each constellation symbol.
        
        For each symbol s, creates an isolated float sub-stack pre-loaded with
        [Re(s), Im(s)] (Im on top). The block runs with access to ALL stacks
        but starts with a clean float stack having only [Re(s), Im(s)].
        
        mode='sum': Accumulates sum of float results
        mode='min': Returns the minimum float result (useful for A*-style heuristics)
        """
        acc = 0.0 if mode == 'sum' else float('inf')
        # Save outer float stack
        outer_float_vals = []
        while self.float_stack.depth() > 0:
            v = self.float_stack.pop()
            if v is not None:
                outer_float_vals.append(v)
        # outer_float_vals is in reverse order (last popped = was on bottom)
        
        for sym in constellation:
            # Give block a clean float environment with just (Re(s), Im(s))
            self.float_stack.push(float(np.real(sym)))
            self.float_stack.push(float(np.imag(sym)))
            self._execute_block(block)
            v = self.float_stack.pop()
            if v is not None and np.isfinite(v):
                if mode == 'sum':
                    acc += float(v)
                else:
                    acc = min(acc, float(v))
            # Clean any leftover floats from this iteration
            while self.float_stack.depth() > 0:
                self.float_stack.pop()
        
        # Restore outer float stack
        for v in reversed(outer_float_vals):
            self.float_stack.push(v)
        
        if mode == 'min' and acc == float('inf'):
            acc = 0.0
        self.float_stack.push(acc)

    # ---- instruction dispatch ----
    def _execute_instruction(self, ins: Instruction):
        self._charge_step()
        n = ins.name

        # ============ stack manipulation ============
        if   n == 'Float.Pop':   self.float_stack.pop()
        elif n == 'Float.Dup':   self.float_stack.dup()
        elif n == 'Float.Swap':  self.float_stack.swap()
        elif n == 'Float.Rot':   self.float_stack.rot()
        elif n == 'Int.Pop':     self.int_stack.pop()
        elif n == 'Int.Dup':     self.int_stack.dup()
        elif n == 'Int.Swap':    self.int_stack.swap()
        elif n == 'Bool.Pop':    self.bool_stack.pop()
        elif n == 'Bool.Dup':    self.bool_stack.dup()
        elif n == 'Vector.Pop':  self.vector_stack.pop()
        elif n == 'Vector.Dup':  self.vector_stack.dup()
        elif n == 'Vector.Swap': self.vector_stack.swap()
        elif n == 'Matrix.Pop':  self.matrix_stack.pop()
        elif n == 'Matrix.Dup':  self.matrix_stack.dup()
        elif n == 'Node.Pop':    self.node_stack.pop()
        elif n == 'Node.Dup':    self.node_stack.dup()
        elif n == 'Node.Swap':   self.node_stack.swap()

        # ============ float arithmetic ============
        elif n == 'Float.Add':    self._float_binop(lambda a, b: a + b, 1)
        elif n == 'Float.Sub':    self._float_binop(lambda a, b: b - a, 1)
        elif n == 'Float.Mul':    self._float_binop(lambda a, b: a * b, 1)
        elif n == 'Float.Div':
            a = self.float_stack.pop()
            b = self.float_stack.pop()
            if a is not None and b is not None:
                self._charge_flops(1)
                self.float_stack.push(self._safe_float(b / a if abs(a) > 1e-30 else 0.0))
        elif n == 'Float.Abs':
            a = self.float_stack.pop()
            if a is not None:
                self._charge_flops(1)
                self.float_stack.push(abs(a))
        elif n == 'Float.Neg':
            a = self.float_stack.pop()
            if a is not None:
                self.float_stack.push(-a)
        elif n == 'Float.Inv':
            a = self.float_stack.pop()
            if a is not None:
                self._charge_flops(1)
                self.float_stack.push(1.0 / a if a != 0.0 else float('inf'))
        elif n == 'Float.Sqrt':
            a = self.float_stack.pop()
            if a is not None:
                self._charge_flops(1)
                self.float_stack.push(float(np.sqrt(max(a, 0.0))))
        elif n == 'Float.Square':
            a = self.float_stack.pop()
            if a is not None:
                self._charge_flops(1)
                self.float_stack.push(a * a)
        elif n == 'Float.Min':
            self._float_binop(lambda a, b: min(a, b), 1)
        elif n == 'Float.Max':
            self._float_binop(lambda a, b: max(a, b), 1)
        elif n == 'Float.Exp':
            a = self.float_stack.pop()
            if a is not None:
                self._charge_flops(4)
                clamped = max(-20.0, min(a, 20.0))
                self.float_stack.push(float(np.exp(clamped)))
        elif n == 'Float.Log':
            a = self.float_stack.pop()
            if a is not None:
                self._charge_flops(4)
                self.float_stack.push(float(np.log(max(a, 1e-30))))
        elif n == 'Float.Tanh':
            a = self.float_stack.pop()
            if a is not None:
                self._charge_flops(4)
                self.float_stack.push(float(np.tanh(a)))

        # ============ comparisons ============
        elif n == 'Float.LT':
            a, b = self.float_stack.pop(), self.float_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(b < a)
        elif n == 'Float.GT':
            a, b = self.float_stack.pop(), self.float_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(b > a)
        elif n == 'Int.LT':
            a, b = self.int_stack.pop(), self.int_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(b < a)
        elif n == 'Int.GT':
            a, b = self.int_stack.pop(), self.int_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(b > a)

        # ============ int arithmetic ============
        elif n == 'Int.Add':
            a, b = self.int_stack.pop(), self.int_stack.pop()
            if a is not None and b is not None:
                self.int_stack.push(a + b)
        elif n == 'Int.Sub':
            a, b = self.int_stack.pop(), self.int_stack.pop()
            if a is not None and b is not None:
                self.int_stack.push(b - a)
        elif n == 'Int.Inc':
            a = self.int_stack.pop()
            if a is not None:
                self.int_stack.push(a + 1)
        elif n == 'Int.Dec':
            a = self.int_stack.pop()
            if a is not None:
                self.int_stack.push(a - 1)

        # ============ bool logic ============
        elif n == 'Bool.And':
            a, b = self.bool_stack.pop(), self.bool_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(a and b)
        elif n == 'Bool.Or':
            a, b = self.bool_stack.pop(), self.bool_stack.pop()
            if a is not None and b is not None:
                self.bool_stack.push(a or b)
        elif n == 'Bool.Not':
            a = self.bool_stack.pop()
            if a is not None:
                self.bool_stack.push(not a)

        # ============ tensor ops ============
        elif n == 'Mat.VecMul':
            v = self.vector_stack.pop()
            M = self.matrix_stack.pop()
            if v is not None and M is not None and M.shape[1] == v.shape[0]:
                self._charge_flops(M.shape[0] * M.shape[1])
                self.vector_stack.push(M @ v)

        elif n == 'Vec.Add':
            a, b = self.vector_stack.pop(), self.vector_stack.pop()
            if a is not None and b is not None and a.shape == b.shape:
                self._charge_flops(a.size)
                self.vector_stack.push(a + b)
        elif n == 'Vec.Sub':
            a, b = self.vector_stack.pop(), self.vector_stack.pop()
            if a is not None and b is not None and a.shape == b.shape:
                self._charge_flops(a.size)
                self.vector_stack.push(b - a)
        elif n == 'Vec.Dot':
            a, b = self.vector_stack.pop(), self.vector_stack.pop()
            if a is not None and b is not None and a.shape == b.shape:
                self._charge_flops(2 * a.size)
                self.float_stack.push(float(np.real(np.dot(a.conj(), b))))
        elif n == 'Vec.Norm2':
            a = self.vector_stack.pop()
            if a is not None:
                self._charge_flops(2 * a.size)
                self.float_stack.push(float(np.real(np.dot(a.conj(), a))))
        elif n == 'Vec.Scale':
            v = self.vector_stack.pop()
            s = self.float_stack.pop()
            if v is not None and s is not None:
                self._charge_flops(v.size)
                self.vector_stack.push(s * v)
        elif n == 'Vec.ElementAt':
            idx = self.int_stack.pop()
            v = self.vector_stack.pop()
            if idx is not None and v is not None and v.size > 0:
                idx = int(idx) % v.size
                val = v[idx]
                self.float_stack.push(float(np.real(val)))
        elif n == 'Mat.ElementAt':
            j = self.int_stack.pop()
            i = self.int_stack.pop()
            M = self.matrix_stack.pop()
            if i is not None and j is not None and M is not None:
                i = int(i) % M.shape[0]
                j = int(j) % M.shape[1]
                self.float_stack.push(float(np.real(M[i, j])))
        elif n == 'Mat.Row':
            i = self.int_stack.pop()
            M = self.matrix_stack.peek()
            if i is not None and M is not None:
                i = int(i) % M.shape[0]
                self.vector_stack.push(M[i, :].copy())
        elif n == 'Mat.PeekAt':
            # Access R[i,j] without consuming matrix or ints
            # Pop j, peek i, peek M, push Re(R[i,j]), push j back
            j = self.int_stack.pop()
            if j is not None:
                i = self.int_stack.peek()
                M = self.matrix_stack.peek()
                if i is not None and M is not None:
                    ii = int(i) % M.shape[0]
                    jj = int(j) % M.shape[1]
                    self.float_stack.push(float(np.real(M[ii, jj])))
                self.int_stack.push(j)  # restore j
        elif n == 'Mat.PeekAtIm':
            # Access Im(R[i,j]) without consuming matrix or ints
            j = self.int_stack.pop()
            if j is not None:
                i = self.int_stack.peek()
                M = self.matrix_stack.peek()
                if i is not None and M is not None:
                    ii = int(i) % M.shape[0]
                    jj = int(j) % M.shape[1]
                    self.float_stack.push(float(np.imag(M[ii, jj])))
                self.int_stack.push(j)  # restore j
        elif n == 'Vec.PeekAt':
            # Access Re(v[idx]) without consuming vector or int
            idx = self.int_stack.peek()
            if idx is not None:
                v = self.vector_stack.peek()
                if v is not None and v.size > 0:
                    ii = int(idx) % v.size
                    self.float_stack.push(float(np.real(v[ii])))
        elif n == 'Vec.PeekAtIm':
            # Access Im(v[idx]) without consuming vector or int
            idx = self.int_stack.peek()
            if idx is not None:
                v = self.vector_stack.peek()
                if v is not None and v.size > 0:
                    ii = int(idx) % v.size
                    self.float_stack.push(float(np.imag(v[ii])))
        elif n == 'Vec.SecondPeekAt':
            # Access Re(y_tilde[idx]) — y_tilde is second on vector stack (below x_partial)
            idx = self.int_stack.peek()
            if idx is not None and self.vector_stack.depth() >= 2:
                top = self.vector_stack.pop()
                v2 = self.vector_stack.peek()
                if v2 is not None and v2.size > 0:
                    ii = int(idx) % v2.size
                    self.float_stack.push(float(np.real(v2[ii])))
                if top is not None:
                    self.vector_stack.push(top)
        elif n == 'Vec.SecondPeekAtIm':
            # Access Im(y_tilde[idx]) — y_tilde is second on vector stack (below x_partial)
            idx = self.int_stack.peek()
            if idx is not None and self.vector_stack.depth() >= 2:
                top = self.vector_stack.pop()
                v2 = self.vector_stack.peek()
                if v2 is not None and v2.size > 0:
                    ii = int(idx) % v2.size
                    self.float_stack.push(float(np.imag(v2[ii])))
                if top is not None:
                    self.vector_stack.push(top)
        elif n == 'Vec.GetResidue':
            # Compute interference-cancelled y_tilde for unseen layers:
            #   residue = y_tilde[0:k] - R[0:k, k:Nt] @ x_partial_ordered
            # where k = top int (layer of current node)
            # ORDERING: x_partial is stored deepest-first (x_partial[0]=x[Nt-1], x_partial[-1]=x[k])
            # R[0:k, k:Nt] has columns in order x[k], x[k+1], ..., x[Nt-1]
            # So we need x_partial reversed: x[k], x[k+1], ..., x[Nt-1]
            k = self.int_stack.peek()
            if k is not None and self.vector_stack.depth() >= 2 and self.matrix_stack.depth() >= 1:
                k_int = int(k)
                x_partial = self.vector_stack.peek()  # top vector (deepest-first order)
                M = self.matrix_stack.peek()           # R matrix
                # y_tilde is second on vector stack
                top_vec = self.vector_stack.pop()
                y_tilde = self.vector_stack.peek()
                self.vector_stack.push(top_vec)
                if x_partial is not None and M is not None and y_tilde is not None:
                    Nt = M.shape[1]
                    if k_int > 0 and k_int <= Nt and x_partial.size == Nt - k_int:
                        # x_partial[0]=x[Nt-1], ..., x_partial[-1]=x[k]
                        # R[0:k, k:Nt] @ [x[k], x[k+1], ..., x[Nt-1]]
                        x_ordered = x_partial[::-1]  # reverse: now x_ordered[0]=x[k]
                        interference = M[:k_int, k_int:] @ x_ordered
                        residue = y_tilde[:k_int] - interference
                        self._charge_flops(k_int * (Nt - k_int) * 8)  # complex FMADs
                        self.vector_stack.push(residue.copy())
        elif n == 'Float.GetMMSELB':
            # MMSE lower bound on remaining layers' accumulated distance.
            # Computes: min_{x[0:k-1] in C^{k}} ||residue - R_sub @ x||^2
            # where residue = y_tilde[0:k] - R[0:k,k:] @ x_ordered  (interference cancelled)
            #       R_sub = R[0:k, 0:k]  (upper-triangular sub-matrix for remaining layers)
            # NOTE: x_partial is stored deepest-first, so we reverse it for the matrix multiply.
            k = self.int_stack.peek()
            if k is not None and self.vector_stack.depth() >= 2 and self.matrix_stack.depth() >= 1:
                k_int = int(k)
                x_partial = self.vector_stack.peek()
                M = self.matrix_stack.peek()
                top_vec = self.vector_stack.pop()
                y_tilde_v = self.vector_stack.peek()
                self.vector_stack.push(top_vec)
                if x_partial is not None and M is not None and y_tilde_v is not None:
                    Nt = M.shape[1]
                    nv = self.noise_var if hasattr(self, 'noise_var') else 1.0
                    if k_int > 0 and k_int <= Nt and x_partial.size == Nt - k_int:
                        try:
                            # Compute residue with correct ordering
                            x_ordered = x_partial[::-1]  # reverse to get x[k], x[k+1], ...
                            interference = M[:k_int, k_int:] @ x_ordered
                            residue = y_tilde_v[:k_int] - interference
                            R_sub = M[:k_int, :k_int]  # upper triangular sub-matrix
                            # MMSE LB = r^H r - r^H R_sub (R_sub^H R_sub + nv*I)^{-1} R_sub^H r
                            Gram = R_sub.conj().T @ R_sub + nv * np.eye(k_int)
                            Rh_r = R_sub.conj().T @ residue
                            t = np.linalg.solve(Gram, Rh_r)
                            mmse_lb = float(np.real(
                                residue.conj() @ residue - Rh_r.conj() @ t
                            ))
                            flops_mmse = k_int**3 + k_int**2 * (Nt - k_int) * 8
                            self._charge_flops(flops_mmse)
                            if mmse_lb < 0:
                                mmse_lb = 0.0
                            self.float_stack.push(mmse_lb)
                        except np.linalg.LinAlgError:
                            self.float_stack.push(0.0)
        elif n == 'Vec.Len':
            v = self.vector_stack.peek()
            if v is not None:
                self.int_stack.push(v.size)
        elif n == 'Mat.Rows':
            M = self.matrix_stack.peek()
            if M is not None:
                self.int_stack.push(M.shape[0])

        # ============ node memory ============
        elif n == 'Node.ReadMem':
            slot = self.int_stack.pop()
            nd = self.node_stack.peek()
            if slot is not None and nd is not None:
                slot = int(slot) % N_MEM
                self.float_stack.push(float(nd.mem[slot]))
        elif n == 'Node.WriteMem':
            val = self.float_stack.pop()
            slot = self.int_stack.pop()
            nd = self.node_stack.peek()
            if val is not None and slot is not None and nd is not None:
                slot = int(slot) % N_MEM
                nd.mem[slot] = self._safe_float(val)
        elif n == 'Node.GetCumDist':
            nd = self.node_stack.peek()
            if nd is not None:
                self.float_stack.push(float(nd.cum_dist))
        elif n == 'Node.GetLocalDist':
            nd = self.node_stack.peek()
            if nd is not None:
                self.float_stack.push(float(nd.local_dist))
        elif n == 'Node.GetSymRe':
            nd = self.node_stack.peek()
            if nd is not None:
                self.float_stack.push(float(np.real(nd.symbol)))
        elif n == 'Node.GetSymIm':
            nd = self.node_stack.peek()
            if nd is not None:
                self.float_stack.push(float(np.imag(nd.symbol)))
        elif n == 'Node.GetLayer':
            nd = self.node_stack.peek()
            if nd is not None:
                self.int_stack.push(nd.layer)

        # ============ graph navigation ============
        elif n == 'Graph.GetRoot':
            g = self.graph_stack.peek()
            if g is not None and g.root is not None:
                self.node_stack.push(g.root)
        elif n == 'Node.GetParent':
            nd = self.node_stack.pop()
            if nd is not None and nd.parent is not None:
                self.node_stack.push(nd.parent)
        elif n == 'Node.NumChildren':
            nd = self.node_stack.peek()
            if nd is not None:
                self.int_stack.push(len(nd.children))
        elif n == 'Node.ChildAt':
            idx = self.int_stack.pop()
            nd = self.node_stack.peek()
            if idx is not None and nd is not None and nd.children:
                idx = int(idx) % len(nd.children)
                self.node_stack.push(nd.children[idx])
        elif n == 'Graph.NodeCount':
            g = self.graph_stack.peek()
            if g is not None:
                self.int_stack.push(g.node_count())
        elif n == 'Graph.FrontierCount':
            g = self.graph_stack.peek()
            if g is not None:
                self.int_stack.push(g.frontier_count())

        # ============ BP-essential ============
        elif n == 'Node.GetScore':
            nd = self.node_stack.peek()
            if nd is not None:
                self.float_stack.push(float(nd.score) if np.isfinite(nd.score) else 0.0)
        elif n == 'Node.SetScore':
            val = self.float_stack.pop()
            nd = self.node_stack.peek()
            if (self.allow_score_writes and val is not None and nd is not None
                    and np.isfinite(val)):
                self.score_write_count += 1
                if nd is not self.candidate_node:
                    self.nonlocal_score_write_count += 1
                nd.score = float(val)
                nd.queue_version += 1  # invalidate old PQ entries
                # Track dirty nodes for PQ rebuild
                g = self.graph_stack.peek()
                if g is not None:
                    g.dirty_nodes.add(nd)
        elif n == 'Node.IsExpanded':
            nd = self.node_stack.peek()
            if nd is not None:
                self.bool_stack.push(nd.is_expanded)

        # ============ structured BP memory ============
        elif n == 'Node.GetMUp':
            nd = self.node_stack.peek()
            if nd is not None:
                self.float_stack.push(float(nd.m_up))
        elif n == 'Node.SetMUp':
            val = self.float_stack.pop()
            nd = self.node_stack.peek()
            if val is not None and nd is not None and np.isfinite(val):
                nd.m_up = float(val)
        elif n == 'Node.GetMDown':
            nd = self.node_stack.peek()
            if nd is not None:
                self.float_stack.push(float(nd.m_down))
        elif n == 'Node.SetMDown':
            val = self.float_stack.pop()
            nd = self.node_stack.peek()
            if val is not None and nd is not None and np.isfinite(val):
                nd.m_down = float(val)

        # ============ constants ============
        elif n == 'Float.Const0':    self.float_stack.push(0.0)
        elif n == 'Float.Const1':    self.float_stack.push(1.0)
        elif n == 'Float.ConstHalf': self.float_stack.push(0.5)
        elif n == 'Float.ConstNeg1': self.float_stack.push(-1.0)
        elif n == 'Float.Const2':    self.float_stack.push(2.0)
        elif n == 'Float.Const0_1':  self.float_stack.push(0.1)
        elif n.startswith('Float.EvoConst'):
            idx = int(n[-1])
            if idx < len(self.evolved_constants):
                self.float_stack.push(float(self.evolved_constants[idx]))
            else:
                self.float_stack.push(1.0)
        elif n == 'Int.Const0':      self.int_stack.push(0)
        elif n == 'Int.Const1':      self.int_stack.push(1)
        elif n == 'Int.Const2':      self.int_stack.push(2)
        elif n == 'Bool.True':       self.bool_stack.push(True)
        elif n == 'Bool.False':      self.bool_stack.push(False)
        elif n == 'Float.GetNoiseVar':
            self.float_stack.push(float(self.noise_var))
        elif n == 'Int.GetNumSymbols':
            if self.constellation is not None:
                self.int_stack.push(len(self.constellation))

        # ============ type conversion ============
        elif n == 'Float.FromInt':
            a = self.int_stack.pop()
            if a is not None:
                self.float_stack.push(float(a))
        elif n == 'Int.FromFloat':
            a = self.float_stack.pop()
            if a is not None:
                self.int_stack.push(int(np.clip(a, -1e6, 1e6)))

        # ============ control flow ============
        elif n == 'Exec.If':
            cond = self.bool_stack.pop()
            if cond is not None:
                self._execute_block(ins.code_block if cond else ins.code_block2)

        elif n == 'Exec.While':
            for _ in range(20):
                cond = self.bool_stack.pop()
                if cond is None or not cond:
                    break
                self._execute_block(ins.code_block)

        elif n == 'Exec.DoTimes':
            count = self.int_stack.pop()
            if count is not None:
                for _ in range(max(0, min(int(count), 15))):
                    self._execute_block(ins.code_block)

        elif n == 'Node.ForEachChild':
            nd = self.node_stack.peek()
            if nd is not None and nd.children:
                self._mapreduce(nd.children, ins.code_block)

        elif n == 'Node.ForEachChildMin':
            nd = self.node_stack.peek()
            if nd is not None and nd.children:
                self._minreduce(nd.children, ins.code_block)

        elif n == 'Node.ForEachSibling':
            nd = self.node_stack.peek()
            g = self.graph_stack.peek()
            if nd is not None and g is not None:
                sibs = g.siblings(nd)
                if sibs:
                    self._mapreduce(sibs, ins.code_block)

        elif n == 'Node.ForEachAncestor':
            nd = self.node_stack.peek()
            g = self.graph_stack.peek()
            if nd is not None and g is not None:
                ancs = g.ancestors(nd)
                if ancs:
                    self._mapreduce(ancs, ins.code_block)

        elif n == 'Exec.ForEachSymbol':
            # MapReduce over QAM constellation symbols → Float sum
            # For each symbol s, pushes Re(s) and Im(s) onto float stack,
            # runs block, pops one float result and accumulates sum.
            if self.constellation is not None:
                self._constellation_mapreduce(self.constellation, ins.code_block,
                                              mode='sum')

        elif n == 'Exec.MinOverSymbols':
            # MinReduce over QAM constellation symbols → Float minimum
            # For each symbol s, pushes Re(s) and Im(s) onto float stack,
            # runs block, keeps the minimum result.
            if self.constellation is not None:
                self._constellation_mapreduce(self.constellation, ins.code_block,
                                              mode='min')
