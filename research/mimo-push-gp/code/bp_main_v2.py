"""
Structured BP Stack Decoder — Evolution Runner v2.

Evolves 4 separate programs (F_down, F_up, F_belief, H_halt) that together
define a complete BP-augmented stack decoder.

Run:
    conda activate AutoGenOld
    cd D:\\ChannelCoding\\RCOM\\AlphaDetect\\research\\mimo-push-gp\\code
    python -u -B bp_main_v2.py --continuous --log-suffix sbp_1 --seed 42 --use-cpp
"""
import argparse
import os
import json
import time
import random
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import sys

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

from vm import MIMOPushVM, Instruction, program_to_string, program_to_oneliner
from bp_decoder_v2 import StructuredBPDecoder, qam16_constellation, qpsk_constellation
from stack_decoder import lmmse_detect, kbest_detect
from stacks import TreeNode, SearchTreeGraph
from evolution import (
    random_program, random_block, random_instruction,
    mutate, crossover, deep_copy_program, program_length,
    Individual, FitnessResult,
    tournament_select, PRIMITIVE_INSTRUCTIONS, CONTROL_INSTRUCTIONS,
)


GLOBAL_SEED = 42
N_EVO_CONSTS = 4   # number of evolvable log-domain constants per genome


# --------------------------------------------------------------------------
# Human-readable formula translator (stack → math expression)
# --------------------------------------------------------------------------

def program_to_formula(prog: List[Instruction], input_names: List[str] = None) -> str:
    """Convert a stack program to a human-readable math formula via symbolic tracing.

    Simulates the VM symbolically using SEPARATE stacks for float, int, bool,
    and node types — matching the real VM's multi-stack architecture.
    Returns the top-of-float-stack expression after execution.
    """
    if input_names is None:
        input_names = []

    # Separate symbolic stacks
    fst = list(input_names)   # float stack  (symbolic strings)
    ist = []                  # int stack    (symbolic strings)
    bst = []                  # bool stack   (symbolic strings)
    # node/vector/matrix stacks are not tracked symbolically; we just note presence

    def _fpop():
        return fst.pop() if fst else None
    def _ipop():
        return ist.pop() if ist else None
    def _bpop():
        return bst.pop() if bst else None

    for instr in prog:
        name = instr.name if isinstance(instr, Instruction) else str(instr)

        # ── float arithmetic (binary) ────────────────────────────────────────
        if name == 'Float.Add':
            b, a = _fpop(), _fpop()
            if a is not None and b is not None:
                fst.append(f"({a} + {b})")
        elif name == 'Float.Sub':
            b, a = _fpop(), _fpop()
            if a is not None and b is not None:
                fst.append(f"({a} - {b})")
        elif name == 'Float.Mul':
            b, a = _fpop(), _fpop()
            if a is not None and b is not None:
                fst.append(f"({a} * {b})")
        elif name == 'Float.Div':
            b, a = _fpop(), _fpop()
            if a is not None and b is not None:
                fst.append(f"({a} / {b})")
        elif name == 'Float.Min':
            b, a = _fpop(), _fpop()
            if a is not None and b is not None:
                fst.append(f"min({a}, {b})")
        elif name == 'Float.Max':
            b, a = _fpop(), _fpop()
            if a is not None and b is not None:
                fst.append(f"max({a}, {b})")

        # ── float arithmetic (unary) ─────────────────────────────────────────
        elif name == 'Float.Abs':
            a = _fpop()
            if a is not None: fst.append(f"|{a}|")
        elif name == 'Float.Neg':
            a = _fpop()
            if a is not None: fst.append(f"(-{a})")
        elif name == 'Float.Inv':
            a = _fpop()
            if a is not None: fst.append(f"(1/{a})")
        elif name == 'Float.Sqrt':
            a = _fpop()
            if a is not None: fst.append(f"sqrt({a})")
        elif name == 'Float.Square':
            a = _fpop()
            if a is not None: fst.append(f"({a})^2")
        elif name == 'Float.Tanh':
            a = _fpop()
            if a is not None: fst.append(f"tanh({a})")
        elif name == 'Float.Exp':
            a = _fpop()
            if a is not None: fst.append(f"exp({a})")
        elif name == 'Float.Log':
            a = _fpop()
            if a is not None: fst.append(f"log({a})")

        # ── float stack manipulation ─────────────────────────────────────────
        elif name == 'Float.Dup':
            if fst: fst.append(fst[-1])
        elif name == 'Float.Swap':
            if len(fst) >= 2: fst[-1], fst[-2] = fst[-2], fst[-1]
        elif name == 'Float.Rot':
            if len(fst) >= 3:
                fst[-1], fst[-2], fst[-3] = fst[-3], fst[-1], fst[-2]
        elif name == 'Float.Pop':
            _fpop()

        # ── float constants ──────────────────────────────────────────────────
        elif name == 'Float.Const0':      fst.append("0")
        elif name == 'Float.Const1':      fst.append("1")
        elif name == 'Float.ConstHalf':   fst.append("0.5")
        elif name == 'Float.ConstNeg1':   fst.append("-1")
        elif name == 'Float.Const2':      fst.append("2")
        elif name == 'Float.Const0_1':    fst.append("0.1")
        elif name.startswith('Float.EvoConst'):
            idx = name[-1]
            fst.append(f"EC{idx}")

        # ── float getters (push float from node/environment) ─────────────────
        elif name == 'Node.GetLocalDist':  fst.append("C_i")
        elif name == 'Node.GetCumDist':    fst.append("D_i")
        elif name == 'Node.GetMUp':        fst.append("M_up")
        elif name == 'Node.GetMDown':      fst.append("M_down")
        elif name == 'Node.GetScore':      fst.append("B_i")
        elif name == 'Node.GetSymRe':      fst.append("Re(s)")
        elif name == 'Node.GetSymIm':      fst.append("Im(s)")
        elif name == 'Node.ReadMem':
            _ipop()  # consumes int slot index
            fst.append("mem[k]")
        elif name == 'Float.GetNoiseVar':  fst.append("noise_var")
        elif name == 'Float.GetMMSELB':    fst.append("MMSE_LB")

        # ── float setters / side-effect pops (pop float, no float output) ───
        elif name in ('Node.SetScore', 'Node.SetMUp', 'Node.SetMDown',
                      'Node.WriteMem'):
            _fpop()   # consumes a float value; no float result

        # ── type conversion ──────────────────────────────────────────────────
        elif name == 'Float.FromInt':
            a = _ipop()
            if a is not None:
                fst.append(f"float({a})")
        elif name == 'Int.FromFloat':
            a = _fpop()
            if a is not None:
                ist.append(f"int({a})")

        # ── comparisons that push onto float stack ───────────────────────────
        elif name == 'Float.LT':
            b, a = _fpop(), _fpop()
            if a is not None and b is not None:
                fst.append(f"({a} < {b})")
        elif name == 'Float.GT':
            b, a = _fpop(), _fpop()
            if a is not None and b is not None:
                fst.append(f"({a} > {b})")

        # ── peek-based element access (push float from matrix/vector) ─────────
        elif name == 'Mat.PeekAt':
            # Pop j from int, peek i from int, peek M from matrix → push Re(M[i,j])
            j = _ipop()
            i_sym = ist[-1] if ist else "i"
            j_sym = j if j is not None else "j"
            fst.append(f"Re(R[{i_sym},{j_sym}])")
            if j is not None:
                ist.append(j)  # restore j
        elif name == 'Mat.PeekAtIm':
            j = _ipop()
            i_sym = ist[-1] if ist else "i"
            j_sym = j if j is not None else "j"
            fst.append(f"Im(R[{i_sym},{j_sym}])")
            if j is not None:
                ist.append(j)
        elif name == 'Vec.PeekAt':
            i_sym = ist[-1] if ist else "i"
            fst.append(f"Re(x[{i_sym}])")
        elif name == 'Vec.PeekAtIm':
            i_sym = ist[-1] if ist else "i"
            fst.append(f"Im(x[{i_sym}])")
        elif name == 'Vec.SecondPeekAt':
            i_sym = ist[-1] if ist else "i"
            fst.append(f"Re(y[{i_sym}])")
        elif name == 'Vec.SecondPeekAtIm':
            i_sym = ist[-1] if ist else "i"
            fst.append(f"Im(y[{i_sym}])")
        elif name == 'Vec.GetResidue':
            pass  # pushes a vector, not a float

        # ── vector ops that result in a float ────────────────────────────────
        elif name == 'Vec.Dot':
            fst.append("dot(v1,v2)")   # consumes 2 vectors (untracked)
        elif name == 'Vec.Norm2':
            fst.append("||v||^2")
        elif name == 'Vec.ElementAt':
            idx = _ipop()
            idx_sym = idx if idx is not None else "i"
            fst.append(f"v[{idx_sym}]")
        elif name == 'Mat.ElementAt':
            j = _ipop(); i = _ipop()
            fst.append(f"R[{i or 'i'},{j or 'j'}]")
        elif name == 'Vec.Scale':
            _fpop()   # consumes scale factor; result is a vector (not float)

        # ── int stack: push constants ────────────────────────────────────────
        elif name == 'Int.Const0':     ist.append("0")
        elif name == 'Int.Const1':     ist.append("1")
        elif name == 'Int.Const2':     ist.append("2")
        elif name == 'Int.GetNumSymbols': ist.append("M")

        # ── int stack: arithmetic (no float effect) ──────────────────────────
        elif name == 'Int.Add':
            b, a = _ipop(), _ipop()
            if a is not None and b is not None:
                ist.append(f"({a}+{b})")
        elif name == 'Int.Sub':
            b, a = _ipop(), _ipop()
            if a is not None and b is not None:
                ist.append(f"({b}-{a})")
        elif name == 'Int.Inc':
            a = _ipop()
            if a is not None: ist.append(f"({a}+1)")
        elif name == 'Int.Dec':
            a = _ipop()
            if a is not None: ist.append(f"({a}-1)")

        # ── int stack: manipulation (no float effect) ────────────────────────
        elif name == 'Int.Pop':   _ipop()
        elif name == 'Int.Dup':
            if ist: ist.append(ist[-1])
        elif name == 'Int.Swap':
            if len(ist) >= 2: ist[-1], ist[-2] = ist[-2], ist[-1]

        # ── int getters (push to INT stack, NOT float) ───────────────────────
        # Node.GetLayer, Graph.NodeCount, Graph.FrontierCount,
        # Node.NumChildren, Vec.Len, Mat.Rows → push to int stack
        elif name == 'Node.GetLayer':       ist.append("layer")
        elif name == 'Node.NumChildren':    ist.append("N_ch")
        elif name == 'Graph.NodeCount':     ist.append("N_nodes")
        elif name == 'Graph.FrontierCount': ist.append("N_frontier")
        elif name == 'Vec.Len':             ist.append("v.len")
        elif name == 'Mat.Rows':            ist.append("M.rows")

        # ── int comparisons (push to bool stack, not float) ─────────────────
        elif name == 'Int.LT':
            b, a = _ipop(), _ipop()
            if a is not None and b is not None:
                bst.append(f"({a}<{b})")
        elif name == 'Int.GT':
            b, a = _ipop(), _ipop()
            if a is not None and b is not None:
                bst.append(f"({a}>{b})")

        # ── bool stack: logic (no float effect) ─────────────────────────────
        elif name == 'Bool.And':
            b, a = _bpop(), _bpop()
            if a is not None and b is not None:
                bst.append(f"({a} AND {b})")
        elif name == 'Bool.Or':
            b, a = _bpop(), _bpop()
            if a is not None and b is not None:
                bst.append(f"({a} OR {b})")
        elif name == 'Bool.Not':
            a = _bpop()
            if a is not None: bst.append(f"NOT({a})")
        elif name == 'Bool.True':   bst.append("TRUE")
        elif name == 'Bool.False':  bst.append("FALSE")
        elif name == 'Bool.Pop':    _bpop()
        elif name == 'Bool.Dup':
            if bst: bst.append(bst[-1])
        elif name == 'Node.IsExpanded': bst.append("is_expanded")

        # ── node/graph navigation (no float effect) ──────────────────────────
        # These push onto the node stack, not the float stack.
        elif name in ('Node.GetParent', 'Graph.GetRoot', 'Node.ChildAt',
                      'Node.Pop', 'Node.Dup', 'Node.Swap',
                      'Matrix.Pop', 'Matrix.Dup',
                      'Vector.Pop', 'Vector.Dup', 'Vector.Swap',
                      'Mat.Row', 'Mat.VecMul',
                      'Vec.Add', 'Vec.Sub'):
            pass  # operate on node/vector/matrix stacks only

        # ── control structures ────────────────────────────────────────────────
        elif name == 'Node.ForEachChild' and hasattr(instr, 'code_block') and instr.code_block:
            # _mapreduce: body runs for each child; body's float-stack top gets summed.
            # Trace with empty input first (body uses Node.GetMUp → 'M_up').
            inner = program_to_formula(instr.code_block, [])
            if inner == 'EMPTY':
                # Body consumes items from the outer float stack; trace with context.
                ctx = list(fst[-2:]) if len(fst) >= 2 else list(fst)
                inner = program_to_formula(instr.code_block, ctx)
            if inner != 'EMPTY':
                fst.append(f"Sum_children({inner})")
            else:
                body_ops = ', '.join(
                    (i.name + '[…]' if getattr(i, 'code_block', None) else i.name)
                    for i in instr.code_block)
                fst.append(f"ForEachChild{{{body_ops}}}")
        elif name == 'Node.ForEachSibling' and hasattr(instr, 'code_block') and instr.code_block:
            inner = program_to_formula(instr.code_block, [])
            if inner == 'EMPTY':
                ctx = list(fst[-2:]) if len(fst) >= 2 else list(fst)
                inner = program_to_formula(instr.code_block, ctx)
            fst.append(f"Sum_siblings({inner})" if inner != 'EMPTY' else f"ForEachSibling{{...}}")
        elif name == 'Node.ForEachAncestor' and hasattr(instr, 'code_block') and instr.code_block:
            inner = program_to_formula(instr.code_block, [])
            if inner == 'EMPTY':
                ctx = list(fst[-2:]) if len(fst) >= 2 else list(fst)
                inner = program_to_formula(instr.code_block, ctx)
            fst.append(f"Sum_ancestors({inner})" if inner != 'EMPTY' else f"ForEachAncestor{{...}}")
        elif name == 'Exec.DoTimes' and hasattr(instr, 'code_block') and instr.code_block:
            n = _ipop() or "n"
            # Trace inner body with context from the current top-of-stack items
            # (DoTimes body operates on the existing float stack items).
            ctx = list(fst[-3:]) if len(fst) >= 3 else list(fst)
            inner = program_to_formula(instr.code_block, ctx)
            # Use loop(n){ body } notation, clearer than repeat(nx: body)
            fst.append(f"loop({n}){{ {inner} }}")
        elif name == 'Exec.ForEachSymbol' and hasattr(instr, 'code_block') and instr.code_block:
            inner = program_to_formula(instr.code_block, ['Re(s_k)', 'Im(s_k)'])
            fst.append(f"Sum_symbols({inner})")
        elif name == 'Exec.MinOverSymbols' and hasattr(instr, 'code_block') and instr.code_block:
            inner = program_to_formula(instr.code_block, ['Re(s_k)', 'Im(s_k)'])
            fst.append(f"Min_symbols({inner})")

        # ── everything else: silently skip (non-float-producing instruction) ─
        else:
            pass  # no [?] tokens — unrecognized ops are no-ops on float stack

    return fst[-1] if fst else "EMPTY"


def program_to_formula_trace(prog: List[Instruction], prog_type: str,
                             genome: 'Genome' = None) -> str:
    """Convert a program to formula by actual VM execution with symbolic inputs.

    Instead of purely symbolic tracing, runs the program on concrete numeric 
    inputs and observes which inputs affect the output via perturbation.
    Returns a description of the effective computation.
    
    prog_type: 'down', 'up', 'belief', 'halt'
    """
    from stacks import SearchTreeGraph
    vm = MIMOPushVM(flops_max=50000, step_max=500)
    if genome is not None:
        vm.evolved_constants = genome.evo_constants
    
    constellation = qam16_constellation()
    # Nt_test must be >= max layer in test tree so GetMMSELB/GetResidue
    # fire (they check k <= R.cols where k = node.layer).
    Nt_test = 4
    rng = np.random.RandomState(12345)
    
    # Build test tree
    graph = SearchTreeGraph()
    root = graph.create_root(layer=Nt_test)  # root.layer = Nt_test
    root.m_down = 0.5
    root.m_up = 0.3
    root.local_dist = 0.2
    root.cum_dist = 0.2
    root.partial_symbols = np.array([], dtype=complex)
    
    for c_i in range(16):
        sym = 0.3 + 0.1j * (c_i % 4)
        ld = 0.1 + 0.05 * c_i          # spread local_dist values
        cd = root.cum_dist + ld
        child = graph.add_child(root, layer=Nt_test - 1, symbol=sym,
                                local_dist=ld, cum_dist=cd,
                                partial_symbols=np.array([sym]))
        child.m_up = 0.1 + 0.07 * c_i  # spread m_up values
        child.m_down = root.m_down + ld
    
    R = (np.eye(Nt_test, dtype=complex) * 1.1 +
         0.1 * (rng.randn(Nt_test, Nt_test) + 1j * rng.randn(Nt_test, Nt_test)))
    y_tilde = rng.randn(Nt_test) + 1j * rng.randn(Nt_test)
    child = root.children[0]
    nv = 0.1
    
    def _run_with_vals(float_vals, int_vals=None, node=None, get_bool=False):
        vm.reset()
        if genome is not None:
            vm.evolved_constants = genome.evo_constants
        n = node or child
        vm.candidate_node = n
        vm.constellation = constellation
        vm.noise_var = nv
        vm.matrix_stack.push(R.copy())
        vm.vector_stack.push(y_tilde.copy())
        vm.vector_stack.push(n.partial_symbols.copy())
        vm.graph_stack.push(graph)
        vm.node_stack.push(n)
        vm.int_stack.push(n.layer)
        for v in float_vals:
            vm.float_stack.push(float(v))
        if int_vals:
            for v in int_vals:
                vm.int_stack.push(v)
        try:
            vm._execute_block(prog)
        except Exception:
            pass
        if get_bool:
            r = vm.bool_stack.peek()
            return bool(r) if r is not None else None
        r = vm.float_stack.peek()
        return float(r) if (r is not None and np.isfinite(r)) else None
    
    # Construct base inputs and test perturbations
    if prog_type == 'down':
        base = [0.5, 0.3]  # M_parent_down, C_i
        labels = ['M_parent_down', 'C_i']
        base_result = _run_with_vals(base)
        deps = []
        for i, lbl in enumerate(labels):
            pert = list(base)
            pert[i] = base[i] + 1.7
            r = _run_with_vals(pert)
            if r is not None and base_result is not None and abs(r - base_result) > 1e-10:
                deps.append(lbl)
        # Also check env deps
        for env_name, env_fn in [('M_up', lambda: setattr(child, 'm_up', child.m_up + 1.5)),
                                  ('M_down', lambda: setattr(child, 'm_down', child.m_down + 1.5))]:
            old = getattr(child, env_name.lower().replace('_', '_'))
            r_before = _run_with_vals(base)
            if env_name == 'M_up': child.m_up += 1.5
            else: child.m_down += 1.5
            r_after = _run_with_vals(base)
            if env_name == 'M_up': child.m_up -= 1.5
            else: child.m_down -= 1.5
            if r_before is not None and r_after is not None and abs(r_before - r_after) > 1e-10:
                deps.append(env_name)
        formula = program_to_formula(prog, ['M_parent_down', 'C_i'])
        if base_result is not None:
            return f"{formula}  [deps: {','.join(deps) if deps else 'CONST'}; val={base_result:.4f}]"
        return f"{formula}  [deps: {','.join(deps) if deps else 'CONST'}; val=FAULT]"
        
    elif prog_type == 'up':
        base_floats = []
        for c in root.children:
            base_floats.extend([c.local_dist, c.m_up])
        base_result = _run_with_vals(base_floats, int_vals=[len(root.children)], node=root)
        deps = []
        # Perturb each child's m_up
        for ci, c in enumerate(root.children):
            old = c.m_up
            c.m_up += 1.5
            pert_floats = []
            for cc in root.children:
                pert_floats.extend([cc.local_dist, cc.m_up])
            r = _run_with_vals(pert_floats, int_vals=[len(root.children)], node=root)
            c.m_up = old
            if r is not None and base_result is not None and abs(r - base_result) > 1e-10:
                deps.append(f'M_up_{ci}')
        flat_names = []
        for _i in range(1, len(root.children) + 1):
            flat_names.extend([f'C_{_i}', f'M_{_i}'])
        formula = program_to_formula(prog, flat_names)
        # Compact deps display: if many/all children's M_up matter, summarise
        if len(deps) >= len(root.children) * 3 // 4 and all(d.startswith('M_up_') for d in deps):
            deps_str = f'all_M_up({len(deps)}/{len(root.children)})'
        elif len(deps) > 6 and all(d.startswith('M_up_') for d in deps):
            deps_str = f'M_up_*({len(deps)}/{len(root.children)})'
        else:
            deps_str = ','.join(deps) if deps else 'CONST/noop'
        if base_result is not None:
            return f"{formula}  [deps: {deps_str}; val={base_result:.4f}]"
        return f"{formula}  [deps: {deps_str}; val=FAULT]"
        
    elif prog_type == 'belief':
        base = [0.5, 0.3, 0.2]  # cum_dist, m_down, m_up
        labels = ['D_i', 'M_down', 'M_up']
        base_result = _run_with_vals(base)
        deps = []
        # Test float-stack perturbations
        for i, lbl in enumerate(labels):
            pert = list(base)
            pert[i] = base[i] + 1.7
            r = _run_with_vals(pert)
            if r is not None and base_result is not None and abs(r - base_result) > 1e-10:
                deps.append(lbl)
        # Also test node-property perturbations (Node.GetCumDist / GetMDown / GetMUp)
        node_props = [('D_i', 'cum_dist'), ('M_down', 'm_down'), ('M_up', 'm_up')]
        for lbl, attr in node_props:
            if lbl in deps:
                continue  # already detected via float-stack
            orig_val = getattr(child, attr)
            setattr(child, attr, orig_val + 1.7)
            r = _run_with_vals(base)
            setattr(child, attr, orig_val)
            if r is not None and base_result is not None and abs(r - base_result) > 1e-10:
                deps.append(lbl)
        formula = program_to_formula(prog, ['D_i', 'M_down', 'M_up'])
        if base_result is not None:
            return f"{formula}  [deps: {','.join(deps) if deps else 'CONST'}; val={base_result:.4f}]"
        return f"{formula}  [deps: {','.join(deps) if deps else 'CONST'}; val=FAULT]"
        
    else:  # halt
        formula = program_to_formula(prog, ['old_M_up', 'new_M_up'])
        results = set()
        for trial in range(10):
            old_m = trial * 0.5 - 2.0
            new_m = old_m + trial * 0.3
            r = _run_with_vals([old_m, new_m], get_bool=True)
            if r is not None:
                results.add(r)
        return f"{formula}  [varies: {'YES' if len(results) >= 2 else 'NO (always ' + str(results.pop() if results else '?') + ')'}]"


# ──────────────────────────────────────────────────────────────────────────────
# Multi-line pseudocode generator
# ──────────────────────────────────────────────────────────────────────────────

def program_to_pseudocode(prog: List[Instruction],
                          input_names: List[str] = None,
                          int_names: List[str] = None,
                          indent: int = 0) -> List[str]:
    """Generate indented Python-style pseudocode from a Push VM stack program.

    Symbolic stack tracking: simple ops build up expressions on stacks.
    Control structures (ForEachChild / DoTimes / If) become indented for-loops.
    Complex expressions are assigned to named temp vars before appearing in loops.

    Args:
        prog:        list of Instructions
        input_names: initial float stack (bottom→top; last = top)
        int_names:   initial int stack (bottom→top; last = top)
        indent:      base indentation level (each level = 2 spaces)

    Returns: list of code lines (strings, no trailing newlines)
    """
    import re as _re
    PAD = '  ' * indent
    lines: List[str] = []

    fst: List[str] = list(input_names or [])
    ist: List[str] = list(int_names or ['layer'])
    bst: List[str] = []
    _cnt = [0]

    def fresh(hint: str = 't') -> str:
        _cnt[0] += 1
        return f'{hint}{_cnt[0]}'

    def emit(s: str):
        lines.append(PAD + s)

    def fpop() -> str:
        return fst.pop() if fst else '?'

    def ipop() -> str:
        return ist.pop() if ist else '?'

    def bpop() -> str:
        return bst.pop() if bst else '?'

    _SIMPLE = _re.compile(r'^[A-Za-z0-9_σ²‖ .|\[\]^()]+$')  # "simple enough" check

    def materialize(expr: str, hint: str = 't') -> str:
        """If expr is complex (nested ops), assign to a temp var and return its name."""
        # Already a simple identifier or short literal → keep inline
        if len(expr) <= 12 and _SIMPLE.match(expr):
            return expr
        v = fresh(hint)
        emit(f'{v} = {expr}')
        return v

    def extract_return(body_lines: List[str]) -> str:
        """Pop the last 'return <expr>' line from body_lines and return the expr."""
        while body_lines and body_lines[-1].strip().startswith('return '):
            ret_line = body_lines.pop()
            return ret_line.strip()[len('return '):].split('  #')[0].strip()
        return '?'

    for instr in prog:
        nm = instr.name if isinstance(instr, Instruction) else str(instr)

        # ── float binary ────────────────────────────────────────────────────
        if nm == 'Float.Add':
            b, a = fpop(), fpop();  fst.append(f'({a} + {b})')
        elif nm == 'Float.Sub':
            b, a = fpop(), fpop();  fst.append(f'({a} - {b})')
        elif nm == 'Float.Mul':
            b, a = fpop(), fpop();  fst.append(f'({a} * {b})')
        elif nm == 'Float.Div':
            b, a = fpop(), fpop();  fst.append(f'({a} / {b})')
        elif nm == 'Float.Min':
            b, a = fpop(), fpop();  fst.append(f'min({a}, {b})')
        elif nm == 'Float.Max':
            b, a = fpop(), fpop();  fst.append(f'max({a}, {b})')

        # ── float unary ──────────────────────────────────────────────────────
        elif nm == 'Float.Abs':    fst.append(f'abs({fpop()})')
        elif nm == 'Float.Neg':    fst.append(f'(-{fpop()})')
        elif nm == 'Float.Inv':    fst.append(f'(1/{fpop()})')
        elif nm == 'Float.Sqrt':   fst.append(f'sqrt({fpop()})')
        elif nm == 'Float.Square': fst.append(f'({fpop()})^2')
        elif nm == 'Float.Tanh':   fst.append(f'tanh({fpop()})')
        elif nm == 'Float.Exp':    fst.append(f'exp({fpop()})')
        elif nm == 'Float.Log':    fst.append(f'log({fpop()})')

        # ── float stack ops ──────────────────────────────────────────────────
        elif nm == 'Float.Pop':
            v = fpop()
            if v not in ('?',) and ('(' in v or len(v) > 8):
                emit(f'# discard: {v}')
        elif nm == 'Float.Dup':
            if fst: fst.append(fst[-1])
        elif nm == 'Float.Swap':
            if len(fst) >= 2: fst[-1], fst[-2] = fst[-2], fst[-1]
        elif nm == 'Float.Rot':
            if len(fst) >= 3:
                fst[-1], fst[-2], fst[-3] = fst[-3], fst[-1], fst[-2]

        # ── float constants ──────────────────────────────────────────────────
        elif nm == 'Float.Const0':    fst.append('0')
        elif nm == 'Float.Const1':    fst.append('1')
        elif nm == 'Float.ConstHalf': fst.append('0.5')
        elif nm == 'Float.ConstNeg1': fst.append('-1')
        elif nm == 'Float.Const2':    fst.append('2')
        elif nm == 'Float.Const0_1':  fst.append('0.1')
        elif nm.startswith('Float.EvoConst'):
            fst.append(f'EC{nm[-1]}')

        # ── node / environment getters → float ────────────────────────────────
        elif nm == 'Node.GetLocalDist': fst.append('C_i')    # local_dist
        elif nm == 'Node.GetCumDist':   fst.append('D_i')    # cum_dist
        elif nm == 'Node.GetMUp':       fst.append('M_up')   # node.m_up
        elif nm == 'Node.GetMDown':     fst.append('M_dn')   # node.m_down
        elif nm == 'Node.GetScore':     fst.append('score')
        elif nm == 'Node.GetSymRe':     fst.append('Re(s)')
        elif nm == 'Node.GetSymIm':     fst.append('Im(s)')
        elif nm == 'Float.GetNoiseVar': fst.append('σ²')
        elif nm == 'Float.GetMMSELB':   fst.append('MMSE_LB')
        elif nm == 'Node.ReadMem':
            k = ipop(); fst.append(f'mem[{k}]')

        # ── float setters (pop float, side effect) ────────────────────────────
        elif nm == 'Node.SetMUp':
            v = materialize(fpop(), 'm'); emit(f'node.m_up = {v}')
        elif nm == 'Node.SetMDown':
            v = materialize(fpop(), 'm'); emit(f'node.m_dn = {v}')
        elif nm == 'Node.SetScore':
            v = materialize(fpop(), 's'); emit(f'node.score = {v}')
        elif nm == 'Node.WriteMem':
            v = fpop(); k = ipop(); emit(f'node.mem[{k}] = {v}')

        # ── type conversions ──────────────────────────────────────────────────
        elif nm == 'Float.FromInt':  fst.append(f'float({ipop()})')
        elif nm == 'Int.FromFloat':  ist.append(f'int({fpop()})')

        # ── int constants & getters ───────────────────────────────────────────
        elif nm == 'Int.Const0':          ist.append('0')
        elif nm == 'Int.Const1':          ist.append('1')
        elif nm == 'Int.Const2':          ist.append('2')
        elif nm == 'Int.GetNumSymbols':   ist.append('M')
        elif nm == 'Node.GetLayer':       ist.append('layer')
        elif nm == 'Node.NumChildren':    ist.append('N_ch')
        elif nm == 'Graph.NodeCount':     ist.append('N_nodes')
        elif nm == 'Graph.FrontierCount': ist.append('N_frt')
        elif nm == 'Vec.Len':             ist.append('v.len')
        elif nm == 'Mat.Rows':            ist.append('M.rows')

        # ── int stack ops ─────────────────────────────────────────────────────
        elif nm == 'Int.Pop':
            v = ipop()
            if v not in ('?', '0', '1') and not (isinstance(v, str) and v.isdigit()):
                emit(f'# int_drop: {v}')
        elif nm == 'Int.Dup':
            if ist: ist.append(ist[-1])
        elif nm == 'Int.Swap':
            if len(ist) >= 2: ist[-1], ist[-2] = ist[-2], ist[-1]
        elif nm == 'Int.Add':
            b, a = ipop(), ipop(); ist.append(f'({a}+{b})')
        elif nm == 'Int.Sub':
            b, a = ipop(), ipop(); ist.append(f'({a}-{b})')
        elif nm == 'Int.Inc': ist.append(f'({ipop()}+1)')
        elif nm == 'Int.Dec': ist.append(f'({ipop()}-1)')
        elif nm == 'Int.LT':
            b, a = ipop(), ipop(); bst.append(f'({a} < {b})')
        elif nm == 'Int.GT':
            b, a = ipop(), ipop(); bst.append(f'({a} > {b})')

        # ── float comparisons → bool ──────────────────────────────────────────
        elif nm == 'Float.LT':
            b, a = fpop(), fpop(); bst.append(f'({a} < {b})')
        elif nm == 'Float.GT':
            b, a = fpop(), fpop(); bst.append(f'({a} > {b})')

        # ── bool ops ──────────────────────────────────────────────────────────
        elif nm == 'Bool.And':
            b, a = bpop(), bpop(); bst.append(f'({a} and {b})')
        elif nm == 'Bool.Or':
            b, a = bpop(), bpop(); bst.append(f'({a} or {b})')
        elif nm == 'Bool.Not':  bst.append(f'not({bpop()})')
        elif nm == 'Bool.True': bst.append('True')
        elif nm == 'Bool.False': bst.append('False')
        elif nm == 'Bool.Pop':  bpop()
        elif nm == 'Bool.Dup':
            if bst: bst.append(bst[-1])
        elif nm == 'Node.IsExpanded': bst.append('is_expanded')

        # ── CONTROL FLOW ──────────────────────────────────────────────────────

        elif nm == 'Node.ForEachChild' and getattr(instr, 'code_block', None):
            # VM: for each child: push child on node_stack, run body, pop float top → sum
            acc = fresh('acc')
            body_ctx = fst[-4:] if len(fst) >= 4 else list(fst)
            body_lines = program_to_pseudocode(instr.code_block, body_ctx,
                                               ['child_layer'], indent + 1)
            ret_expr = extract_return(body_lines)
            emit(f'{acc} = 0.0')
            emit(f'for child_j in node.children:  # ForEachChild → Σ body(child_j)')
            lines.extend(body_lines)
            emit(f'  {acc} += {ret_expr}')
            fst.append(acc)

        elif nm == 'Node.ForEachSibling' and getattr(instr, 'code_block', None):
            acc = fresh('acc')
            body_ctx = fst[-2:] if len(fst) >= 2 else list(fst)
            body_lines = program_to_pseudocode(instr.code_block, body_ctx,
                                               ['sib_layer'], indent + 1)
            ret_expr = extract_return(body_lines)
            emit(f'{acc} = 0.0')
            emit(f'for sib_j in node.siblings:  # ForEachSibling → Σ body(sib_j)')
            lines.extend(body_lines)
            emit(f'  {acc} += {ret_expr}')
            fst.append(acc)

        elif nm == 'Node.ForEachAncestor' and getattr(instr, 'code_block', None):
            acc = fresh('acc')
            body_ctx = fst[-2:] if len(fst) >= 2 else list(fst)
            body_lines = program_to_pseudocode(instr.code_block, body_ctx,
                                               ['anc_layer'], indent + 1)
            ret_expr = extract_return(body_lines)
            emit(f'{acc} = 0.0')
            emit(f'for anc_j in node.ancestors:  # ForEachAncestor')
            lines.extend(body_lines)
            emit(f'  {acc} += {ret_expr}')
            fst.append(acc)

        elif nm == 'Exec.DoTimes' and getattr(instr, 'code_block', None):
            n_sym = ipop()
            # Materialize complex stack items to named vars before the loop,
            # so the pseudocode body can reference them by name.
            for i in range(len(fst)):
                fst[i] = materialize(fst[i], 'v')
            body_ctx = fst[-4:] if len(fst) >= 4 else list(fst)
            body_lines = program_to_pseudocode(instr.code_block, body_ctx,
                                               list(ist), indent + 1)
            # Convert the final 'return X' line to a comment showing the stack
            # top value after each iteration, rather than stripping it silently.
            ret_expr = '?'
            if body_lines and body_lines[-1].strip().startswith('return '):
                ret_line = body_lines[-1]
                stripped = ret_line.strip()
                ret_expr = stripped[len('return '):].split('  #')[0].strip()
                ind_chars = ret_line[:len(ret_line) - len(ret_line.lstrip())]
                body_lines[-1] = ind_chars + f'# → stack_top = {ret_expr}'
            n_label = n_sym if n_sym != '?' else 'n'
            emit(f'for _ in range({n_label}):  # Exec.DoTimes (max 15 iters)')
            lines.extend(body_lines)
            # After the loop, approximate: the float stack top becomes ret_expr
            if ret_expr != '?':
                if fst:
                    fst[-1] = ret_expr
                else:
                    fst.append(ret_expr)

        elif nm == 'Exec.ForEachSymbol' and getattr(instr, 'code_block', None):
            acc = fresh('acc')
            body_lines = program_to_pseudocode(instr.code_block,
                                               ['Re(s)', 'Im(s)'], [], indent + 1)
            ret_expr = extract_return(body_lines)
            emit(f'{acc} = 0.0')
            emit(f'for s in constellation:  # ForEachSymbol: push Re(s), Im(s) per iter')
            lines.extend(body_lines)
            emit(f'  {acc} += {ret_expr}')
            fst.append(acc)

        elif nm == 'Exec.MinOverSymbols' and getattr(instr, 'code_block', None):
            acc = fresh('best')
            body_lines = program_to_pseudocode(instr.code_block,
                                               ['Re(s)', 'Im(s)'], [], indent + 1)
            ret_expr = extract_return(body_lines)
            emit(f'{acc} = inf')
            emit(f'for s in constellation:  # MinOverSymbols: push Re(s), Im(s) per iter')
            lines.extend(body_lines)
            emit(f'  {acc} = min({acc}, {ret_expr})')
            fst.append(acc)

        elif nm == 'Exec.If' and getattr(instr, 'code_block', None):
            cond = bpop()
            emit(f'if {cond}:')
            then_lines = program_to_pseudocode(instr.code_block, list(fst),
                                               list(ist), indent + 1)
            lines.extend(then_lines)
            if getattr(instr, 'code_block2', None):
                emit('else:')
                else_lines = program_to_pseudocode(instr.code_block2, list(fst),
                                                   list(ist), indent + 1)
                lines.extend(else_lines)

        # ── everything else: no float output ──────────────────────────────────
        else:
            pass  # Mat.Row, Vec.Add, Node.GetParent, etc.

    # Final return: prefer bool result (used by halt programs); show both when
    # both stacks are non-empty so the caller can see what the program computes.
    if bst and fst:
        emit(f'return {bst[-1]}  # bool  (float stack also has: {fst[-1]})')
    elif bst:
        emit(f'return {bst[-1]}  # bool')
    elif fst:
        emit(f'return {fst[-1]}')
    else:
        emit('return <nothing>')

    return lines


def _format_pseudocode_block(prog: List[Instruction],
                              prog_type: str,
                              genome: 'Genome',
                              base_indent: int = 0) -> List[str]:
    """Build pseudocode lines with the correct input context for each program type."""
    ec = genome.evo_constants

    if prog_type == 'down':
        # float: [M_parent_down, C_i] (C_i = top), int: [layer]
        return program_to_pseudocode(prog, ['M_par_down', 'C_i'], ['layer'], base_indent)

    elif prog_type == 'up':
        # float: [C_1,M_1, C_2,M_2, ..., C_16,M_16] (M_16 = top)
        # int:   [layer, N_ch] (N_ch = top)
        flat = []
        for i in range(1, 17):
            flat.extend([f'C_{i}', f'M_{i}'])
        return program_to_pseudocode(prog, flat, ['layer', 'N_ch'], base_indent)

    elif prog_type == 'belief':
        # float: [D_i, M_dn, M_up] (M_up = top), int: [layer]
        return program_to_pseudocode(prog, ['D_i', 'M_dn', 'M_up'], ['layer'], base_indent)

    else:  # halt
        # float: [old_M_up, new_M_up] (new_M_up = top), int: [layer]
        return program_to_pseudocode(prog, ['old_Mup', 'new_Mup'], ['layer'], base_indent)


def print_genome_formulas(genome: 'Genome', label: str = ""):
    """Print all 4 programs as human-readable formulas with trace info."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    
    print(f"  F_down(M_parent_down, C_i) = "
          f"{program_to_formula_trace(genome.prog_down, 'down', genome)}")
    print(f"  F_up({{C_j, M_j_up}}) = "
          f"{program_to_formula_trace(genome.prog_up, 'up', genome)}")
    print(f"  F_belief(D_i, M_down, M_up) = "
          f"{program_to_formula_trace(genome.prog_belief, 'belief', genome)}")
    print(f"  H_halt(old_M_up, new_M_up) = "
          f"{program_to_formula_trace(genome.prog_halt, 'halt', genome)}")
    
    # Show mathematical relationship
    print(f"\n  === BP Message Passing Flow ===")
    print(f"  1. Expand node → create M children with local_dist, cum_dist")
    print(f"  2. UP sweep (leaves→root): parent.m_up = F_up(children's m_up, local_dist)")
    print(f"  3. DOWN sweep (root→leaves): child.m_down = F_down(parent.m_down, child.local_dist)")
    print(f"  4. Score frontier: node.score = F_belief(cum_dist, m_down, m_up)")
    print(f"  5. Repeat BP if H_halt(old_root_m_up, new_root_m_up) = False")
    print(f"  6. Pop best-scored frontier node → expand → goto 1")
    
    # Show evolved constants
    ec = genome.evo_constants
    print(f"\n  Evolved constants: EC0={ec[0]:.6f}, EC1={ec[1]:.6f}, "
          f"EC2={ec[2]:.6f}, EC3={ec[3]:.6f}")
    print(f"  Log-domain:        {', '.join(f'{v:.4f}' for v in genome.log_constants)}")
    print()
    print(f"  H_halt(old_M_up, new_M_up) = "
          f"{program_to_formula(genome.prog_halt, ['old_M_up', 'new_M_up'])}")
    print()


def _log_genome_formulas(logger, genome: 'Genome', label: str = ""):
    """Write genome formulas to the log file (not console)."""
    lines = []
    if label:
        lines.append(f"\n{'='*60}")
        lines.append(f"  {label}")
        lines.append(f"{'='*60}")
    lines.append(f"  F_down(M_parent_down, C_i) = "
                 f"{program_to_formula_trace(genome.prog_down, 'down', genome)}")
    lines.append(f"  F_up({{C_j, M_j_up}}) = "
                 f"{program_to_formula_trace(genome.prog_up, 'up', genome)}")
    lines.append(f"  F_belief(D_i, M_down, M_up) = "
                 f"{program_to_formula_trace(genome.prog_belief, 'belief', genome)}")
    lines.append(f"  H_halt(old_M_up, new_M_up) = "
                 f"{program_to_formula_trace(genome.prog_halt, 'halt', genome)}")
    ec = genome.evo_constants
    lines.append(f"  Evolved constants: EC0={ec[0]:.6f}, EC1={ec[1]:.6f}, EC2={ec[2]:.6f}, EC3={ec[3]:.6f}")
    lines.append(f"  === BP Flow: UP(leaves→root) → DOWN(root→leaves) → Score(frontier) ===")
    lines.append("")
    logger._w("\n".join(lines) + "\n")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def complex_gaussian(shape, rng):
    return (rng.randn(*shape) + 1j * rng.randn(*shape)) / np.sqrt(2.0)


def generate_mimo_sample(Nr, Nt, constellation, snr_db, rng):
    H = complex_gaussian((Nr, Nt), rng)
    x_idx = rng.randint(0, len(constellation), size=Nt)
    x = constellation[x_idx]
    sig_power = float(np.mean(np.abs(H @ x) ** 2))
    noise_var = sig_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_var / 2.0) * (rng.randn(Nr) + 1j * rng.randn(Nr))
    y = H @ x + noise
    return H, x, y, noise_var


def ber_calc(x_true, x_hat):
    return float(np.mean(x_true != x_hat))


def constellation_for(mod_order):
    if mod_order == 4:
        return qpsk_constellation()
    if mod_order == 16:
        return qam16_constellation()
    raise ValueError(mod_order)


# --------------------------------------------------------------------------
# 4-Program Genome
# --------------------------------------------------------------------------

# Instructions biased for each program type
_DOWN_INSTR = [
    'Float.Add', 'Float.Sub', 'Float.Mul', 'Float.Div',
    'Float.Min', 'Float.Max', 'Float.Abs', 'Float.Neg',
    'Float.Sqrt', 'Float.Square', 'Float.Tanh', 'Float.Exp', 'Float.Log',
    'Float.Dup', 'Float.Swap',
    'Node.GetLocalDist', 'Node.GetCumDist', 'Node.GetLayer',
    'Node.GetMDown', 'Node.GetMUp',
    'Float.GetNoiseVar',
    'Float.Const0', 'Float.Const1', 'Float.ConstHalf', 'Float.Const2',
    'Node.GetSymRe', 'Node.GetSymIm',
    # matrix / vector peek for building custom heuristics
    'Mat.PeekAt', 'Mat.PeekAtIm', 'Vec.PeekAt', 'Vec.PeekAtIm',
    'Vec.SecondPeekAt', 'Vec.SecondPeekAtIm',
    'Vec.Dot', 'Vec.Norm2', 'Vec.Scale',
    'Mat.VecMul', 'Mat.Row', 'Vec.ElementAt',
    'Vec.Len', 'Mat.Rows',
    'Int.Const0', 'Int.Const1', 'Int.Const2',
    'Int.Add', 'Int.Sub', 'Int.Inc', 'Int.Dec',
    'Exec.DoTimes',
] + [f'Float.EvoConst{i}' for i in range(N_EVO_CONSTS)]

_UP_INSTR = [
    'Float.Add', 'Float.Sub', 'Float.Mul', 'Float.Div',
    'Float.Min', 'Float.Max', 'Float.Abs', 'Float.Neg',
    'Float.Sqrt', 'Float.Square', 'Float.Tanh',
    'Float.Dup', 'Float.Swap', 'Float.Pop', 'Float.Rot',
    'Node.GetLocalDist', 'Node.GetCumDist', 'Node.GetMUp',
    'Node.GetLayer', 'Node.GetScore',
    'Float.GetNoiseVar',
    'Float.Const0', 'Float.Const1', 'Float.ConstHalf',
    'Int.Pop', 'Int.Dup',
    'Node.ForEachChild',     # iterate children (sum reduce)
    'Node.ForEachChildMin',  # iterate children (min reduce — A* heuristic)
    'Node.NumChildren',
    'Exec.DoTimes',
] + [f'Float.EvoConst{i}' for i in range(N_EVO_CONSTS)]

_BELIEF_INSTR = [
    'Float.Add', 'Float.Sub', 'Float.Mul', 'Float.Div',
    'Float.Min', 'Float.Max', 'Float.Abs',
    'Float.Sqrt', 'Float.Square',
    'Float.Dup', 'Float.Swap',
    'Node.GetCumDist', 'Node.GetLocalDist', 'Node.GetMUp', 'Node.GetMDown',
    'Node.GetLayer',
    'Float.GetNoiseVar',
    'Float.Const0', 'Float.Const1', 'Float.ConstHalf',
    'Mat.PeekAt', 'Mat.PeekAtIm', 'Vec.PeekAt', 'Vec.PeekAtIm',
    'Vec.SecondPeekAt', 'Vec.SecondPeekAtIm',
    'Vec.Dot', 'Vec.Norm2',
] + [f'Float.EvoConst{i}' for i in range(N_EVO_CONSTS)]

_HALT_INSTR = [
    'Float.Sub', 'Float.Abs', 'Float.LT', 'Float.GT',
    'Float.Dup', 'Float.Swap',
    'Float.Const0', 'Float.Const0_1', 'Float.Const1', 'Float.ConstHalf',
    'Bool.True', 'Bool.False', 'Bool.Not', 'Bool.And', 'Bool.Or',
    'Node.GetLayer', 'Int.Const0', 'Int.LT',
    'Float.GetNoiseVar',
] + [f'Float.EvoConst{i}' for i in range(N_EVO_CONSTS)]


def random_typed_program(instr_set: List[str], min_size=2, max_size=10,
                         rng=None) -> List[Instruction]:
    """Random program biased toward a specific instruction set."""
    if rng is None:
        rng = np.random.RandomState()
    size = rng.randint(min_size, max_size + 1)
    prog = []
    for _ in range(size):
        if rng.rand() < 0.15 and any(i in CONTROL_INSTRUCTIONS for i in instr_set):
            # Control instruction
            ctrl_opts = [i for i in instr_set if i in CONTROL_INSTRUCTIONS]
            if ctrl_opts:
                ctrl = rng.choice(ctrl_opts)
                inner_set = [i for i in instr_set if i not in CONTROL_INSTRUCTIONS]
                if not inner_set:
                    inner_set = instr_set
                b = [Instruction(rng.choice(inner_set))
                     for _ in range(rng.randint(1, 4))]
                prog.append(Instruction(ctrl, code_block=b))
                continue
        # Primitive (from biased set 70%, from global 30%)
        if rng.rand() < 0.7:
            prim_opts = [i for i in instr_set if i not in CONTROL_INSTRUCTIONS]
            if prim_opts:
                prog.append(Instruction(rng.choice(prim_opts)))
                continue
        prog.append(Instruction(rng.choice(PRIMITIVE_INSTRUCTIONS)))
    return prog


def random_f_belief_program(rng) -> List[Instruction]:
    """Generate a purely random F_belief program from the belief instruction set.
    No templates — validity is enforced by the perturbation check in random_genome."""
    return random_typed_program(_BELIEF_INSTR, min_size=3, max_size=12, rng=rng)


def random_f_down_program(rng) -> List[Instruction]:
    """Generate a purely random F_down program from the down instruction set.
    No templates — validity is enforced by the perturbation check in random_genome."""
    return random_typed_program(_DOWN_INSTR, min_size=1, max_size=8, rng=rng)


def random_f_up_program(rng) -> List[Instruction]:
    """Generate a purely random F_up program from the up instruction set.
    No templates — validity is enforced by the perturbation check in random_genome."""
    return random_typed_program(_UP_INSTR, min_size=3, max_size=12, rng=rng)


class Genome:
    """4-program genome for structured BP with evolvable log-domain constants."""
    def __init__(self, prog_down: List[Instruction],
                 prog_up: List[Instruction],
                 prog_belief: List[Instruction],
                 prog_halt: List[Instruction],
                 log_constants: np.ndarray = None):
        self.prog_down = prog_down
        self.prog_up = prog_up
        self.prog_belief = prog_belief
        self.prog_halt = prog_halt
        # Evolvable constants in log-domain: actual value = 10^(log_const)
        if log_constants is None:
            self.log_constants = np.zeros(N_EVO_CONSTS)
        else:
            self.log_constants = np.array(log_constants, dtype=float)

    @property
    def evo_constants(self) -> np.ndarray:
        """Actual constant values (10^log_constants), clamped for safety."""
        return np.clip(np.power(10.0, self.log_constants), 1e-6, 1e6)

    def total_length(self) -> int:
        return (program_length(self.prog_down) +
                program_length(self.prog_up) +
                program_length(self.prog_belief) +
                program_length(self.prog_halt))

    def to_oneliner(self) -> str:
        c_str = ','.join(f'{v:.4f}' for v in self.log_constants)
        return (f"DOWN:[{program_to_oneliner(self.prog_down)}] | "
                f"UP:[{program_to_oneliner(self.prog_up)}] | "
                f"BEL:[{program_to_oneliner(self.prog_belief)}] | "
                f"HALT:[{program_to_oneliner(self.prog_halt)}] | "
                f"LOGC:[{c_str}]")


def random_genome(rng) -> Genome:
    """Generate a random genome by filling each of the 4 program slots
    independently.  In each trial we build one random test context, then for
    every unfilled slot we generate a fresh candidate program and test only
    *that* slot's BP-dependency condition.  As soon as a slot's condition is
    satisfied the program is banked; we stop only when all 4 slots are filled.

    This is far more efficient than requiring all 4 conditions to hold
    simultaneously in a single genome: each slot's acceptance probability is
    independent and much higher.
    """
    Nt_test = 4  # must match has_valid_bp_dependency so peek/matrix ops fire correctly
    constellation = qam16_constellation()
    vm = MIMOPushVM(flops_max=50000, step_max=500)

    log_constants = rng.uniform(-2.0, 2.0, N_EVO_CONSTS)
    vm.evolved_constants = np.clip(np.power(10.0, log_constants), 1e-6, 1e6)

    # found[i] holds the accepted program for slot i (None = not yet filled)
    found = [None, None, None, None]
    _SETS = [_DOWN_INSTR, _UP_INSTR, _BELIEF_INSTR, _HALT_INSTR]
    _LENS = [(4, 16), (3, 12), (3, 12), (2, 6)]

    attempts = 0
    while any(s is None for s in found):
        attempts += 1
        trng = np.random.RandomState(rng.randint(0, 2**31))

        # Build one random test context shared across all slot tests this trial
        graph = _make_test_tree(trng)
        R = (np.eye(Nt_test, dtype=complex)
             + 0.1 * (trng.randn(Nt_test, Nt_test)
                      + 1j * trng.randn(Nt_test, Nt_test)))
        y_tilde = trng.randn(Nt_test) + 1j * trng.randn(Nt_test)
        nv = abs(trng.randn()) + 0.1
        root = graph.root
        child = root.children[0]

        for slot in range(4):
            if found[slot] is not None:
                continue
            lo, hi = _LENS[slot]
            # Try multiple random programs per slot per context to amortize
            # the context setup cost (pure random has low acceptance rate).
            n_tries = 20 if slot in (1, 2) else 5
            for _try in range(n_tries):
                if found[slot] is not None:
                    break
                prog = random_typed_program(_SETS[slot], lo, hi, trng)

                if slot == 0:  # F_down: depend on BOTH M_parent_down AND C_i
                    base = float(root.m_down)
                    pert_md = base + trng.randn() * 2.0 + 0.5
                    ci = float(child.local_dist)
                    ci_pert = ci + trng.randn() * 2.0 + 0.5
                    r1 = _run_test_program(
                        vm, prog, child, graph, R, y_tilde, constellation,
                        float_vals=[base, ci], noise_var=nv)
                    r2 = _run_test_program(
                        vm, prog, child, graph, R, y_tilde, constellation,
                        float_vals=[pert_md, ci], noise_var=nv)
                    md_dep = (r1 is not None and r2 is not None and abs(r1 - r2) > 1e-12)
                    orig_ld = child.local_dist
                    child.local_dist = ci_pert
                    r3 = _run_test_program(
                        vm, prog, child, graph, R, y_tilde, constellation,
                        float_vals=[base, ci_pert], noise_var=nv)
                    child.local_dist = orig_ld
                    ci_dep = (r1 is not None and r3 is not None and abs(r1 - r3) > 1e-12)
                    if md_dep and ci_dep:
                        found[0] = prog

                elif slot == 1:  # F_up: depend on children's m_up, permutation-equivariant
                    orig_ups = [c.m_up for c in root.children]
                    orig_floats = []
                    for c in root.children:
                        orig_floats.extend([c.local_dist, c.m_up])
                    r1 = _run_test_program(
                        vm, prog, root, graph, R, y_tilde, constellation,
                        float_vals=orig_floats,
                        int_vals=[len(root.children)], noise_var=nv)
                    pert_floats = []
                    for c in root.children:
                        nm = c.m_up + trng.randn() * 2.0 + 0.5
                        c.m_up = nm
                        pert_floats.extend([c.local_dist, nm])
                    r2 = _run_test_program(
                        vm, prog, root, graph, R, y_tilde, constellation,
                        float_vals=pert_floats,
                        int_vals=[len(root.children)], noise_var=nv)
                    for c, m in zip(root.children, orig_ups):
                        c.m_up = m
                    noop_top = orig_floats[-1]
                    is_noop = (r1 is not None and abs(r1 - noop_top) < 1e-12)
                    if (r1 is not None and r2 is not None
                            and abs(r1 - r2) > 1e-12 and not is_noop):
                        perm_ok = True
                        for _pt in range(2):
                            perm = list(range(len(root.children)))
                            trng.shuffle(perm)
                            if perm == list(range(len(root.children))):
                                continue
                            shuf_floats = []
                            for idx in perm:
                                c = root.children[idx]
                                shuf_floats.extend([c.local_dist, c.m_up])
                            r_shuf = _run_test_program(
                                vm, prog, root, graph, R, y_tilde, constellation,
                                float_vals=shuf_floats,
                                int_vals=[len(root.children)], noise_var=nv)
                            if r_shuf is None or abs(r1 - r_shuf) > 1e-10:
                                perm_ok = False
                                break
                        if perm_ok:
                            found[1] = prog

                elif slot == 2:  # F_belief: depend on cum_dist AND m_down AND m_up
                    cd = float(child.cum_dist)
                    md = float(child.m_down)
                    mu = float(child.m_up)
                    r_base = _run_test_program(
                        vm, prog, child, graph, R, y_tilde, constellation,
                        float_vals=[cd, md, mu], noise_var=nv)
                    cd_pert = cd + trng.randn() * 2.0 + 0.5
                    r_cd_stack = _run_test_program(
                        vm, prog, child, graph, R, y_tilde, constellation,
                        float_vals=[cd_pert, md, mu], noise_var=nv)
                    cd_stack_ok = (r_base is not None and r_cd_stack is not None
                                   and abs(r_base - r_cd_stack) > 1e-12)
                    orig_cd = child.cum_dist
                    child.cum_dist = cd_pert
                    r_cd_node = _run_test_program(
                        vm, prog, child, graph, R, y_tilde, constellation,
                        float_vals=[cd, md, mu], noise_var=nv)
                    child.cum_dist = orig_cd
                    cd_node_ok = (r_base is not None and r_cd_node is not None
                                  and abs(r_base - r_cd_node) > 1e-12)
                    cd_ok = cd_stack_ok or cd_node_ok
                    r_md = _run_test_program(
                        vm, prog, child, graph, R, y_tilde, constellation,
                        float_vals=[cd, md + trng.randn() * 2.0 + 0.5, mu],
                        noise_var=nv)
                    md_ok = (r_base is not None and r_md is not None
                             and abs(r_base - r_md) > 1e-12)
                    orig_mu = child.m_up
                    child.m_up = mu + trng.randn() * 2.0 + 0.5
                    r_mu_node = _run_test_program(
                        vm, prog, child, graph, R, y_tilde, constellation,
                        float_vals=[cd, md, mu], noise_var=nv)
                    child.m_up = orig_mu
                    mu_ok = (r_base is not None and r_mu_node is not None
                             and abs(r_base - r_mu_node) > 1e-12)
                    mu_pert_val = mu + trng.randn() * 2.0 + 0.5
                    r_mu_stack = _run_test_program(
                        vm, prog, child, graph, R, y_tilde, constellation,
                        float_vals=[cd, md, mu_pert_val], noise_var=nv)
                    mu_stack_ok = (r_base is not None and r_mu_stack is not None
                                   and abs(r_base - r_mu_stack) > 1e-12
                                   and abs(r_mu_stack - mu_pert_val) > 1e-12)
                    mu_ok_combined = mu_ok or mu_stack_ok
                    orig_md_node = child.m_down
                    child.m_down = md + trng.randn() * 2.0 + 0.5
                    r_md_node = _run_test_program(
                        vm, prog, child, graph, R, y_tilde, constellation,
                        float_vals=[cd, md, mu], noise_var=nv)
                    child.m_down = orig_md_node
                    md_node_ok = (r_base is not None and r_md_node is not None
                                  and abs(r_base - r_md_node) > 1e-12)
                    md_ok_combined = md_ok or md_node_ok
                    if cd_ok and md_ok_combined and mu_ok_combined:
                        found[2] = prog

                else:  # H_halt: depend on BOTH old_m_up and new_m_up
                    halt_results: set = set()
                    old_dep = False
                    new_dep = False
                    for _ in range(10):
                        old_m = trng.randn() * 3.0
                        new_m = old_m + trng.randn() * 2.0
                        r = _run_test_program(
                            vm, prog, child, graph, R, y_tilde, constellation,
                            float_vals=[old_m, new_m], noise_var=nv, get_bool=True)
                        if r is not None:
                            halt_results.add(r)
                            r_old = _run_test_program(
                                vm, prog, child, graph, R, y_tilde, constellation,
                                float_vals=[old_m + 5.0, new_m], noise_var=nv, get_bool=True)
                            if r_old is not None and r_old != r:
                                old_dep = True
                            r_new = _run_test_program(
                                vm, prog, child, graph, R, y_tilde, constellation,
                                float_vals=[old_m, new_m + 5.0], noise_var=nv, get_bool=True)
                            if r_new is not None and r_new != r:
                                new_dep = True
                        if len(halt_results) >= 2 and old_dep and new_dep:
                            break
                    if len(halt_results) >= 2 and old_dep and new_dep:
                        found[3] = prog

        if attempts % 50 == 0:
            status = [('DOWN' if found[0] else '____'),
                      ('UP  ' if found[1] else '____'),
                      ('BEL ' if found[2] else '____'),
                      ('HALT' if found[3] else '____')]
            print(f"[random_genome] {attempts} trials  filled: {status}",
                  flush=True)

    return Genome(
        prog_down=found[0],
        prog_up=found[1],
        prog_belief=found[2],
        prog_halt=found[3],
        log_constants=log_constants,
    )


def deep_copy_genome(g: Genome) -> Genome:
    return Genome(
        prog_down=deep_copy_program(g.prog_down),
        prog_up=deep_copy_program(g.prog_up),
        prog_belief=deep_copy_program(g.prog_belief),
        prog_halt=deep_copy_program(g.prog_halt),
        log_constants=g.log_constants.copy(),
    )


def mutate_genome(g: Genome, rng, n_mutations=1, max_attempts=200) -> Genome:
    for attempt in range(max_attempts):
        gm = deep_copy_genome(g)
        # With probability 0.15, replace the entire F_belief with a fresh
        # diverse program. This fights premature convergence to a single
        # belief family (observed: all genomes converge to D_i/M_down/M_up).
        belief_replaced = False
        if rng.rand() < 0.15:
            new_bel = random_typed_program(_BELIEF_INSTR, min_size=3, max_size=12, rng=rng)
            if new_bel:
                gm.prog_belief = new_bel
                belief_replaced = True
        programs = [
            (gm.prog_down, _DOWN_INSTR),
            (gm.prog_up, _UP_INSTR),
            (gm.prog_belief, _BELIEF_INSTR),
            (gm.prog_halt, _HALT_INSTR),
        ]
        # More aggressive mutation when attempts are failing
        actual_mutations = n_mutations + (attempt // 50)
        for _ in range(actual_mutations):
            # With probability 0.2, mutate a constant instead
            if rng.rand() < 0.2:
                ci = rng.randint(N_EVO_CONSTS)
                gm.log_constants[ci] += rng.randn() * 0.3
                gm.log_constants[ci] = np.clip(gm.log_constants[ci], -3.0, 3.0)
                continue
            idx = rng.randint(0, 4)
            prog, instr_set = programs[idx]
            op = rng.randint(0, 4)
            if op == 0 and prog:
                i = rng.randint(len(prog))
                prog[i] = Instruction(rng.choice(instr_set))
            elif op == 1 and len(prog) < 30:
                pos = rng.randint(0, len(prog) + 1)
                prog.insert(pos, Instruction(rng.choice(instr_set)))
            elif op == 2 and len(prog) > 2:
                prog.pop(rng.randint(len(prog)))
            else:
                if len(prog) >= 2:
                    i, j = rng.choice(len(prog), 2, replace=False)
                    prog[i], prog[j] = prog[j], prog[i]
            if idx == 0:
                gm.prog_down = prog
            elif idx == 1:
                gm.prog_up = prog
            elif idx == 2:
                gm.prog_belief = prog
            else:
                gm.prog_halt = prog
        if _has_valid_bp_syntax(gm):
            return gm
    # Fallback: generate a fresh valid genome instead of returning unmutated copy.
    # Returning the parent unchanged was a major stagnation source.
    return random_genome(rng)


def crossover_genome(g1: Genome, g2: Genome, rng, max_attempts=200) -> Genome:
    """Per-program crossover with constant blending."""
    def _cross(p1, p2):
        a = deep_copy_program(p1)
        b = deep_copy_program(p2)
        if not a or not b:
            return a or b
        c1 = rng.randint(0, len(a))
        c2 = rng.randint(0, len(b))
        child = a[:c1] + b[c2:]
        if len(child) > 40:
            child = child[:20]
        return child if child else a

    for _ in range(max_attempts):
        # Blend constants (random interpolation)
        alpha = rng.rand(N_EVO_CONSTS)
        blended_consts = alpha * g1.log_constants + (1 - alpha) * g2.log_constants
        child = Genome(
            prog_down=_cross(g1.prog_down, g2.prog_down),
            prog_up=_cross(g1.prog_up, g2.prog_up),
            prog_belief=_cross(g1.prog_belief, g2.prog_belief),
            prog_halt=_cross(g1.prog_halt, g2.prog_halt),
            log_constants=blended_consts,
        )
        if _has_valid_bp_syntax(child):
            return child
    # Crossover failed: fall back to a fresh valid random genome (never invalid).
    return random_genome(rng)


# --------------------------------------------------------------------------
# BP Dependency Verification — perturbation-based data-flow analysis
# --------------------------------------------------------------------------
# Instead of checking syntactic presence of instructions, we verify that
# each program's output actually depends on the BP-relevant inputs by
# running the program with original inputs, then with perturbed inputs,
# and checking if the output changes.  Programs with dead-code BP
# instructions are rejected.
# --------------------------------------------------------------------------


def _has_valid_bp_syntax(genome) -> bool:
    """Fast syntactic check: verify critical instructions are PRESENT in each
    program.  O(N) where N is total instruction count — no VM execution.
    
    Used in mutate_genome/crossover_genome instead of the expensive
    perturbation-based has_valid_bp_dependency to avoid being a bottleneck.
    """
    def _names(prog):
        s = set()
        for instr in prog:
            s.add(instr.name)
            if hasattr(instr, 'code_block') and instr.code_block:
                for inner in instr.code_block:
                    s.add(inner.name)
        return s

    # F_down: must contain Node.GetLocalDist (C_i dependency)
    down_names = _names(genome.prog_down)
    if 'Node.GetLocalDist' not in down_names:
        return False

    # F_up: must contain ForEachChild or ForEachChildMin AND Node.GetMUp
    up_names = _names(genome.prog_up)
    if not (('Node.ForEachChild' in up_names or 'Node.ForEachChildMin' in up_names)
            and 'Node.GetMUp' in up_names):
        return False

    # F_belief: must contain BOTH Node.GetMDown AND Node.GetMUp
    bel_names = _names(genome.prog_belief)
    if 'Node.GetMDown' not in bel_names or 'Node.GetMUp' not in bel_names:
        return False

    # H_halt: must have at least 2 instructions (not trivially constant)
    if len(genome.prog_halt) < 2:
        return False

    return True


def _make_test_tree(rng, n_children=16, Nt_test=4):
    """Build a search tree for perturbation tests.
    
    n_children: number of children per node.  Should match M (constellation
    size) so that F_up tests operate on the same stack depth as real decoding.
    For 16-QAM this is 16.
    Nt_test: system dimension.  Root layer = Nt_test, children at Nt_test-1.
             Must match the R matrix dimension so GetMMSELB/GetResidue fire.
    """
    graph = SearchTreeGraph()
    root = graph.create_root(layer=Nt_test)
    root.m_down = rng.randn()
    root.m_up = rng.randn()
    root.local_dist = abs(rng.randn())
    root.cum_dist = abs(rng.randn())
    root.partial_symbols = np.array([], dtype=complex)

    for c_i in range(n_children):
        sym = (rng.randn() + 1j * rng.randn())
        # Use distinct spread: multiply by (c_i+1) to ensure values are well-separated
        ld = abs(rng.randn()) * (c_i + 1) * 0.3 + 0.1
        cd = root.cum_dist + ld
        ps = np.array([sym], dtype=complex)
        child = graph.add_child(root, layer=Nt_test - 1, symbol=sym,
                                local_dist=ld, cum_dist=cd,
                                partial_symbols=ps)
        child.m_up = rng.randn() * (c_i + 1) * 0.5
        child.m_down = root.m_down + ld
        # Add grandchildren (limit to first 3 children to avoid explosion)
        if c_i < 3:
            for gc_i in range(2):
                sym2 = (rng.randn() + 1j * rng.randn())
                ld2 = abs(rng.randn()) + 0.1
                gc = graph.add_child(child, layer=Nt_test - 2, symbol=sym2,
                                     local_dist=ld2, cum_dist=cd + ld2,
                                     partial_symbols=np.array([sym, sym2], dtype=complex))
                gc.m_up = rng.randn()
                gc.m_down = child.m_down + ld2
    return graph


def _run_test_program(vm, prog, node, graph, R, y_tilde, constellation,
                      float_vals, int_vals=None, noise_var=1.0,
                      get_bool=False):
    """Set up VM like bp_decoder_v2 does and run a program."""
    vm.reset()
    vm.candidate_node = node
    vm.constellation = constellation
    vm.noise_var = noise_var
    vm.matrix_stack.push(R.copy())
    vm.vector_stack.push(y_tilde.copy())
    vm.vector_stack.push(node.partial_symbols.copy())
    vm.graph_stack.push(graph)
    vm.node_stack.push(node)
    vm.int_stack.push(node.layer)
    for v in float_vals:
        vm.float_stack.push(float(v))
    if int_vals:
        for v in int_vals:
            vm.int_stack.push(v)
    try:
        vm._execute_block(prog)
    except Exception:
        pass
    if get_bool:
        r = vm.bool_stack.peek()
        return bool(r) if r is not None else None
    r = vm.float_stack.peek()
    return float(r) if (r is not None and np.isfinite(r)) else None


def has_valid_bp_dependency(genome: Genome, rng, n_trials=5) -> bool:
    """Perturbation-based check: does each program's output actually depend
    on its BP-relevant inputs?

    Evidence is accumulated ACROSS trials — each condition only needs to be
    satisfied in at least one trial (not all at once in the same trial).
    Returns True only when ALL of:
      - F_down output changes when M_parent_down is perturbed
      - F_up output changes when children's m_up values are perturbed
      - F_belief output changes when m_down AND/OR m_up is perturbed
      - H_halt bool output varies across different (old_m_up, new_m_up) pairs
    """
    # IMPORTANT: Nt_test must be >= max layer in test tree (4) so that
    # GetMMSELB / GetResidue fire correctly.  With Nt_test=2 they become
    # NOPs (k > R.cols), which changes stack layout and causes spurious
    # dependency positives.
    Nt_test, Nr_test = 4, 8
    constellation = qam16_constellation()
    vm = MIMOPushVM(flops_max=50000, step_max=500)
    vm.evolved_constants = genome.evo_constants

    down_ok = False
    up_ok = False
    belief_ok = False
    halt_ok = False
    halt_results: set = set()

    for trial in range(n_trials):
        trial_rng = np.random.RandomState(rng.randint(0, 2**31) + trial)
        graph = _make_test_tree(trial_rng)
        R = np.eye(Nt_test, dtype=complex) + 0.1 * (
            trial_rng.randn(Nt_test, Nt_test) + 1j * trial_rng.randn(Nt_test, Nt_test))
        y_tilde = trial_rng.randn(Nt_test) + 1j * trial_rng.randn(Nt_test)
        nv = abs(trial_rng.randn()) + 0.1
        root = graph.root
        child = root.children[0]

        # --- F_down: perturb BOTH M_parent_down AND C_i (local_dist) ---
        if not down_ok:
            base_m_down = float(root.m_down)
            perturbed_m_down = base_m_down + trial_rng.randn() * 2.0 + 0.5
            ci_base = float(child.local_dist)
            ci_pert = ci_base + trial_rng.randn() * 2.0 + 0.5
            r1 = _run_test_program(vm, genome.prog_down, child, graph, R, y_tilde,
                                   constellation,
                                   float_vals=[base_m_down, ci_base],
                                   noise_var=nv)
            r2 = _run_test_program(vm, genome.prog_down, child, graph, R, y_tilde,
                                   constellation,
                                   float_vals=[perturbed_m_down, ci_base],
                                   noise_var=nv)
            md_dep = (r1 is not None and r2 is not None and abs(r1 - r2) > 1e-12)
            # Perturb C_i via BOTH float_vals AND node property
            orig_ld = child.local_dist
            child.local_dist = ci_pert
            r3 = _run_test_program(vm, genome.prog_down, child, graph, R, y_tilde,
                                   constellation,
                                   float_vals=[base_m_down, ci_pert],
                                   noise_var=nv)
            child.local_dist = orig_ld
            ci_dep = (r1 is not None and r3 is not None and abs(r1 - r3) > 1e-12)
            if md_dep and ci_dep:
                down_ok = True

        # --- F_up: perturb children's m_up + permutation equivariance ---
        if not up_ok:
            node_for_up = root
            orig_m_ups = [c.m_up for c in node_for_up.children]
            orig_floats = []
            for c in node_for_up.children:
                orig_floats.extend([c.local_dist, c.m_up])

            r1 = _run_test_program(vm, genome.prog_up, node_for_up, graph,
                                   R, y_tilde, constellation,
                                   float_vals=orig_floats,
                                   int_vals=[len(node_for_up.children)],
                                   noise_var=nv)

            # Perturb all children's m_up (both float_vals and node properties)
            perturbed_floats = []
            for c in node_for_up.children:
                new_m = c.m_up + trial_rng.randn() * 2.0 + 0.5
                c.m_up = new_m
                perturbed_floats.extend([c.local_dist, new_m])

            r2 = _run_test_program(vm, genome.prog_up, node_for_up, graph,
                                   R, y_tilde, constellation,
                                   float_vals=perturbed_floats,
                                   int_vals=[len(node_for_up.children)],
                                   noise_var=nv)

            for c, m in zip(node_for_up.children, orig_m_ups):
                c.m_up = m

            # Noop check: reject if output == raw float stack top (last child's m_up)
            noop_top = orig_floats[-1] if orig_floats else None
            is_noop = (r1 is not None and noop_top is not None
                       and abs(r1 - noop_top) < 1e-12)
            if (r1 is not None and r2 is not None
                    and abs(r1 - r2) > 1e-12 and not is_noop):
                # Permutation equivariance: multiple shuffle trials.
                # ALL must produce the same output as original order.
                perm_ok = True
                for _pt in range(2):  # 2 trials sufficient for equivariance check
                    perm = list(range(len(node_for_up.children)))
                    trial_rng.shuffle(perm)
                    if perm == list(range(len(node_for_up.children))):
                        continue  # identity permutation, skip
                    shuffled_floats = []
                    for idx in perm:
                        c = node_for_up.children[idx]
                        shuffled_floats.extend([c.local_dist, c.m_up])
                    r_shuf = _run_test_program(
                        vm, genome.prog_up, node_for_up, graph,
                        R, y_tilde, constellation,
                        float_vals=shuffled_floats,
                        int_vals=[len(node_for_up.children)],
                        noise_var=nv)
                    if r_shuf is None or abs(r1 - r_shuf) > 1e-10:
                        perm_ok = False
                        break
                if perm_ok:
                    up_ok = True

        # --- F_belief: must depend on ALL THREE: cum_dist AND m_down AND m_up.
        #     This prevents trivial formulas like min(D_i, M_up) that ignore
        #     downward BP messages entirely.  Genuine BP requires information
        #     flow in BOTH directions.
        #     m_down is tested via both float_vals and node attribute.
        #     m_up is tested via node attribute (Node.GetMUp) and float_vals
        #     with anti-noop guard (output must not equal perturbed input).
        if not belief_ok:
            cd = float(child.cum_dist)
            md = float(child.m_down)
            mu = float(child.m_up)
            r_base = _run_test_program(vm, genome.prog_belief, child, graph,
                                       R, y_tilde, constellation,
                                       float_vals=[cd, md, mu],
                                       noise_var=nv)
            # Perturbation 0a: change cum_dist via float_vals (stack access)
            cd_pert = cd + trial_rng.randn() * 2.0 + 0.5
            r_cd_stack = _run_test_program(vm, genome.prog_belief, child, graph,
                                           R, y_tilde, constellation,
                                           float_vals=[cd_pert, md, mu],
                                           noise_var=nv)
            cd_stack_ok = (r_base is not None and r_cd_stack is not None and
                           abs(r_base - r_cd_stack) > 1e-12)
            # Perturbation 0b: change node.cum_dist property (Node.GetCumDist)
            orig_node_cd = child.cum_dist
            child.cum_dist = cd_pert
            r_cd_node = _run_test_program(vm, genome.prog_belief, child, graph,
                                          R, y_tilde, constellation,
                                          float_vals=[cd, md, mu],
                                          noise_var=nv)
            child.cum_dist = orig_node_cd  # restore
            cd_node_ok = (r_base is not None and r_cd_node is not None and
                          abs(r_base - r_cd_node) > 1e-12)
            cd_changed = cd_stack_ok or cd_node_ok
            # Perturbation 1a: change m_down in float_vals (stack position 2nd from top)
            md_pert = md + trial_rng.randn() * 2.0 + 0.5
            r_md_stack = _run_test_program(vm, genome.prog_belief, child, graph,
                                           R, y_tilde, constellation,
                                           float_vals=[cd, md_pert, mu],
                                           noise_var=nv)
            md_stack_ok = (r_base is not None and r_md_stack is not None and
                           abs(r_base - r_md_stack) > 1e-12)
            # Perturbation 1b: change node.m_down property (Node.GetMDown)
            orig_node_md = child.m_down
            child.m_down = md_pert
            r_md_node = _run_test_program(vm, genome.prog_belief, child, graph,
                                          R, y_tilde, constellation,
                                          float_vals=[cd, md, mu],
                                          noise_var=nv)
            child.m_down = orig_node_md  # restore
            md_node_ok = (r_base is not None and r_md_node is not None and
                          abs(r_base - r_md_node) > 1e-12)
            md_changed = md_stack_ok or md_node_ok
            # Perturbation 2a: change node.m_up property only (float_vals unchanged).
            # Programs accessing m_up via Node.GetMUp instruction will react;
            # a passthrough program reads mu from float_vals (unchanged) → no change.
            orig_node_mu = child.m_up
            mu_pert_val = mu + trial_rng.randn() * 2.0 + 0.5
            child.m_up = mu_pert_val
            r_mu_node = _run_test_program(vm, genome.prog_belief, child, graph,
                                          R, y_tilde, constellation,
                                          float_vals=[cd, md, mu],   # original!
                                          noise_var=nv)
            child.m_up = orig_node_mu  # restore
            mu_node_ok = (r_base is not None and r_mu_node is not None and
                          abs(r_base - r_mu_node) > 1e-12)
            # Perturbation 2b: change m_up via float_vals (top of stack)
            # with anti-noop guard: output must not equal the perturbed value itself.
            r_mu_stack = _run_test_program(vm, genome.prog_belief, child, graph,
                                           R, y_tilde, constellation,
                                           float_vals=[cd, md, mu_pert_val],
                                           noise_var=nv)
            mu_stack_ok = (r_base is not None and r_mu_stack is not None and
                           abs(r_base - r_mu_stack) > 1e-12 and
                           abs(r_mu_stack - mu_pert_val) > 1e-12)  # anti-noop
            mu_changed = mu_node_ok or mu_stack_ok
            # ALL THREE must be satisfied: genuine BP requires both directions
            if cd_changed and md_changed and mu_changed:
                belief_ok = True

        # --- H_halt: must depend on BOTH old_m_up and new_m_up ---
        # Not just variable output, but genuinely comparing them (e.g. convergence check)
        if not halt_ok:
            old_dep_seen = False
            new_dep_seen = False
            for _ in range(8):
                old_m = trial_rng.randn() * 3.0
                new_m = old_m + trial_rng.randn() * 2.0
                r_base = _run_test_program(vm, genome.prog_halt, child, graph,
                                           R, y_tilde, constellation,
                                           float_vals=[old_m, new_m],
                                           noise_var=nv, get_bool=True)
                if r_base is None:
                    continue
                halt_results.add(r_base)
                # Perturb old_m_up only
                r_old = _run_test_program(vm, genome.prog_halt, child, graph,
                                          R, y_tilde, constellation,
                                          float_vals=[old_m + 5.0, new_m],
                                          noise_var=nv, get_bool=True)
                if r_old is not None and r_old != r_base:
                    old_dep_seen = True
                # Perturb new_m_up only
                r_new = _run_test_program(vm, genome.prog_halt, child, graph,
                                          R, y_tilde, constellation,
                                          float_vals=[old_m, new_m + 5.0],
                                          noise_var=nv, get_bool=True)
                if r_new is not None and r_new != r_base:
                    new_dep_seen = True
            if len(halt_results) >= 2 and old_dep_seen and new_dep_seen:
                halt_ok = True
        if len(halt_results) >= 2:
            halt_ok = True

        # Early exit once all conditions satisfied
        if down_ok and up_ok and belief_ok and halt_ok:
            return True

    return False


class GenomeIndividual:
    def __init__(self, genome: Genome, fitness=None):
        self.genome = genome
        self.fitness = fitness
        self.per_sample_bers = None
        self.age = 0


# --------------------------------------------------------------------------
# Logger
# --------------------------------------------------------------------------

class Logger:
    def __init__(self, path):
        self.path = path
        ensure_dir(os.path.dirname(path))
        self._w("=" * 80 + f"\nStructured-BP GP  started {ts()}\n" + "=" * 80 + "\n")

    def _w(self, s):
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(s)

    def gen(self, g, fit, genome, note=""):
        self._w(f"\n[{ts()}] Gen {g}  {fit}")
        if note:
            self._w(f"  ({note})")
        self._w(f"\n  {genome.to_oneliner()}\n")

    def info(self, title, body):
        self._w(f"\n[{ts()}] {title}\n{body}\n" + "-" * 60 + "\n")


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------

class StructuredBPEvaluator:
    def __init__(self, Nt=8, Nr=16, mod_order=16,
                 flops_max=2_000_000, max_nodes=500,
                 train_samples=16, snr_choices=None,
                 step_max=2000, use_cpp=False):
        self.Nt = Nt
        self.Nr = Nr
        self.mod_order = mod_order
        self.constellation = constellation_for(mod_order)
        self.train_samples = train_samples
        self.snr_choices = snr_choices or [10.0, 12.0, 14.0]
        self.max_nodes = max_nodes
        self.use_cpp = use_cpp
        self.cpp_eval = None

        self.vm = MIMOPushVM(flops_max=flops_max, step_max=step_max)
        self.decoder = StructuredBPDecoder(
            Nt=Nt, Nr=Nr, constellation=self.constellation,
            max_nodes=max_nodes, vm=self.vm)

        # Smaller-budget challenge decoder (1/2 budget for gen gap check)
        self.challenge_decoder = StructuredBPDecoder(
            Nt=Nt, Nr=Nr, constellation=self.constellation,
            max_nodes=max(128, max_nodes // 2),
            vm=MIMOPushVM(flops_max=flops_max, step_max=step_max))

        if use_cpp:
            try:
                from cpp_bridge import CppBPEvaluator
                self.cpp_eval = CppBPEvaluator(
                    Nt=Nt, Nr=Nr, mod_order=mod_order,
                    max_nodes=max_nodes, flops_max=flops_max,
                    step_max=step_max, max_bp_iters=3)
                print("[C++] BP evaluator loaded successfully")
            except Exception as e:
                print(f"[C++] Failed to load BP evaluator: {e}")
                print("[C++] Falling back to Python evaluator")
                self.use_cpp = False

    def build_dataset(self, seed, n=None, snrs=None):
        rng = np.random.RandomState(seed)
        ds = []
        n = n or self.train_samples
        snrs = snrs or self.snr_choices
        for _ in range(n):
            snr = float(rng.choice(snrs))
            ds.append(generate_mimo_sample(self.Nr, self.Nt,
                                           self.constellation, snr, rng))
        return ds

    def _eval_per_sample(self, genome: Genome, dataset, decoder):
        # C++ fast path
        if self.use_cpp and self.cpp_eval is not None:
            avg_ber, avg_flops, faults, avg_bp = self.cpp_eval.evaluate_genome(
                genome, dataset)
            n = len(dataset)
            bers = [avg_ber] * n  # approximation for per-sample
            flops_list = [avg_flops] * n
            bp_up_total = int(avg_bp * n)
            return bers, flops_list, faults, bp_up_total

        # Python fallback — set evolved constants on VM
        evo_consts = genome.evo_constants
        decoder.vm.evolved_constants = evo_consts
        bers, flops_list, faults = [], [], 0
        bp_up_total = 0
        for H, x_true, y, nv in dataset:
            try:
                x_hat, fl = decoder.detect(
                    H, y,
                    prog_down=genome.prog_down,
                    prog_up=genome.prog_up,
                    prog_belief=genome.prog_belief,
                    prog_halt=genome.prog_halt,
                    noise_var=float(nv))
                bers.append(ber_calc(x_true, x_hat))
                flops_list.append(float(fl))
                # StructuredBPDecoder.total_bp_calls counts every F_up/F_down/F_belief
                # invocation across the entire tree in the last detection call.
                # NOTE: previously this line read decoder.bp_up_calls (v1 attribute
                # name), causing AttributeError after detect() succeeded.  That bug
                # made each sample produce TWO BER entries (one correct + one 1.0
                # fault), so reported training BER ≈ (true_BER + 1.0) / 2 ≈ 0.55,
                # while full_evaluation() (which never reads this field) showed the
                # real BER correctly.  Fixed by using total_bp_calls.
                bp_up_total += decoder.total_bp_calls
            except Exception:
                bers.append(1.0)
                flops_list.append(float(decoder.vm.flops_max * 10))
                faults += 1
        return bers, flops_list, faults, bp_up_total

    def evaluate(self, genome: Genome, ds_main, ds_hold):
        bers_m, fl_m, f1, bp1 = self._eval_per_sample(
            genome, ds_main, self.decoder)
        bers_h, fl_h, f2, bp2 = self._eval_per_sample(
            genome, ds_hold, self.challenge_decoder)

        # LMMSE baseline
        if self.use_cpp and self.cpp_eval is not None:
            baseline_ber, _, _ = self.cpp_eval.evaluate_baselines(ds_main)
        else:
            lmmse_bers = []
            for H, x_true, y, nv in ds_main:
                xl, _ = lmmse_detect(H, y, nv, self.constellation)
                lmmse_bers.append(ber_calc(x_true, xl))
            baseline_ber = float(np.mean(lmmse_bers)) if lmmse_bers else 1.0

        ber1 = float(np.mean(bers_m))
        ber2 = float(np.mean(bers_h))

        # ── Fitness metric glossary ─────────────────────────────────────────
        # avg_ber   : weighted average BER across main (55%) and hold (45%)
        #             datasets.  Lower is better.  This is the PRIMARY signal.
        # avg_fl    : weighted average total FLOPs per detection call.  Faulted
        #             calls contribute flops_max*10 as a heavy penalty.
        # frac_f    : fraction of detection calls that raised an exception
        #             (weighted same as BER).  Penalised 500× in composite_score.
        #             IMPORTANT: any Python exception inside detect() counts —
        #             including AttributeError, numpy overflow, etc.  A high
        #             value (e.g. 0.78) usually means a systematic bug (wrong
        #             attribute name, type error) rather than real step-limit
        #             violations.  Should be ~0 for well-formed genomes.
        # bp        : avg total BP VM calls (F_up + F_down + F_belief invocations)
        #             per sample.  Reflects how much BP work was done.  Not used
        #             directly in composite_score — informational only.
        # nlbp      : nonlocal_bp_updates — ALWAYS 0.0 in v2.  Was used in v1 to
        #             count Node.SetScore writes to nodes other than the candidate
        #             (the signature of genuine belief propagation).  Not tracked
        #             in v2 because the BP structure is enforced by the framework,
        #             not by SetScore calls.  Safe to ignore.
        # bpg       : bp_gain — ALWAYS 0.0 in v2.  Was the measured BER improvement
        #             of BP over a no-write ablation in v1.  Not computed in v2.
        #             Safe to ignore.
        # gap       : generalization_gap = |ber_main − ber_hold|.  Measures
        #             overfitting to the main dataset.  Penalised 2× in
        #             composite_score.  Low gap = good generalisation.
        # ratio     : avg_ber / lmmse_ber.  <1 means beating LMMSE, >1 means
        #             worse.  Informational, not used in composite_score.
        # ────────────────────────────────────────────────────────────────────
        avg_ber = 0.55 * ber1 + 0.45 * ber2
        avg_fl = 0.55 * float(np.mean(fl_m)) + 0.45 * float(np.mean(fl_h))
        frac_f = (0.55 * f1 + 0.45 * f2) / max(1, len(ds_main))
        gap = abs(ber1 - ber2)
        ratio = avg_ber / max(baseline_ber, 1e-6)
        n_samples = len(ds_main) + len(ds_hold)
        avg_bp = (bp1 + bp2) / max(1, n_samples)

        return FitnessResult(
            ber=avg_ber, mse=0.0, avg_flops=avg_fl,
            code_length=genome.total_length(),
            frac_faults=frac_f, baseline_ber=baseline_ber,
            ber_ratio=ratio, generalization_gap=gap,
            bp_updates=avg_bp,
            nonlocal_bp_updates=0.0,  # not tracked in v2 (see glossary)
            bp_gain=0.0), bers_m      # not tracked in v2 (see glossary)


# --------------------------------------------------------------------------
# Evolution Engine
# --------------------------------------------------------------------------

class StructuredBPEngine:
    def __init__(self, pop_size=100, tournament_size=5, elitism=6,
                 mutation_rate=0.75, crossover_rate=0.25, seed=0,
                 evaluator: StructuredBPEvaluator = None,
                 fresh_injection_rate=0.15):
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.RandomState(seed)
        self.evaluator = evaluator or StructuredBPEvaluator()
        self.fresh_injection_rate = fresh_injection_rate
        self._hall_of_fame: List[Genome] = []
        # Stagnation tracking
        self._best_ber = float('inf')
        self._stagnant_gens = 0
        self._stagnation_threshold = 4  # gens without improvement before boost

    def init_population(self) -> List[GenomeIndividual]:
        pop = []
        # Inject seed genomes (+ mutations of them) if available
        seeds = getattr(self, '_seed_genomes', [])
        for sg in seeds:
            pop.append(GenomeIndividual(deep_copy_genome(sg)))
            # Add mutations of seed genome for diversity
            for _ in range(min(10, self.pop_size // 5)):
                m = mutate_genome(deep_copy_genome(sg), self.rng,
                                  n_mutations=self.rng.randint(1, 4))
                pop.append(GenomeIndividual(m))
        # Fill remaining with random genomes
        while len(pop) < self.pop_size:
            pop.append(GenomeIndividual(random_genome(self.rng)))
        return pop[:self.pop_size]

    def evaluate_pop(self, pop: List[GenomeIndividual], seed: int):
        epoch_seed = 7000 + (seed // 3) * 3
        ds = self.evaluator.build_dataset(epoch_seed)
        ds_hold = self.evaluator.build_dataset(
            epoch_seed + 10000,
            n=max(3, self.evaluator.train_samples // 2))

        to_eval = [ind for ind in pop if ind.fitness is None]
        if not to_eval:
            return

        # ── C++ batch fast path (OpenMP-parallel across genomes) ──────────
        if (self.evaluator.use_cpp and self.evaluator.cpp_eval is not None
                and len(to_eval) >= 2):
            cpp = self.evaluator.cpp_eval
            genomes = [ind.genome for ind in to_eval]

            # Batch evaluate on main and hold datasets
            ber_m, fl_m, faults_m, bp_m = cpp.evaluate_batch(genomes, ds)
            ber_h, fl_h, faults_h, bp_h = cpp.evaluate_batch(genomes, ds_hold)

            # LMMSE baseline (computed once for the shared dataset)
            baseline_ber, _, _ = cpp.evaluate_baselines(ds)

            n_main = len(ds)
            n_hold = len(ds_hold)
            n_total = n_main + n_hold

            for i, ind in enumerate(to_eval):
                avg_ber = 0.55 * ber_m[i] + 0.45 * ber_h[i]
                avg_fl = 0.55 * fl_m[i] + 0.45 * fl_h[i]
                frac_f = (0.55 * faults_m[i] + 0.45 * faults_h[i]) / max(1, n_main)
                gap = abs(ber_m[i] - ber_h[i])
                ratio = avg_ber / max(baseline_ber, 1e-6)
                avg_bp = (bp_m[i] * n_main + bp_h[i] * n_hold) / max(1, n_total)

                fit = FitnessResult(
                    ber=avg_ber, mse=0.0, avg_flops=avg_fl,
                    code_length=ind.genome.total_length(),
                    frac_faults=frac_f, baseline_ber=baseline_ber,
                    ber_ratio=ratio, generalization_gap=gap,
                    bp_updates=avg_bp,
                    nonlocal_bp_updates=0.0,
                    bp_gain=0.0)
                ind.fitness = fit
                ind.per_sample_bers = [ber_m[i]] * n_main  # approx per-sample
            return

        # ── Sequential fallback ───────────────────────────────────────────
        for ind in tqdm(to_eval, desc="Eval", leave=False,
                        disable=len(to_eval) < 4, file=sys.stderr):
            fit, per_sample = self.evaluator.evaluate(
                ind.genome, ds, ds_hold)
            ind.fitness = fit
            ind.per_sample_bers = per_sample

    def next_gen(self, pop: List[GenomeIndividual]) -> List[GenomeIndividual]:
        ranked = sorted(pop, key=lambda x: x.fitness.composite_score()
                        if x.fitness else 1e9)
        nxt = []

        # Stagnation detection: track BER improvement
        cur_ber = ranked[0].fitness.ber if ranked[0].fitness else float('inf')
        if cur_ber < self._best_ber - 1e-5:
            self._best_ber = cur_ber
            self._stagnant_gens = 0
        else:
            self._stagnant_gens += 1

        stagnant = self._stagnant_gens >= self._stagnation_threshold
        hard_restart = self._stagnant_gens >= 8  # reduced from 12 → 8
        if stagnant:
            print(f"  [STAGNATION] {self._stagnant_gens} gens without improvement — "
                  f"{'HARD RESTART' if hard_restart else 'boosting diversity'}", flush=True)

        # Adaptive rates: boost fresh injection and mutation when stagnant
        fresh_rate = self.fresh_injection_rate
        n_mutations_base = 1
        if hard_restart:
            # Keep only the single best, fill rest with fresh
            fresh_rate = 0.98
            n_mutations_base = 1
            self._stagnant_gens = 0  # reset counter after restart
        elif stagnant:
            fresh_rate = min(0.5, self.fresh_injection_rate * 3.0)
            n_mutations_base = 3

        # Elitism (reduced when stagnant to avoid locking in bad genomes)
        if hard_restart:
            n_elite = 1  # only best survives restart
        elif stagnant:
            n_elite = max(2, self.elitism // 2)
        else:
            n_elite = self.elitism

        # --- Niche-based diversity enforcement ---
        # Group ranked individuals by formula archetype to avoid population homogeneity.
        # Limit each archetype to at most 2 elites; fill remaining elite slots
        # from under-represented archetypes.
        niche_counts = {}
        diverse_elites = []
        for ind in ranked:
            if len(diverse_elites) >= n_elite:
                break
            if ind.fitness is None:
                continue
            arch = self._archetype(ind.genome)
            cnt = niche_counts.get(arch, 0)
            max_per_niche = 2 if not stagnant else 1
            if cnt < max_per_niche:
                niche_counts[arch] = cnt + 1
                diverse_elites.append(ind)

        for ind in diverse_elites:
            e = GenomeIndividual(genome=deep_copy_genome(ind.genome),
                                 fitness=ind.fitness)
            e.age = ind.age + 1
            nxt.append(e)

        # Fresh injection (boosted when stagnant)
        n_fresh = int(self.pop_size * fresh_rate)
        for _ in range(n_fresh):
            nxt.append(GenomeIndividual(random_genome(self.rng)))

        # Hall of fame mutations
        if self._hall_of_fame:
            n_hof = max(2, self.pop_size // 15)
            for _ in range(n_hof):
                base = self._hall_of_fame[self.rng.randint(len(self._hall_of_fame))]
                child = mutate_genome(deep_copy_genome(base), self.rng,
                                      n_mutations=self.rng.randint(2, 6))
                nxt.append(GenomeIndividual(child))

        # Update hall of fame — maintain DIVERSE entries by archetype
        if ranked[0].fitness is not None:
            new_sig = ranked[0].genome.to_oneliner()
            existing_sigs = {g.to_oneliner() for g in self._hall_of_fame}
            if new_sig not in existing_sigs:
                self._hall_of_fame.append(deep_copy_genome(ranked[0].genome))
            # Also add other good distinct genomes (diverse archetypes) into HoF
            hof_archetypes = {self._archetype(g) for g in self._hall_of_fame}
            for ind in ranked[1:min(10, len(ranked))]:
                if ind.fitness is not None and ind.fitness.ber < 0.5:
                    sig = ind.genome.to_oneliner()
                    arch = self._archetype(ind.genome)
                    if sig not in existing_sigs and len(self._hall_of_fame) < 25:
                        # Prefer diverse archetypes
                        if arch not in hof_archetypes or len(self._hall_of_fame) < 15:
                            self._hall_of_fame.append(deep_copy_genome(ind.genome))
                            existing_sigs.add(sig)
                            hof_archetypes.add(arch)
            # Trim HoF to 25 max, keeping diverse entries
            if len(self._hall_of_fame) > 25:
                self._hall_of_fame = self._hall_of_fame[-25:]

        # Offspring (tournament + mutation/crossover)
        while len(nxt) < self.pop_size:
            if self.rng.rand() < self.crossover_rate:
                p1 = self._tournament(pop)
                p2 = self._tournament(pop)
                child = crossover_genome(p1.genome, p2.genome, self.rng)
            else:
                parent = self._tournament(pop)
                if stagnant:
                    nm = self.rng.randint(n_mutations_base, n_mutations_base + 4)
                else:
                    nm = 1 if self.rng.rand() < 0.4 else self.rng.randint(2, 5)
                child = mutate_genome(parent.genome, self.rng, nm)
            nxt.append(GenomeIndividual(child))

        return nxt[:self.pop_size]

    def _tournament(self, pop):
        idxs = self.rng.choice(len(pop), min(self.tournament_size, len(pop)),
                                replace=False)
        cands = [pop[i] for i in idxs]
        cands.sort(key=lambda x: x.fitness.composite_score()
                   if x.fitness else 1e9)
        return cands[0]

    @staticmethod
    def _archetype(genome: Genome) -> str:
        """Compute a coarse structural fingerprint for niche classification.
        
        Groups genomes by the set of instruction CATEGORIES used in each program,
        ignoring parameter values and ordering.  Two genomes with the same archetype
        use the same types of operations (e.g., both use ForEachChild + GetMUp in F_up,
        both use GetMDown + GetMUp + cumDist in F_belief).
        """
        def _prog_sig(prog):
            # Extract category-level instruction names (ignore EvoConst indices)
            cats = set()
            for instr in prog:
                name = instr.name
                # Collapse EvoConst0-3 → EvoConst
                if name.startswith('Float.EvoConst'):
                    name = 'Float.EvoConst'
                # Collapse Const variants
                if name in ('Float.Const0', 'Float.Const1', 'Float.ConstHalf',
                            'Float.Const2', 'Float.Const0_1'):
                    name = 'Float.Const'
                cats.add(name)
                if hasattr(instr, 'code_block') and instr.code_block:
                    for inner in instr.code_block:
                        iname = inner.name
                        if iname.startswith('Float.EvoConst'):
                            iname = 'Float.EvoConst'
                        cats.add(iname)
            return frozenset(cats)
        return str((_prog_sig(genome.prog_down),
                    _prog_sig(genome.prog_up),
                    _prog_sig(genome.prog_belief)))

    def constant_hill_climb(self, pop, gen_seed, top_k=3, n_trials=20,
                            sigma=0.3):
        """Local search on evolved constants for top-k individuals.
        
        For each of the top-k individuals, try n_trials random perturbations
        of their evolved constants, keeping the best.  Uses C++ batch
        evaluation when available (all top_k × n_trials trials in one batch).
        """
        ranked = sorted(
            [ind for ind in pop if ind.fitness is not None],
            key=lambda x: x.fitness.composite_score())
        if not ranked:
            return
        epoch_seed = 7000 + (gen_seed // 3) * 3
        ds = self.evaluator.build_dataset(epoch_seed)
        ds_hold = self.evaluator.build_dataset(
            epoch_seed + 10000,
            n=max(3, self.evaluator.train_samples // 2))

        cpp = (self.evaluator.cpp_eval
               if (self.evaluator.use_cpp
                   and self.evaluator.cpp_eval is not None) else None)

        selected = ranked[:top_k]

        # ── C++ batch path: evaluate all trials in parallel ───────────────
        if cpp is not None:
            trial_genomes = []
            trial_consts_list = []
            for ind in selected:
                base_consts = ind.genome.log_constants.copy()
                for _ in range(n_trials):
                    tc = base_consts + self.rng.randn(N_EVO_CONSTS) * sigma
                    tc = np.clip(tc, -3.0, 3.0)
                    tg = deep_copy_genome(ind.genome)
                    tg.log_constants = tc
                    trial_genomes.append(tg)
                    trial_consts_list.append(tc)

            ber_m, fl_m, faults_m, bp_m = cpp.evaluate_batch(
                trial_genomes, ds)
            ber_h, fl_h, faults_h, bp_h = cpp.evaluate_batch(
                trial_genomes, ds_hold)
            baseline_ber, _, _ = cpp.evaluate_baselines(ds)
            n_main, n_hold = len(ds), len(ds_hold)
            n_total = n_main + n_hold

            n_improved = 0
            for k, ind in enumerate(selected):
                best_fit = ind.fitness
                best_consts = ind.genome.log_constants.copy()
                for t in range(n_trials):
                    i = k * n_trials + t
                    avg_ber = 0.55 * ber_m[i] + 0.45 * ber_h[i]
                    avg_fl = 0.55 * fl_m[i] + 0.45 * fl_h[i]
                    frac_f = (0.55 * faults_m[i] + 0.45 * faults_h[i]
                              ) / max(1, n_main)
                    gap = abs(ber_m[i] - ber_h[i])
                    ratio = avg_ber / max(baseline_ber, 1e-6)
                    avg_bp = (bp_m[i] * n_main + bp_h[i] * n_hold
                              ) / max(1, n_total)
                    fit = FitnessResult(
                        ber=avg_ber, mse=0.0, avg_flops=avg_fl,
                        code_length=trial_genomes[i].total_length(),
                        frac_faults=frac_f, baseline_ber=baseline_ber,
                        ber_ratio=ratio, generalization_gap=gap,
                        bp_updates=avg_bp,
                        nonlocal_bp_updates=0.0, bp_gain=0.0)
                    if fit.composite_score() < best_fit.composite_score():
                        best_fit = fit
                        best_consts = trial_consts_list[i].copy()
                if not np.array_equal(best_consts, ind.genome.log_constants):
                    ind.genome.log_constants = best_consts
                    ind.fitness = best_fit
                    n_improved += 1
            if n_improved > 0:
                print(f"  [ConstSearch] improved {n_improved}/{top_k}"
                      f" individuals", flush=True)
            return

        # ── Sequential fallback ───────────────────────────────────────────
        n_improved = 0
        for ind in selected:
            best_fit = ind.fitness
            best_consts = ind.genome.log_constants.copy()
            for _ in range(n_trials):
                trial_consts = best_consts + self.rng.randn(N_EVO_CONSTS) * sigma
                trial_consts = np.clip(trial_consts, -3.0, 3.0)
                trial_genome = deep_copy_genome(ind.genome)
                trial_genome.log_constants = trial_consts
                trial_fit, _ = self.evaluator.evaluate(trial_genome, ds, ds_hold)
                if trial_fit.composite_score() < best_fit.composite_score():
                    best_fit = trial_fit
                    best_consts = trial_consts.copy()
            if not np.array_equal(best_consts, ind.genome.log_constants):
                ind.genome.log_constants = best_consts
                ind.fitness = best_fit
                n_improved += 1
        if n_improved > 0:
            print(f"  [ConstSearch] improved {n_improved}/{top_k} individuals",
                  flush=True)


# --------------------------------------------------------------------------
# Full evaluation
# --------------------------------------------------------------------------

def baseline_evaluation(Nt, Nr, mod_order, n_trials=200,
                        snr_dbs=None, min_bit_errors=500, use_cpp=False):
    """Evaluate baselines only (LMMSE, K-Best 16, K-Best 32).
    Runs at experiment start to establish reference BER curves."""
    if snr_dbs is None:
        snr_dbs = [8.0, 10.0, 12.0, 14.0, 16.0]
    constellation = constellation_for(mod_order)
    rng = np.random.RandomState(2026)
    results = []

    # C++ fast path: evaluate all samples per SNR in one C++ call
    cpp_bl = None
    if use_cpp:
        try:
            from cpp_bridge import CppBPEvaluator
            cpp_bl = CppBPEvaluator(Nt=Nt, Nr=Nr, mod_order=mod_order,
                                     max_nodes=500, flops_max=2_000_000,
                                     step_max=1500, max_bp_iters=3)
            print("  [C++ baselines loaded]")
        except Exception as e:
            print(f"  WARNING: C++ baseline failed ({e}), falling back to Python")
            cpp_bl = None

    for snr in tqdm(snr_dbs, desc="Baseline SNRs", unit="SNR"):
        lm_errors, lm_bits = 0, 0
        kb16_errors, kb16_bits = 0, 0
        kb32_errors, kb32_bits = 0, 0
        n_done = 0
        pbar = tqdm(total=n_trials * 10, desc=f"  SNR={snr:.0f}dB", unit="sample",
                    leave=False, miniters=50)

        while True:
            batch_n = min(50, max(n_trials - n_done, 50))
            samples = [generate_mimo_sample(Nr, Nt, constellation, snr, rng)
                        for _ in range(batch_n)]

            if cpp_bl is not None:
                dataset = [(H, x, y, float(nv)) for H, x, y, nv in samples]
                bl, bk16, bk32 = cpp_bl.evaluate_baselines(dataset)
                lm_errors += int(round(bl * Nt * batch_n))
                lm_bits += Nt * batch_n
                kb16_errors += int(round(bk16 * Nt * batch_n))
                kb16_bits += Nt * batch_n
                kb32_errors += int(round(bk32 * Nt * batch_n))
                kb32_bits += Nt * batch_n
            else:
                for H, x, y, nv in samples:
                    xl, _ = lmmse_detect(H, y, nv, constellation)
                    lm_errors += int(np.sum(x != xl))
                    lm_bits += Nt

                    xk16, _ = kbest_detect(H, y, constellation, K=16)
                    kb16_errors += int(np.sum(x != xk16))
                    kb16_bits += Nt

                    xk32, _ = kbest_detect(H, y, constellation, K=32)
                    kb32_errors += int(np.sum(x != xk32))
                    kb32_bits += Nt

            n_done += batch_n
            pbar.update(batch_n)

            all_enough = all(e >= min_bit_errors for e in
                            [lm_errors, kb16_errors, kb32_errors])
            if n_done >= n_trials and (all_enough or n_done >= n_trials * 10):
                pbar.close()
                break

        results.append({
            'snr_db': snr,
            'lmmse_ber': lm_errors / max(1, lm_bits),
            'lmmse_bit_errors': lm_errors,
            'kbest16_ber': kb16_errors / max(1, kb16_bits),
            'kbest16_bit_errors': kb16_errors,
            'kbest32_ber': kb32_errors / max(1, kb32_bits),
            'kbest32_bit_errors': kb32_errors,
            'samples': n_done,
        })
    return results


def full_evaluation(genome: Genome, Nt, Nr, mod_order, n_trials=200,
                    snr_dbs=None, max_nodes=2000, flops_max=5_000_000,
                    step_max=8000, cpp_evaluator=None, min_bit_errors=500,
                    if_eval_baseline=False):
    """Full evaluation with minimum bit error count for statistical reliability.
    
    For each SNR, keeps running until at least min_bit_errors bit errors are
    collected OR n_trials samples are processed, whichever comes last.
    
    Args:
        if_eval_baseline: If True, also evaluate LMMSE/KB16/KB32 baselines.
                         If False, only evaluate the evolved genome.
    """
    if snr_dbs is None:
        snr_dbs = [8.0, 10.0, 12.0, 14.0, 16.0]
    constellation = constellation_for(mod_order)
    rng = np.random.RandomState(2026)
    results = []
    completed_results = {}  # Store completed SNR results for display

    use_cpp_eval = (cpp_evaluator is not None)

    if not use_cpp_eval:
        vm = MIMOPushVM(flops_max=flops_max, step_max=step_max)
        vm.evolved_constants = genome.evo_constants
        decoder = StructuredBPDecoder(Nt=Nt, Nr=Nr, constellation=constellation,
                                       max_nodes=max_nodes, vm=vm)

    for snr in tqdm(snr_dbs, desc="Full eval SNRs", unit="SNR"):
        evo_errors, evo_bits, evo_fl_sum = 0, 0, 0.0
        lm_errors, lm_bits = 0, 0
        kb16_errors, kb16_bits = 0, 0
        kb32_errors, kb32_bits = 0, 0
        n_done = 0
        batch_size = 50  # process in batches for C++ efficiency
        pbar = tqdm(total=n_trials * 10, desc=f"  SNR={snr:.0f}dB", unit="sample",
                    leave=False, miniters=50)

        while True:
            # Generate batch of samples
            batch_n = min(batch_size, max(n_trials - n_done, batch_size))
            samples = [generate_mimo_sample(Nr, Nt, constellation, snr, rng)
                        for _ in range(batch_n)]

            if use_cpp_eval:
                dataset = [(H, x, y, float(nv)) for H, x, y, nv in samples]
                avg_ber, avg_flops, _, _ = cpp_evaluator.evaluate_genome(genome, dataset)
                evo_errors += int(round(avg_ber * Nt * batch_n))
                evo_bits += Nt * batch_n
                evo_fl_sum += avg_flops * batch_n
            else:
                for H, x, y, nv in samples:
                    xh, fl = decoder.detect(H, y,
                                            prog_down=genome.prog_down,
                                            prog_up=genome.prog_up,
                                            prog_belief=genome.prog_belief,
                                            prog_halt=genome.prog_halt,
                                            noise_var=float(nv))
                    evo_errors += int(np.sum(x != xh))
                    evo_bits += Nt
                    evo_fl_sum += fl

            # Only evaluate baselines if requested
            if if_eval_baseline:
                if cpp_evaluator is not None:
                    dataset_bl = [(H, x, y, float(nv)) for H, x, y, nv in samples]
                    bl_lm, bl_k16, bl_k32 = cpp_evaluator.evaluate_baselines(dataset_bl)
                    lm_errors += int(round(bl_lm * Nt * batch_n))
                    lm_bits += Nt * batch_n
                    kb16_errors += int(round(bl_k16 * Nt * batch_n))
                    kb16_bits += Nt * batch_n
                    kb32_errors += int(round(bl_k32 * Nt * batch_n))
                    kb32_bits += Nt * batch_n
                else:
                    for H, x, y, nv in samples:
                        xl, _ = lmmse_detect(H, y, nv, constellation)
                        lm_errors += int(np.sum(x != xl))
                        lm_bits += Nt

                        xk16, _ = kbest_detect(H, y, constellation, K=16)
                        kb16_errors += int(np.sum(x != xk16))
                        kb16_bits += Nt

                        xk32, _ = kbest_detect(H, y, constellation, K=32)
                        kb32_errors += int(np.sum(x != xk32))
                        kb32_bits += Nt

            n_done += batch_n
            
            # Calculate current BER for progress display
            evo_ber = evo_errors / max(1, evo_bits)
            lm_ber = lm_errors / max(1, lm_bits) if if_eval_baseline else None
            kb16_ber = kb16_errors / max(1, kb16_bits) if if_eval_baseline else None
            kb32_ber = kb32_errors / max(1, kb32_bits) if if_eval_baseline else None
            
            # Build progress bar suffix with real-time BER info
            postfix_parts = [f"BER={evo_ber:.5f}"]
            if if_eval_baseline:
                postfix_parts.append(f"LM={lm_ber:.5f}")
                postfix_parts.append(f"K16={kb16_ber:.5f}")
                postfix_parts.append(f"K32={kb32_ber:.5f}")
            
            # Add completed SNRs info
            if completed_results:
                completed_info = " | ".join(
                    [f"SNR={s:.0f}:{completed_results[s]['ber']:.5f}" 
                     for s in sorted(completed_results.keys())])
                postfix_parts.append(completed_info)
            
            pbar.set_postfix_str(" ".join(postfix_parts))
            pbar.update(batch_n)

            # Check termination: all detectors must have enough errors OR enough samples
            if if_eval_baseline:
                all_enough = all(e >= min_bit_errors for e in
                                [evo_errors, lm_errors, kb16_errors, kb32_errors])
            else:
                all_enough = evo_errors >= min_bit_errors
                
            if n_done >= n_trials and (all_enough or n_done >= n_trials * 10):
                pbar.close()
                break

        result = {
            'snr_db': snr,
            'evolved_ber': evo_errors / max(1, evo_bits),
            'evolved_flops': evo_fl_sum / max(1, n_done),
            'evolved_samples': n_done,
            'evolved_bit_errors': evo_errors,
        }
        
        # Add baseline results only if evaluated
        if if_eval_baseline:
            result.update({
                'lmmse_ber': lm_errors / max(1, lm_bits),
                'kbest16_ber': kb16_errors / max(1, kb16_bits),
                'kbest32_ber': kb32_errors / max(1, kb32_bits),
            })
        
        results.append(result)
        completed_results[snr] = {'ber': result['evolved_ber']}
        
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Structured BP Algorithm Discovery")
    p.add_argument('--generations', type=int, default=30)
    p.add_argument('--population', type=int, default=40)
    p.add_argument('--train-samples', type=int, default=200)
    p.add_argument('--train-nt', type=int, default=8)
    p.add_argument('--train-nr', type=int, default=8)
    p.add_argument('--mod-order', type=int, default=16)
    p.add_argument('--train-max-nodes', type=int, default=1000) # 300
    p.add_argument('--train-flops-max', type=int, default=3_000_000)
    p.add_argument('--step-max', type=int, default=2000)
    p.add_argument('--eval-trials', type=int, default=1000)
    p.add_argument('--eval-max-nodes', type=int, default=1500)
    p.add_argument('--eval-flops-max', type=int, default=5_000_000)
    p.add_argument('--eval-nt', type=int, default=8,
                   help='Nt for eval (default: same as train)')
    p.add_argument('--eval-nr', type=int, default=8,
                   help='Nr for eval (default: same as train)')
    p.add_argument('--eval-step-max', type=int, default=8000)
    p.add_argument('--train-snrs', type=str, default='22')
    p.add_argument('--eval-snrs', type=str, default='16,18,20,22,24')
    p.add_argument('--seed', type=int, default=GLOBAL_SEED)
    p.add_argument('--continuous', action='store_true')
    p.add_argument('--use-cpp', action='store_true',
                   help='Use C++ DLL for BP evaluation (faster)')
    p.add_argument('--batch-gens', type=int, default=5)
    p.add_argument('--log-suffix', type=str, default='sbp0414-1')
    p.add_argument('--seed-genome-json', type=str, default=None,
                   help='Optional path to a JSON file with seed genome fields '
                        '(prog_down, prog_up, prog_belief, prog_halt, log_constants). '
                        'The genome is injected into the initial population and Hall of Fame.')
    return p.parse_args()


def parse_snrs(s):
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def run_batch(engine, logger, gens, pop=None, start_gen=0):
    if pop is None:
        pop = engine.init_population()
        print(f"Evaluating initial population ({len(pop)} individuals)...",
              flush=True)
        t0 = time.time()
        engine.evaluate_pop(pop, seed=100 + start_gen)
        print(f"  initial eval done in {time.time()-t0:.1f}s", flush=True)
        best = min(pop, key=lambda x: x.fitness.composite_score()
                   if x.fitness else 1e9)
        logger.gen(start_gen, best.fitness, best.genome, "initial")
    else:
        best = min(pop, key=lambda x: x.fitness.composite_score()
                   if x.fitness else 1e9)

    history = []
    g = start_gen
    for _ in range(gens):
        g += 1
        t0 = time.time()
        pop = engine.next_gen(pop)
        engine.evaluate_pop(pop, seed=100 + g)
        engine.constant_hill_climb(pop, gen_seed=100 + g)
        dt = time.time() - t0

        # Sort by composite_score; best individual is ranked[0]
        ranked = sorted(
            [ind for ind in pop if ind.fitness is not None],
            key=lambda x: x.fitness.composite_score())

        cur = ranked[0] if ranked else pop[0]
        if cur.fitness.composite_score() < best.fitness.composite_score():
            best = cur
            note = "NEW BEST"
        else:
            note = ""
        history.append({
            'gen': g, 'ber': cur.fitness.ber,
            'flops': cur.fitness.avg_flops,
            'len': cur.fitness.code_length,
            'bp_up': cur.fitness.bp_updates,
        })
        logger.gen(g, cur.fitness, cur.genome, note)
        n_unique = len(set(ind.genome.to_oneliner() for ind in pop
                          if ind.fitness is not None))
        print(f"Gen {g:04d} [{dt:.1f}s] | BER={cur.fitness.ber:.5f} "
              f"ratio={cur.fitness.ber_ratio:.3f} "
              f"FLOPs={cur.fitness.avg_flops:.0f} "
              f"len={cur.fitness.code_length} "
              f"uniq={n_unique}/{len(pop)} "
              f"faults={cur.fitness.frac_faults:.2f} "
              f"bp_up={cur.fitness.bp_updates:.1f}"
              + (f"  *** {note}" if note else ""), flush=True)

        # Log diversity info
        logger._w(f"  [Diversity] unique={n_unique}/{len(pop)}  "
                  f"stagnant_gens={engine._stagnant_gens}\n")

        # Write top-10 individuals with formulas to LOG (not console)
        top10 = ranked[:10]
        log_lines = [f"  Top-{len(top10)} individuals (Gen {g}):"]
        for rank, ind in enumerate(top10, 1):
            f = ind.fitness
            g_obj = ind.genome
            fd = program_to_formula_trace(g_obj.prog_down, 'down', g_obj)
            fu = program_to_formula_trace(g_obj.prog_up, 'up', g_obj)
            fb = program_to_formula_trace(g_obj.prog_belief, 'belief', g_obj)
            fh = program_to_formula_trace(g_obj.prog_halt, 'halt', g_obj)
            log_lines.append(
                f"    #{rank:2d}  score={f.composite_score():.5f}  "
                f"BER={f.ber:.5f}  faults={f.frac_faults:.2f}  "
                f"len={f.code_length}")
            log_lines.append(f"          F_down  = {fd}")
            log_lines.append(f"          F_up    = {fu}")
            log_lines.append(f"          F_belief= {fb}")
            log_lines.append(f"          H_halt  = {fh}")
            if rank == 1:
                ec = g_obj.evo_constants
                log_lines.append(f"          [EC]  EC0={ec[0]:.6f} EC1={ec[1]:.6f} "
                                 f"EC2={ec[2]:.6f} EC3={ec[3]:.6f}")
                # Pseudocode for each program — much more readable than raw bytecode
                # Variable names:
                #   F_down  inputs: [M_par_down, C_i] (float) / [layer] (int)
                #   F_up    inputs: [C_1,M_1,...,C_16,M_16] (float) / [layer,N_ch] (int)
                #   F_belief inputs: [D_i, M_dn, M_up] (float) / [layer] (int)
                #   H_halt  inputs: [old_Mup, new_Mup] (float) / [layer] (int)
                for pt_label, pt_type, pt_prog in [
                    ('F_down ', 'down',   g_obj.prog_down),
                    ('F_up   ', 'up',     g_obj.prog_up),
                    ('F_belief', 'belief', g_obj.prog_belief),
                    ('H_halt ', 'halt',   g_obj.prog_halt),
                ]:
                    plines = _format_pseudocode_block(pt_prog, pt_type, g_obj, base_indent=0)
                    log_lines.append(f"          [{pt_label}]:  ({len(pt_prog)} instrs)")
                    for pl in plines:
                        log_lines.append(f"            {pl}")
        logger._w("\n".join(log_lines) + "\n")
    return best, history, pop, g


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    base = os.path.dirname(os.path.abspath(__file__))
    topic = os.path.dirname(base)
    logs_dir = os.path.join(topic, 'logs')
    results_dir = os.path.join(topic, 'results')
    ensure_dir(logs_dir)
    ensure_dir(results_dir)

    log = Logger(os.path.join(logs_dir,
                              f'sbp_evolution_{args.log_suffix}.log'))
    train_snrs = parse_snrs(args.train_snrs)
    eval_snrs = parse_snrs(args.eval_snrs)

    log.info("Configuration", json.dumps(vars(args), indent=2))
    print(f"\nStructured BP Stack Decoder — Algorithm Discovery v2")
    print(f"  4-program genome: F_down, F_up, F_belief, H_halt")
    print(f"  max_nodes={args.train_max_nodes}, step_max={args.step_max}")
    print(f"  flops_max={args.train_flops_max}, pop={args.population}")
    print(f"  train_snrs={train_snrs}, eval_snrs={eval_snrs}\n")

    # ── Baseline evaluation at experiment start ──────────────────────────
    eval_nt = args.eval_nt or args.train_nt
    eval_nr = args.eval_nr or args.train_nr
    print(f"--- Baseline evaluation (LMMSE, K-Best 16, K-Best 32) ---", flush=True)
    baseline_results = baseline_evaluation(
        Nt=eval_nt, Nr=eval_nr, mod_order=args.mod_order,
        n_trials=args.eval_trials, snr_dbs=eval_snrs, min_bit_errors=100,
        use_cpp=args.use_cpp)
    for r in baseline_results:
        print(f"  SNR={r['snr_db']:5.1f}  "
              f"LMMSE={r['lmmse_ber']:.5f}({r['lmmse_bit_errors']}err)  "
              f"KB16={r['kbest16_ber']:.5f}({r['kbest16_bit_errors']}err)  "
              f"KB32={r['kbest32_ber']:.5f}({r['kbest32_bit_errors']}err)  "
              f"[{r['samples']} samples]")
    log.info("Baseline evaluation (experiment start)", json.dumps(baseline_results, indent=2))
    print()

    evaluator = StructuredBPEvaluator(
        Nt=args.train_nt, Nr=args.train_nr, mod_order=args.mod_order,
        flops_max=args.train_flops_max, max_nodes=args.train_max_nodes,
        train_samples=args.train_samples, snr_choices=train_snrs,
        step_max=args.step_max, use_cpp=args.use_cpp)

    # Build a separate C++ evaluator for full-eval (larger nodes/flops budget)
    full_eval_cpp = None
    if args.use_cpp:
        try:
            from cpp_bridge import CppBPEvaluator
            full_eval_cpp = CppBPEvaluator(
                Nt=eval_nt, Nr=eval_nr,
                mod_order=args.mod_order,
                max_nodes=args.eval_max_nodes,
                flops_max=args.eval_flops_max,
                step_max=args.eval_step_max)
            print(f"  Full-eval C++ accelerator ready (max_nodes={args.eval_max_nodes})")
        except Exception as e:
            print(f"  WARNING: could not load C++ full-eval accelerator: {e}")

    engine = StructuredBPEngine(
        pop_size=args.population, tournament_size=5, elitism=6,
        mutation_rate=0.75, crossover_rate=0.25,
        seed=args.seed, evaluator=evaluator)

    # Optional: inject seed genome into Hall of Fame for warm-start
    if args.seed_genome_json:
        try:
            with open(args.seed_genome_json, 'r') as _f:
                _sd = json.load(_f)
            def _make_instr(item):
                """Build Instruction, handling ForEachChild with body."""
                if isinstance(item, dict):
                    body = [_make_instr(b) for b in item.get('body', [])]
                    return Instruction(name=item['name'], code_block=body)
                return Instruction(name=item)
            _seed_genome = Genome(
                prog_down=[_make_instr(n) for n in _sd.get('prog_down', [])],
                prog_up=[_make_instr(n) for n in _sd.get('prog_up', [])],
                prog_belief=[_make_instr(n) for n in _sd.get('prog_belief', [])],
                prog_halt=[_make_instr(n) for n in _sd.get('prog_halt', [])],
                log_constants=np.array(_sd.get('log_constants', [0.0]*N_EVO_CONSTS)))
            engine._hall_of_fame.append(_seed_genome)
            # Also inject as first individual in population for warm-start
            engine._seed_genomes = [_seed_genome]
            print(f"  Loaded seed genome from {args.seed_genome_json}")
            log.info("Seed genome loaded", json.dumps(_sd, indent=2))
        except Exception as e:
            import traceback
            print(f"  WARNING: could not load seed genome: {e}")
            traceback.print_exc()

    history = []
    pop = None
    g = 0
    best = None

    def _do_full_eval(gen_num, genome, label="Full eval"):
        """Helper: run full SNR evaluation and log results."""
        print(f"\n--- {label} (gen {gen_num}) ---", flush=True)
        _log_genome_formulas(log, genome, f"Best formulas (gen {gen_num})")
        ev = full_evaluation(
            genome, Nt=eval_nt, Nr=eval_nr,
            mod_order=args.mod_order, n_trials=args.eval_trials,
            snr_dbs=eval_snrs, max_nodes=args.eval_max_nodes,
            flops_max=args.eval_flops_max, step_max=args.eval_step_max,
            cpp_evaluator=full_eval_cpp, min_bit_errors=500,
            if_eval_baseline=True)
        for r in ev:
            line = f"  SNR={r['snr_db']:5.1f}  " \
                   f"Evo={r['evolved_ber']:.5f}({r.get('evolved_bit_errors','?')}err,{r['evolved_flops']:.0f}fl)"
            if 'lmmse_ber' in r:
                line += f"  LMMSE={r['lmmse_ber']:.5f}  KB16={r['kbest16_ber']:.5f}  KB32={r['kbest32_ber']:.5f}"
            print(line)
        log.info(f"{label} (gen {gen_num})", json.dumps(ev, indent=2))
        save_path = os.path.join(results_dir,
                                 f'sbp_{args.log_suffix}_gen{gen_num}.json')
        with open(save_path, 'w') as f:
            json.dump({
                'gen': gen_num,
                'best_genome': genome.to_oneliner(),
                'eval_results': ev,
                'baseline_results': baseline_results,
                'history': history[-20:],
            }, f, indent=2)
        return ev

    if args.continuous:
        try:
            while True:
                best, bh, pop, g = run_batch(engine, log, args.batch_gens,
                                             pop, g)
                history.extend(bh)

                # Full eval every batch_gens generations
                _do_full_eval(g, best.genome, "Periodic full eval")
                print()

        except KeyboardInterrupt:
            print("\nInterrupted.")
    else:
        best, history, pop, g = run_batch(engine, log, args.generations)
        _do_full_eval(g, best.genome, "Full eval final")

    print(f"\nBest genome:\n{best.genome.to_oneliner()}")
    print_genome_formulas(best.genome, "Best Evolved Formulas")
    print(f"\nF_down:\n{program_to_string(best.genome.prog_down)}")
    print(f"\nF_up:\n{program_to_string(best.genome.prog_up)}")
    print(f"\nF_belief:\n{program_to_string(best.genome.prog_belief)}")
    print(f"\nH_halt:\n{program_to_string(best.genome.prog_halt)}")


if __name__ == '__main__':
    main()