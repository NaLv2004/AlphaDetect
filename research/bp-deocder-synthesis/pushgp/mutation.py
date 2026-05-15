"""Genetic mutation operators for Push-GP programs.

Each operator takes a program (a `List[Instruction]`) and an
`np.random.Generator`, plus the `RandomProgramGenerator` and the
instruction subset to draw new code from, and returns a NEW program
(the input is never mutated in place).

A program is a *tree* of `Instruction` nodes whose internal nodes are
control instructions (`Exec.If`, `Exec.DoTimes`, `Exec.DoRange`,
`Exec.When`, `Exec.While`).  Each control instruction owns one or two
`code_block` fields, which are themselves `List[Instruction]` — i.e.
sub-programs structurally identical to the top level.

All seven mutation operators below operate on **arbitrary positions in
the tree**, not just the top-level list.  A position is picked by
uniformly sampling over every (parent_list, index) pair reachable by
recursive descent.  This means:

  * `mut_point` may replace an instruction inside a `DoTimes` body, or
    inside the body of a nested `DoTimes` inside another `DoTimes`.
  * `mut_insert` may insert into an inner block.
  * `mut_swap` may swap two instructions that sit in different blocks.
  * `mut_block` keeps its old behaviour (regenerate one whole block) but
    is now also offered the inner blocks as candidates.

The full list (mirrors `mimo-push-gp` plus full-tree generalisation):
  * point     — replace one instruction (anywhere in the tree)
  * insert    — insert one instruction (anywhere in the tree)
  * delete    — remove one instruction (anywhere in the tree)
  * swap      — swap two instructions (any two positions in the tree)
  * block     — regenerate code_block / code_block2 of a random control node
  * segment   — replace a contiguous slice of any block in the tree
  * grow      — prepend or append a small block to a random list in the tree

Plus a separate operator that perturbs the log-constant vector of a
`Genome` (constants live outside the program list).
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

from .genome import (
    Genome,
    LOG_CONST_MAX,
    LOG_CONST_MIN,
    MAX_PROG_LEN,
    N_EVO_CONSTS,
)
from .instructions import has_two_blocks, is_control
from .program import Instruction, deep_copy_program, program_length
from .random_program import (
    C2V_INSTR,
    RandomProgramGenerator,
    V2C_INSTR,
    truncate_program_to_max,
)


MutOp = Callable[
    [List[Instruction], np.random.Generator, RandomProgramGenerator, Sequence[str]],
    List[Instruction],
]


# ============================================================ Tree traversal


# A node position: (parent_list, index_in_parent, depth_of_node).
# `parent_list` is a *live reference* into the program tree — mutating
# parent_list[index] mutates the program in place (callers must work on
# a deep copy if they need an immutable input).
NodePos = Tuple[List[Instruction], int, int]
# A list position used for inserting / appending / segment replacement:
# (target_list, depth_of_list).
ListPos = Tuple[List[Instruction], int]


def _all_node_positions(prog: List[Instruction]) -> List[NodePos]:
    """Every (parent_list, index, depth) in the tree (depth=0 at top)."""
    out: List[NodePos] = []

    def rec(lst: List[Instruction], depth: int) -> None:
        for i, ins in enumerate(lst):
            out.append((lst, i, depth))
            if ins.code_block is not None:
                rec(ins.code_block, depth + 1)
            if ins.code_block2 is not None:
                rec(ins.code_block2, depth + 1)

    rec(prog, 0)
    return out


def _all_list_positions(prog: List[Instruction]) -> List[ListPos]:
    """Every (list, depth_of_list) reachable.

    The top-level list is depth 0; any code_block lives at depth =
    (depth_of_owning_node) + 1.  An EMPTY block is still a valid target
    for insertion / growth.
    """
    out: List[ListPos] = [(prog, 0)]

    def rec(lst: List[Instruction], depth: int) -> None:
        for ins in lst:
            if ins.code_block is not None:
                out.append((ins.code_block, depth + 1))
                rec(ins.code_block, depth + 1)
            if ins.code_block2 is not None:
                out.append((ins.code_block2, depth + 1))
                rec(ins.code_block2, depth + 1)

    rec(prog, 0)
    return out


def _all_control_positions(prog: List[Instruction]) -> List[NodePos]:
    """Every position whose instruction is a control (has a code_block)."""
    return [(par, i, d) for par, i, d in _all_node_positions(prog)
            if par[i].is_control()]


def _depth_budget(gen: RandomProgramGenerator, current_depth: int) -> int:
    """Maximum allowed _subtree_max_depth for an instruction placed at
    `current_depth`.  Equals max_recur_depth − current_depth, clipped ≥ 0."""
    return max(0, gen.max_recur_depth - current_depth)


def _subtree_max_depth(ins: Instruction) -> int:
    """Maximum nesting depth inside this single instruction (atomic = 0)."""
    if not ins.is_control():
        return 0
    sub = 0
    for blk in (ins.code_block, ins.code_block2):
        if blk is None:
            continue
        for child in blk:
            sub = max(sub, 1 + _subtree_max_depth(child))
    return sub


def _clamp_subtree_depth(
    ins: Instruction,
    max_subtree_depth: int,
    instr_set: Sequence[str],
    rng: np.random.Generator,
) -> None:
    """In-place: enforce _subtree_max_depth(ins) ≤ max_subtree_depth.

    If `max_subtree_depth == 0` and `ins` is a control, `ins` is
    DEMOTED in place to a random atomic (name + blocks blanked).  This
    is the only way to satisfy a 0 budget without losing the slot.
    """
    if max_subtree_depth < 0:
        max_subtree_depth = 0
    if not ins.is_control():
        return  # already depth 0
    if max_subtree_depth == 0:
        atomic_names = [n for n in instr_set if not is_control(n)]
        if not atomic_names:
            atomic_names = list(instr_set)
        ins.name = str(rng.choice(atomic_names))
        ins.code_block = None
        ins.code_block2 = None
        return
    # max_subtree_depth >= 1: ins may stay a control whose blocks have
    # subtree_depth ≤ max_subtree_depth − 1.
    for blk_attr in ("code_block", "code_block2"):
        blk = getattr(ins, blk_attr)
        if blk is None:
            continue
        for child in blk:
            _clamp_subtree_depth(child, max_subtree_depth - 1, instr_set, rng)


def _build_random_subtree(
    gen: RandomProgramGenerator,
    instr_set: Sequence[str],
    rng: np.random.Generator,
    max_subtree_depth: int,
) -> Instruction:
    """Build one random instruction whose subtree depth ≤ max_subtree_depth."""
    if max_subtree_depth <= 0:
        # Force atomic.
        atomic = [n for n in instr_set if not is_control(n)]
        if not atomic:
            atomic = list(instr_set)
        return Instruction(name=str(rng.choice(atomic)))
    seq = gen.random_program(instr_set, min_size=1, max_size=1)
    _clamp_subtree_depth(seq[0], max_subtree_depth, instr_set, rng)
    return seq[0]


# ============================================================ Operators


def mut_point(prog, rng, gen, instr_set):
    """Replace ONE instruction at a uniformly-sampled tree position."""
    p = deep_copy_program(prog)
    positions = _all_node_positions(p)
    if not positions:
        return [gen.random_instruction(instr_set)]
    par, idx, depth = positions[int(rng.integers(0, len(positions)))]
    par[idx] = _build_random_subtree(gen, instr_set, rng,
                                     _depth_budget(gen, depth))
    return truncate_program_to_max(p)


def mut_insert(prog, rng, gen, instr_set):
    """Insert ONE instruction into a uniformly-sampled list (any depth)."""
    p = deep_copy_program(prog)
    lists = _all_list_positions(p)
    target, depth = lists[int(rng.integers(0, len(lists)))]
    pos = int(rng.integers(0, len(target) + 1))
    target.insert(pos, _build_random_subtree(gen, instr_set, rng,
                                             _depth_budget(gen, depth)))
    return truncate_program_to_max(p)


def mut_delete(prog, rng, gen, instr_set):
    """Remove ONE instruction at a uniformly-sampled tree position.

    Refuses to delete the only remaining top-level instruction; if the
    chosen position is the sole element of an inner block we also keep
    one element to avoid orphaning a control node with empty children.
    """
    p = deep_copy_program(prog)
    if program_length(p) <= 1:
        return p
    positions = _all_node_positions(p)
    # Filter out positions whose deletion would empty their parent list
    # AND whose parent list is at depth>0 (top-level may shrink down to
    # 1 element legally).
    safe = [(par, idx, d) for par, idx, d in positions
            if not (d > 0 and len(par) == 1)]
    if not safe:
        # All inner blocks are size-1; fall back to top-level deletion.
        safe = [(par, idx, d) for par, idx, d in positions if d == 0 and len(p) > 1]
        if not safe:
            return p
    par, idx, _ = safe[int(rng.integers(0, len(safe)))]
    del par[idx]
    return p


def mut_swap(prog, rng, gen, instr_set):
    """Swap two uniformly-sampled tree positions (may straddle blocks)."""
    p = deep_copy_program(prog)
    positions = _all_node_positions(p)
    if len(positions) < 2:
        return p
    a, b = rng.choice(len(positions), size=2, replace=False)
    par_a, idx_a, _ = positions[int(a)]
    par_b, idx_b, _ = positions[int(b)]
    par_a[idx_a], par_b[idx_b] = par_b[idx_b], par_a[idx_a]
    return p


def mut_block(prog, rng, gen, instr_set):
    """Regenerate code_block (and possibly code_block2) of a random control
    node anywhere in the tree."""
    p = deep_copy_program(prog)
    ctrls = _all_control_positions(p)
    if not ctrls:
        return mut_point(p, rng, gen, instr_set)
    par, idx, depth = ctrls[int(rng.integers(0, len(ctrls)))]
    node = par[idx]
    new_size = int(rng.integers(2, 6))
    inner_depth = depth + 1
    # Children of the new block are placed at `inner_depth`; their own
    # subtrees may have depth ≤ max_recur_depth − inner_depth.
    child_budget = max(0, gen.max_recur_depth - inner_depth)
    block = gen.random_program(instr_set, min_size=new_size, max_size=new_size + 2)
    for child in block:
        _clamp_subtree_depth(child, child_budget, instr_set, rng)
    node.code_block = block
    if has_two_blocks(node.name) and node.code_block2 is not None and rng.random() < 0.5:
        block2 = gen.random_program(instr_set, min_size=new_size, max_size=new_size + 2)
        for child in block2:
            _clamp_subtree_depth(child, child_budget, instr_set, rng)
        node.code_block2 = block2
    return truncate_program_to_max(p)


def mut_segment(prog, rng, gen, instr_set):
    """Replace a contiguous segment of a uniformly-sampled list."""
    p = deep_copy_program(prog)
    lists = [(lst, d) for lst, d in _all_list_positions(p) if len(lst) > 0]
    if not lists:
        return [gen.random_instruction(instr_set)]
    target, depth = lists[int(rng.integers(0, len(lists)))]
    n = len(target)
    a = int(rng.integers(0, n))
    b = int(rng.integers(a + 1, n + 1))
    seg_size = max(1, int(rng.integers(1, max(2, b - a + 2))))
    new_seg = gen.random_program(instr_set, min_size=seg_size, max_size=seg_size + 1)
    # Children placed at `depth`; allowed subtree depth = max_recur_depth − depth.
    child_budget = max(0, gen.max_recur_depth - depth)
    for child in new_seg:
        _clamp_subtree_depth(child, child_budget, instr_set, rng)
    target[a:b] = new_seg
    return truncate_program_to_max(p)


def mut_grow(prog, rng, gen, instr_set):
    """Prepend or append a fresh small block to a uniformly-sampled list."""
    p = deep_copy_program(prog)
    lists = _all_list_positions(p)
    target, depth = lists[int(rng.integers(0, len(lists)))]
    block = gen.random_program(instr_set, min_size=2, max_size=4)
    child_budget = max(0, gen.max_recur_depth - depth)
    for child in block:
        _clamp_subtree_depth(child, child_budget, instr_set, rng)
    if rng.random() < 0.5:
        target[:0] = block
    else:
        target.extend(block)
    return truncate_program_to_max(p)


_OPS: List[MutOp] = [
    mut_point,
    mut_insert,
    mut_delete,
    mut_swap,
    mut_block,
    mut_segment,
    mut_grow,
]

# ----------------------------------------------------------- Constant tweak


def mutate_log_constants(
    log_consts: np.ndarray,
    rng: np.random.Generator,
    n_tweaks: int = 1,
    sigma: float = 0.3,
) -> np.ndarray:
    out = log_consts.copy()
    for _ in range(n_tweaks):
        i = int(rng.integers(0, N_EVO_CONSTS))
        out[i] = float(np.clip(out[i] + rng.normal(0.0, sigma), LOG_CONST_MIN, LOG_CONST_MAX))
    return out


# ----------------------------------------------------------- High-level entry


def mutate_program(
    prog: List[Instruction],
    rng: np.random.Generator,
    gen: RandomProgramGenerator,
    instr_set: Sequence[str],
    n_mutations: int = 1,
) -> List[Instruction]:
    p = prog
    for _ in range(n_mutations):
        op = _OPS[int(rng.integers(0, len(_OPS)))]
        p = op(p, rng, gen, instr_set)
    return p


def mutate_genome(
    genome: Genome,
    rng: np.random.Generator,
    gen: RandomProgramGenerator,
    n_mutations: int = 2,
    p_const_tweak: float = 0.2,
) -> Genome:
    new = genome.copy()
    new.prog_v2c = mutate_program(new.prog_v2c, rng, gen, V2C_INSTR, n_mutations)
    new.prog_c2v = mutate_program(new.prog_c2v, rng, gen, C2V_INSTR, n_mutations)
    if rng.random() < p_const_tweak:
        new.log_constants = mutate_log_constants(new.log_constants, rng)
    return new


__all__ = [
    "mut_point",
    "mut_insert",
    "mut_delete",
    "mut_swap",
    "mut_block",
    "mut_segment",
    "mut_grow",
    "mutate_program",
    "mutate_genome",
    "mutate_log_constants",
]
