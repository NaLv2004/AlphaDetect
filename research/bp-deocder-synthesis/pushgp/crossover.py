"""Genetic crossover for Push-GP genomes.

Operators:

* `crossover_program(p1, p2, rng, *, mode="auto")` — combines two
  programs.  Three modes:

    "alternation"   classic single-cut alternation on the TOP-LEVEL list
                    only (legacy behaviour, kept for ablation).
    "subtree"       GP subtree crossover: pick a random node anywhere
                    in p1's tree, replace it with a random subtree taken
                    from anywhere in p2's tree.  Inner blocks therefore
                    participate in recombination.
    "auto"          choose subtree with probability `p_subtree` (0.7 by
                    default), alternation otherwise.  This is what the
                    GA uses.

* `crossover_genome(g1, g2, rng)` — independent program-level crossover
  for V2C and C2V (using "auto"), plus uniform random blending of the
  log-constant vector.

The depth of the inserted subtree is clamped so that nesting in the
child cannot exceed `RandomProgramGenerator.max_recur_depth` deeper than
the *insertion site*; offending nested controls inside the donor are
flattened to atomic placeholders before insertion.  This keeps the VM
budget invariants intact.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .genome import Genome, MAX_PROG_LEN, N_EVO_CONSTS
from .instructions import is_control
from .program import Instruction, deep_copy_program
from .random_program import (
    C2V_INSTR,
    RandomProgramGenerator,
    V2C_INSTR,
    truncate_program_to_max,
)


# ----------------------------------------------------------- Tree helpers

NodePos = Tuple[List[Instruction], int, int]


def _all_node_positions(prog: List[Instruction]) -> List[NodePos]:
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


def _subtree_max_depth(ins: Instruction) -> int:
    """Maximum nesting depth inside this single instruction (0 = atomic)."""
    if not ins.is_control():
        return 0
    sub = 0
    for blk in (ins.code_block, ins.code_block2):
        if blk is None:
            continue
        for child in blk:
            sub = max(sub, 1 + _subtree_max_depth(child))
    return sub


def _flatten_to_depth(ins: Instruction, max_subtree_depth: int,
                      instr_set, rng: np.random.Generator) -> None:
    """In-place: enforce _subtree_max_depth(ins) ≤ max_subtree_depth.

    If max_subtree_depth == 0 and ins is a control, ins itself is
    DEMOTED to a random atomic (rename + drop both blocks)."""
    if max_subtree_depth < 0:
        max_subtree_depth = 0
    if not ins.is_control():
        return
    if max_subtree_depth == 0:
        atomic = [n for n in instr_set if not is_control(n)]
        if not atomic:
            atomic = list(instr_set)
        ins.name = str(rng.choice(atomic))
        ins.code_block = None
        ins.code_block2 = None
        return
    for blk_attr in ("code_block", "code_block2"):
        blk = getattr(ins, blk_attr)
        if blk is None:
            continue
        for child in blk:
            _flatten_to_depth(child, max_subtree_depth - 1, instr_set, rng)


# ----------------------------------------------------------- Operators


def crossover_alternation(
    p1: List[Instruction],
    p2: List[Instruction],
    rng: np.random.Generator,
) -> List[Instruction]:
    """Single-cut top-level alternation (legacy)."""
    if not p1 and not p2:
        return []
    if not p1:
        return deep_copy_program(p2)
    if not p2:
        return deep_copy_program(p1)
    c1 = int(rng.integers(0, len(p1) + 1))
    c2 = int(rng.integers(0, len(p2) + 1))
    child = deep_copy_program(p1[:c1]) + deep_copy_program(p2[c2:])
    if not child:
        child = deep_copy_program(p1 if rng.random() < 0.5 else p2)
    return truncate_program_to_max(child)


def crossover_subtree(
    p1: List[Instruction],
    p2: List[Instruction],
    rng: np.random.Generator,
    *,
    max_recur_depth: int = 2,
    instr_set: Optional[List[str]] = None,
) -> List[Instruction]:
    """Replace a random NODE in p1 with a random NODE-rooted subtree from p2.

    Both donor and recipient may sit at any depth in their respective
    trees.  If the donor subtree's internal nesting plus the recipient
    site's depth would exceed `max_recur_depth`, the donor is flattened.
    """
    if not p1:
        return deep_copy_program(p2)
    if not p2:
        return deep_copy_program(p1)

    child = deep_copy_program(p1)
    p2_copy = deep_copy_program(p2)  # so donor edits don't leak back

    recip_positions = _all_node_positions(child)
    donor_positions = _all_node_positions(p2_copy)
    if not recip_positions or not donor_positions:
        return child

    par_r, idx_r, depth_r = recip_positions[
        int(rng.integers(0, len(recip_positions)))]
    par_d, idx_d, _ = donor_positions[
        int(rng.integers(0, len(donor_positions)))]

    donor = par_d[idx_d]
    # Slot constraint: donor's max internal depth + depth_r ≤ max_recur_depth.
    allowed_inner = max(0, max_recur_depth - depth_r)
    if instr_set is None:
        instr_set = [donor.name]  # fallback for atomic-only flatten target
    if _subtree_max_depth(donor) > allowed_inner:
        _flatten_to_depth(donor, allowed_inner, instr_set, rng)
    par_r[idx_r] = donor
    return truncate_program_to_max(child)


def crossover_program(
    p1: List[Instruction],
    p2: List[Instruction],
    rng: np.random.Generator,
    *,
    mode: str = "auto",
    p_subtree: float = 0.7,
    max_recur_depth: int = 2,
    instr_set: Optional[List[str]] = None,
) -> List[Instruction]:
    if mode == "alternation":
        return crossover_alternation(p1, p2, rng)
    if mode == "subtree":
        return crossover_subtree(p1, p2, rng,
                                 max_recur_depth=max_recur_depth,
                                 instr_set=instr_set)
    if mode == "auto":
        if rng.random() < p_subtree:
            return crossover_subtree(p1, p2, rng,
                                     max_recur_depth=max_recur_depth,
                                     instr_set=instr_set)
        return crossover_alternation(p1, p2, rng)
    raise ValueError(f"unknown crossover mode: {mode!r}")


def crossover_log_constants(
    c1: np.ndarray,
    c2: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-element uniform mixing alpha ∈ U[0,1]."""
    alpha = rng.uniform(0.0, 1.0, size=N_EVO_CONSTS)
    return alpha * c1 + (1.0 - alpha) * c2


def crossover_genome(
    g1: Genome,
    g2: Genome,
    rng: np.random.Generator,
    *,
    mode: str = "auto",
    max_recur_depth: int = 2,
) -> Genome:
    return Genome(
        prog_v2c=crossover_program(g1.prog_v2c, g2.prog_v2c, rng,
                                   mode=mode,
                                   max_recur_depth=max_recur_depth,
                                   instr_set=V2C_INSTR),
        prog_c2v=crossover_program(g1.prog_c2v, g2.prog_c2v, rng,
                                   mode=mode,
                                   max_recur_depth=max_recur_depth,
                                   instr_set=C2V_INSTR),
        log_constants=crossover_log_constants(g1.log_constants, g2.log_constants, rng),
    )


__all__ = [
    "crossover_program",
    "crossover_alternation",
    "crossover_subtree",
    "crossover_log_constants",
    "crossover_genome",
]
