"""Random program generation for V2C / C2V programs.

The generator samples atomic instructions uniformly at random from a
program-type-specific subset of `pushgp.instructions.HANDLERS`.  When a
control-flow instruction is sampled it recursively generates a nested
code block, with a smaller size budget per level.  Recursion depth is
strictly bounded.

The two instruction subsets enforce the only domain-specific constraint
we impose on the search space:

* `Env.GetChannelLLR` is allowed in V2C only (CN side has no channel LLR).

Everything else — vector ops, memory, control flow, conversions,
comparisons, arithmetic — is shared.  Search-space breadth is the goal.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .genome import Genome, MAX_PROG_LEN, N_EVO_CONSTS, RAND_LOG_CONST_MIN, RAND_LOG_CONST_MAX
from .instructions import HANDLERS, has_two_blocks, is_control
from .op_filter import OpFilter, filter_instr_set
from .program import Instruction, program_length

# ----------------------------------------------------------------- Subsets

ALL_NAMES: List[str] = sorted(HANDLERS.keys())

# Names that depend on V2C-only context.
_V2C_ONLY = {"Env.GetChannelLLR"}
# Currently no C2V-only names.
_C2V_ONLY: set[str] = set()

V2C_INSTR: List[str] = sorted(n for n in ALL_NAMES if n not in _C2V_ONLY)
C2V_INSTR: List[str] = sorted(n for n in ALL_NAMES if n not in _V2C_ONLY)

# Default sampling weights bias the distribution slightly toward control
# flow (so loops get explored) and away from raw stack ops (so programs
# don't waste budget shuffling things around).  Weights are heuristic;
# evolution will adapt around them.
_WEIGHT_CONTROL = 0.10
_WEIGHT_STACKOP = 0.5  # reduce baseline weight of pure stack manipulation
_STACKOP_NAMES = {
    n
    for n in ALL_NAMES
    if any(
        n.endswith(suf) for suf in (".Pop", ".Dup", ".Swap", ".Rot", ".Yank", ".Shove")
    )
}


def _instr_weights(names: Sequence[str]) -> np.ndarray:
    w = np.ones(len(names), dtype=np.float64)
    for i, n in enumerate(names):
        if is_control(n):
            w[i] = _WEIGHT_CONTROL * len(names) / max(
                1, sum(1 for nm in names if is_control(nm))
            )
        elif n in _STACKOP_NAMES:
            w[i] = _WEIGHT_STACKOP
    w /= w.sum()
    return w


# ----------------------------------------------------------------- Generator


class RandomProgramGenerator:
    """Stateful random program generator (carries its own RNG)."""

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        max_recur_depth: int = 2,
        op_filter: Optional[OpFilter] = None,
    ) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_recur_depth = max_recur_depth
        self.op_filter = op_filter
        # Pre-compute side-specific filtered instruction sets.  When no
        # filter is active these are identical to V2C_INSTR / C2V_INSTR.
        self._v2c_instr: List[str] = filter_instr_set("v2c", V2C_INSTR, op_filter)
        self._c2v_instr: List[str] = filter_instr_set("c2v", C2V_INSTR, op_filter)
        if op_filter is not None and op_filter.applies():
            if not self._v2c_instr:
                raise ValueError(
                    "OpFilter leaves V2C instruction set empty after "
                    "intersection with base V2C ops."
                )
            if not self._c2v_instr:
                raise ValueError(
                    "OpFilter leaves C2V instruction set empty after "
                    "intersection with base C2V ops."
                )

    # ---------------------------------------------------------- public API
    def random_program(
        self,
        instr_set: Sequence[str],
        min_size: int = 4,
        max_size: int = 30,
    ) -> List[Instruction]:
        """Generate a flat program (top level) of length in [min_size, max_size]."""
        n = int(self.rng.integers(min_size, max_size + 1))
        return self._gen(instr_set, n, depth=0)

    def random_v2c(self, min_size: int = 4, max_size: int = 30) -> List[Instruction]:
        return self.random_program(self._v2c_instr, min_size, max_size)

    def random_c2v(self, min_size: int = 4, max_size: int = 30) -> List[Instruction]:
        return self.random_program(self._c2v_instr, min_size, max_size)

    # ------- Accessor helpers used by evolution.py / mutation.py ------
    def v2c_instr_set(self) -> List[str]:
        """Effective V2C op list (filtered if op_filter active)."""
        return list(self._v2c_instr)

    def c2v_instr_set(self) -> List[str]:
        """Effective C2V op list (filtered if op_filter active)."""
        return list(self._c2v_instr)

    def random_log_constants(self) -> np.ndarray:
        return self.rng.uniform(RAND_LOG_CONST_MIN, RAND_LOG_CONST_MAX, size=N_EVO_CONSTS)

    def random_genome(
        self, min_size: int = 4, max_size: int = 30
    ) -> Genome:
        return Genome(
            prog_v2c=self.random_v2c(min_size, max_size),
            prog_c2v=self.random_c2v(min_size, max_size),
            log_constants=self.random_log_constants(),
        )

    def random_instruction(self, instr_set: Sequence[str]) -> Instruction:
        """Pick a single (possibly control) instruction at random."""
        return self._sample_one(instr_set, depth=0)

    # ---------------------------------------------------------- internals
    def _gen(
        self,
        instr_set: Sequence[str],
        n: int,
        depth: int,
    ) -> List[Instruction]:
        names = list(instr_set)
        weights = _instr_weights(names)
        out: List[Instruction] = []
        for _ in range(n):
            # When already deep, avoid spawning more control instructions.
            if depth >= self.max_recur_depth:
                eligible = [nm for nm in names if not is_control(nm)]
                if not eligible:
                    eligible = names
                w = _instr_weights(eligible)
                name = self.rng.choice(eligible, p=w)
            else:
                name = self.rng.choice(names, p=weights)
            out.append(self._wrap(name, instr_set, depth))
        return out

    def _sample_one(self, instr_set: Sequence[str], depth: int) -> Instruction:
        names = list(instr_set)
        if depth >= self.max_recur_depth:
            names = [nm for nm in names if not is_control(nm)]
            if not names:
                names = list(instr_set)
        weights = _instr_weights(names)
        name = self.rng.choice(names, p=weights)
        return self._wrap(name, instr_set, depth)

    def _wrap(self, name: str, instr_set: Sequence[str], depth: int) -> Instruction:
        ins = Instruction(name=str(name))
        if is_control(name):
            block_size = max(2, int(self.rng.integers(2, 7)))
            ins.code_block = self._gen(instr_set, block_size, depth + 1)
            if has_two_blocks(name):
                block_size2 = max(2, int(self.rng.integers(2, 7)))
                ins.code_block2 = self._gen(instr_set, block_size2, depth + 1)
        return ins


# ------------------------------------------------------------- size guards


def truncate_program_to_max(prog: List[Instruction], max_total: int = MAX_PROG_LEN) -> List[Instruction]:
    """Iteratively drop trailing instructions until total length ≤ max_total.

    Nested blocks count toward the total (see `program_length`).
    """
    while program_length(prog) > max_total and prog:
        prog = prog[:-1]
    return prog


__all__ = [
    "RandomProgramGenerator",
    "V2C_INSTR",
    "C2V_INSTR",
    "ALL_NAMES",
    "truncate_program_to_max",
]
