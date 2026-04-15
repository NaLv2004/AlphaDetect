"""
Evolutionary Engine v2 for MIMO-Push GP.
Truly from-scratch: NO seed programs, NO domain-specific fitness bonuses.
Population starts 100 % random.  Selection pressure is pure BER + FLOPs + parsimony.
"""
import copy
import random
import numpy as np
from typing import List, Optional
from vm import (Instruction, PRIMITIVE_INSTRUCTIONS, CONTROL_INSTRUCTIONS,
                program_to_string, program_to_oneliner)


# --------------------------------------------------------------------------
# Random program generation
# --------------------------------------------------------------------------

# Environment-access instructions (access R matrix, y_tilde, etc.)
_ENV_ACCESS_INSTRUCTIONS = [
    'Mat.ElementAt', 'Mat.PeekAt', 'Mat.PeekAtIm', 'Mat.Row', 'Mat.VecMul',
    'Vec.ElementAt', 'Vec.PeekAt', 'Vec.PeekAtIm', 'Vec.SecondPeekAt', 'Vec.SecondPeekAtIm',
    'Vec.Dot', 'Vec.Norm2', 'Vec.Add', 'Vec.Sub', 'Vec.Scale',
    'Vec.GetResidue',      # high-level: y_tilde[0:k] - R[0:k,k:] @ x_partial
    'Float.GetMMSELB',     # admissible lower bound on remaining distance via MMSE projection
    'Node.GetCumDist', 'Node.GetLocalDist', 'Node.GetLayer',
    'Node.GetSymRe', 'Node.GetSymIm',
    'Node.ReadMem', 'Node.WriteMem',
    # BP-essential: score access and tree structure queries
    'Node.GetScore', 'Node.SetScore', 'Node.IsExpanded',
    'Matrix.Dup', 'Vector.Dup', 'Int.Dup',
    'Exec.ForEachSymbol', 'Exec.MinOverSymbols',
    'Float.GetNoiseVar', 'Int.GetNumSymbols',
]

def random_instruction(max_block_depth: int = 0,
                       rng: np.random.RandomState = None,
                       env_bias: float = 0.0) -> Instruction:
    if rng is None:
        rng = np.random.RandomState()
    if max_block_depth > 0 and rng.random() < 0.12:
        ctrl = rng.choice(CONTROL_INSTRUCTIONS)
        b1 = random_block(rng.randint(1, 5), max_block_depth - 1, rng)
        if ctrl == 'Exec.If':
            b2 = random_block(rng.randint(1, 5), max_block_depth - 1, rng)
            return Instruction(ctrl, code_block=b1, code_block2=b2)
        return Instruction(ctrl, code_block=b1)
    # With env_bias probability, prefer environment-access instructions
    if env_bias > 0 and rng.random() < env_bias:
        return Instruction(rng.choice(_ENV_ACCESS_INSTRUCTIONS))
    return Instruction(rng.choice(PRIMITIVE_INSTRUCTIONS))


def random_block(size: int, max_depth: int = 2,
                 rng: np.random.RandomState = None,
                 env_bias: float = 0.0) -> List[Instruction]:
    return [random_instruction(max_depth, rng, env_bias) for _ in range(size)]


def random_program(min_size: int = 3, max_size: int = 20, max_depth: int = 2,
                   rng: np.random.RandomState = None,
                   env_bias: float = 0.0) -> List[Instruction]:
    if rng is None:
        rng = np.random.RandomState()
    size = rng.randint(min_size, max_size + 1)
    return random_block(size, max_depth, rng, env_bias)


# --------------------------------------------------------------------------
# BP-pattern program generation
# --------------------------------------------------------------------------
# These generators inject programs that contain BiDiRectional message passing
# building blocks. They are NOT hand-designed algorithms — the scoring logic
# is random. But they provide the RAW GENETIC MATERIAL (SetScore inside
# tree traversal loops) that evolution can refine into real BP.

_BP_SCORE_SOURCES = [
    'Node.GetCumDist', 'Node.GetLocalDist', 'Node.GetScore',
    'Float.GetMMSELB', 'Vec.Norm2', 'Float.Const1',
]

_BP_AGGREGATES = [
    'Float.Add', 'Float.Sub', 'Float.Mul', 'Float.Min', 'Float.Max',
    'Float.Neg', 'Float.Inv', 'Float.Abs',
]

_BP_TRAVERSALS = [
    'Node.ForEachSibling', 'Node.ForEachChild', 'Node.ForEachAncestor',
]


def random_bp_pattern(rng: np.random.RandomState) -> List[Instruction]:
    """Generate a random program that CONTAINS BP building blocks.

    Critical: Inside ForEach* blocks, SetScore needs a Float that was
    pushed WITHIN the same iteration (not relying on outer stack).
    So the block must push at least 1 Float before calling SetScore.

    Structure: [preamble] + [traversal with self-contained SetScore] + [tail]
    """
    prog = []

    # Random preamble (2-5 instructions)
    for _ in range(rng.randint(2, 6)):
        prog.append(random_instruction(1, rng, env_bias=0.5))

    # Choose a BP pattern variant — all have self-contained Float pushes
    variant = rng.randint(0, 6)

    if variant == 0:
        # Pattern: ForEachSibling([push_float, SetScore])
        # Minimal: read sibling's cum_dist, set as new score
        src = rng.choice(_BP_SCORE_SOURCES)
        inner = [Instruction(src), Instruction('Node.SetScore')]
        prog.append(Instruction('Node.ForEachSibling', code_block=inner))

    elif variant == 1:
        # Pattern: ForEachSibling([push_A, push_B, combine, SetScore])
        # Two floats combined then used as score
        s1 = rng.choice(_BP_SCORE_SOURCES)
        s2 = rng.choice(_BP_SCORE_SOURCES)
        op = rng.choice(['Float.Add', 'Float.Sub', 'Float.Mul', 'Float.Min'])
        inner = [Instruction(s1), Instruction(s2),
                 Instruction(op), Instruction('Node.SetScore')]
        prog.append(Instruction('Node.ForEachSibling', code_block=inner))

    elif variant == 2:
        # Pattern: GetParent ; ForEachChild([push_float, SetScore])
        # Navigate to parent, then iterate ITS children (= siblings + self)
        prog.append(Instruction('Node.GetParent'))
        src = rng.choice(_BP_SCORE_SOURCES)
        s2 = rng.choice(_BP_SCORE_SOURCES)
        op = rng.choice(_BP_AGGREGATES)
        inner = [Instruction(src), Instruction(s2),
                 Instruction(op), Instruction('Node.SetScore')]
        prog.append(Instruction('Node.ForEachChild', code_block=inner))

    elif variant == 3:
        # Pattern: ForEachAncestor([ForEachChild([read_score]), transform, SetScore])
        # Nested: for each ancestor, aggregate its children's info, update ancestor
        src = rng.choice(_BP_SCORE_SOURCES)
        inner_child = [Instruction(src)]  # just read, accumulate via mapreduce sum
        # outer block: ForEachChild pushes sum → transform → SetScore
        transform = rng.choice(['Float.Neg', 'Float.Abs', 'Float.Inv',
                                'Float.Sqrt', 'Float.Exp'])
        inner_anc = [
            Instruction('Node.ForEachChild', code_block=inner_child),
            Instruction(transform),
            Instruction('Node.SetScore'),
        ]
        prog.append(Instruction('Node.ForEachAncestor', code_block=inner_anc))

    elif variant == 4:
        # Pattern: GetParent ; GetParent ; ForEachSibling([push, SetScore])
        # Go up 2 levels, modify uncle scores
        prog.append(Instruction('Node.GetParent'))
        prog.append(Instruction('Node.GetParent'))
        src = rng.choice(_BP_SCORE_SOURCES)
        inner = [Instruction(src), Instruction('Node.SetScore')]
        prog.append(Instruction('Node.ForEachSibling', code_block=inner))

    else:
        # Pattern: ForEachSibling([GetScore, GetCumDist, Sub, SetScore])
        # Compute score correction based on cum_dist deviation
        inner = [
            Instruction('Node.GetScore'),
            Instruction('Node.GetCumDist'),
            Instruction(rng.choice(['Float.Sub', 'Float.Add', 'Float.Min'])),
            Instruction('Node.SetScore'),
        ]
        trav = rng.choice(['Node.ForEachSibling', 'Node.ForEachChild'])
        prog.append(Instruction(trav, code_block=inner))

    # Random tail (1-4 instructions for the correction value)
    for _ in range(rng.randint(1, 5)):
        prog.append(random_instruction(1, rng, env_bias=0.5))

    return prog


# --------------------------------------------------------------------------
# Program utilities
# --------------------------------------------------------------------------

def program_length(prog: List[Instruction]) -> int:
    c = 0
    for ins in prog:
        c += 1
        if ins.code_block:
            c += program_length(ins.code_block)
        if ins.code_block2:
            c += program_length(ins.code_block2)
    return c


def deep_copy_program(prog: List[Instruction]) -> List[Instruction]:
    return copy.deepcopy(prog)


# --------------------------------------------------------------------------
# Mutation operators
# --------------------------------------------------------------------------
MAX_PROG_LEN = 80


def mutate_point(prog, rng):
    p = deep_copy_program(prog)
    if not p:
        return p
    p[rng.randint(len(p))] = random_instruction(1, rng, env_bias=0.3)
    return p


def mutate_insert(prog, rng):
    p = deep_copy_program(prog)
    if program_length(p) >= MAX_PROG_LEN:
        return p
    idx = rng.randint(0, len(p) + 1)
    p.insert(idx, random_instruction(1, rng, env_bias=0.3))
    return p


def mutate_delete(prog, rng):
    p = deep_copy_program(prog)
    if len(p) <= 2:
        return p
    p.pop(rng.randint(len(p)))
    return p


def mutate_swap(prog, rng):
    p = deep_copy_program(prog)
    if len(p) < 2:
        return p
    i, j = rng.choice(len(p), 2, replace=False)
    p[i], p[j] = p[j], p[i]
    return p


def mutate_block(prog, rng):
    p = deep_copy_program(prog)
    ctrl_idxs = [i for i, ins in enumerate(p) if ins.name in CONTROL_INSTRUCTIONS]
    if not ctrl_idxs:
        return mutate_insert(prog, rng)
    idx = rng.choice(ctrl_idxs)
    p[idx].code_block = random_block(rng.randint(1, 6), 1, rng, env_bias=0.3)
    if p[idx].name == 'Exec.If':
        p[idx].code_block2 = random_block(rng.randint(1, 6), 1, rng, env_bias=0.3)
    return p


def mutate_segment(prog, rng):
    """Replace a contiguous segment with fresh random code."""
    p = deep_copy_program(prog)
    if len(p) < 3:
        # For short programs, append random segment instead
        ext = random_block(rng.randint(2, 6), 1, rng, env_bias=0.4)
        p = p + ext
        if program_length(p) > MAX_PROG_LEN:
            p = p[:MAX_PROG_LEN // 2]
        return p
    start = rng.randint(0, len(p) - 1)
    end = min(len(p), start + rng.randint(1, 4))
    replacement = random_block(rng.randint(1, 5), 1, rng, env_bias=0.3)
    p[start:end] = replacement
    if program_length(p) > MAX_PROG_LEN:
        p = p[:MAX_PROG_LEN // 2]
    return p


def mutate_grow(prog, rng):
    """Grow a program by prepending or appending a random block."""
    p = deep_copy_program(prog)
    n_new = rng.randint(2, 8)
    block = random_block(n_new, 1, rng, env_bias=0.4)
    if rng.rand() < 0.5:
        p = block + p  # prepend
    else:
        p = p + block  # append
    if program_length(p) > MAX_PROG_LEN:
        p = p[:MAX_PROG_LEN // 2]
    return p


_MUTATION_OPS = [mutate_point, mutate_insert, mutate_delete,
                 mutate_swap, mutate_block, mutate_segment, mutate_grow]


def mutate(prog: List[Instruction], rng: np.random.RandomState,
           n_mutations: int = 1) -> List[Instruction]:
    p = prog
    for _ in range(n_mutations):
        p = rng.choice(_MUTATION_OPS)(p, rng)
    return p


# --------------------------------------------------------------------------
# Crossover
# --------------------------------------------------------------------------

def crossover(p1: List[Instruction], p2: List[Instruction],
              rng: np.random.RandomState) -> List[Instruction]:
    a = deep_copy_program(p1)
    b = deep_copy_program(p2)
    if not a or not b:
        return a or b
    c1 = rng.randint(0, len(a))
    c2 = rng.randint(0, len(b))
    child = a[:c1] + b[c2:]
    if program_length(child) > MAX_PROG_LEN:
        child = child[: MAX_PROG_LEN // 2]
    return child


# --------------------------------------------------------------------------
# Fitness
# --------------------------------------------------------------------------

class FitnessResult:
    """Performance-based fitness with measured BP utility."""

    def __init__(self, ber: float, mse: float, avg_flops: float,
                 code_length: int, frac_faults: float,
                 baseline_ber: float = 1.0, ber_ratio: float = 1.0,
                 generalization_gap: float = 0.0,
                 bp_updates: float = 0.0,
                 nonlocal_bp_updates: float = 0.0,
                 bp_gain: float = 0.0):
        self.ber = ber
        self.mse = mse
        self.avg_flops = avg_flops
        self.code_length = code_length
        self.frac_faults = frac_faults
        self.baseline_ber = baseline_ber
        self.ber_ratio = ber_ratio
        self.generalization_gap = generalization_gap
        self.bp_updates = bp_updates  # avg dirty-node updates per sample
        self.nonlocal_bp_updates = nonlocal_bp_updates
        self.bp_gain = bp_gain

    def composite_score(self) -> float:
        fault_pen = 500.0 * self.frac_faults
        # Reduced gen_pen: was 2.0 → 0.3 to prioritize BER over gap.
        # At 2.0, a genome with BER=0.046 but gap=0.038 was rejected in
        # favor of BER=0.067 but gap=0.002 — losing a 44% BER improvement.
        gen_pen = 0.3 * self.generalization_gap
        # Reduced 1e-9 → 1e-11: at 1e-9, FLOPs=40M gives penalty 0.04
        # which is 66% of BER=0.06 — severely distorting fitness.
        # At 1e-11, FLOPs=40M → 0.0004 (< 1% of BER), FLOPs=1B → 0.01.
        complexity_pen = 1e-11 * self.avg_flops
        # Reward only BP that improves BER over a no-write ablation.
        # Nonlocal score writes without measurable gain are penalized.
        bp_term = 0.0
        if self.nonlocal_bp_updates > 0:
            if self.bp_gain > 0:
                bp_term = -min(0.03, 0.5 * self.bp_gain)
            else:
                bp_term = 0.02 * min(1.0, self.nonlocal_bp_updates / 20.0)
        return self.ber + fault_pen + gen_pen + complexity_pen + bp_term

    def __repr__(self):
        return (f"Fit(BER={self.ber:.5f} ratio={self.ber_ratio:.3f} "
                f"FLOPs={self.avg_flops:.0f} len={self.code_length} "
                f"faults={self.frac_faults:.2f} gap={self.generalization_gap:.3f} "
                f"bp={self.bp_updates:.1f} nlbp={self.nonlocal_bp_updates:.1f} "
                f"bpg={self.bp_gain:.3f})")


class Individual:
    def __init__(self, program: List[Instruction],
                 fitness: Optional[FitnessResult] = None):
        self.program = program
        self.fitness = fitness
        self.age = 0

    def __repr__(self):
        return f"Ind(len={program_length(self.program)}, fit={self.fitness})"


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def tournament_select(pop: List[Individual], size: int,
                      rng: np.random.RandomState) -> Individual:
    idxs = rng.choice(len(pop), min(size, len(pop)), replace=False)
    cands = [pop[i] for i in idxs]
    cands.sort(key=lambda x: x.fitness.composite_score() if x.fitness else 1e9)
    return cands[0]


def lexicase_select(pop: List[Individual],
                    rng: np.random.RandomState) -> Individual:
    cands = [ind for ind in pop if ind.fitness is not None]
    if not cands:
        return pop[rng.randint(len(pop))]
    objs = ['ber', 'avg_flops', 'code_length', 'frac_faults']
    rng.shuffle(objs)
    for obj in objs:
        if len(cands) <= 1:
            break
        vals = [getattr(c.fitness, obj) for c in cands]
        finite = [(c, v) for c, v in zip(cands, vals) if np.isfinite(v)]
        if not finite:
            break
        best = min(v for _, v in finite)
        eps = 0.01 * (max(v for _, v in finite) - best + 1e-10)
        nxt = [c for c, v in finite if v <= best + eps]
        if nxt:
            cands = nxt
    return cands[rng.randint(len(cands))]
