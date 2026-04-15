"""
Evolutionary Engine for MIMO-Push GP.
Implements random program generation, mutation, crossover, and
multi-objective selection (lexicase-inspired).
"""
import numpy as np
import copy
import random
from typing import List, Tuple, Optional
from vm import Instruction, PRIMITIVE_INSTRUCTIONS, CONTROL_INSTRUCTIONS, program_to_string, program_to_oneliner


SEED_PROGRAMS = [
    [Instruction('Node.GetDistance')],
    [Instruction('Node.GetData')],
    [Instruction('Node.GetDistance'), Instruction('Node.GetLayer'), Instruction('Float.FromInt'), Instruction('Float.Add')],
    [Instruction('Node.GetChildren'), Instruction('Float.FromInt'), Instruction('Node.GetDistance'), Instruction('Float.Add')],
    [Instruction('Node.GetData'), Instruction('Vec.Norm2')],
    [Instruction('Graph.GetRoot'), Instruction('Node.GetChildren'), Instruction('Float.FromInt')],
    [Instruction('Node.GetDistance'), Instruction('Node.GetState0'), Instruction('Float.ConstHalf'), Instruction('Float.Mul'), Instruction('Float.Add')],
    [Instruction('Node.GetDistance'), Instruction('Node.GetState1'), Instruction('Float.ConstHalf'), Instruction('Float.Mul'), Instruction('Float.Add')],
    [Instruction('Node.GetDistance'), Instruction('Node.GetState1'), Instruction('Float.Mul'), Instruction('Float.ConstHalf'), Instruction('Float.Add')],
    [Instruction('Node.GetDistance'), Instruction('Node.GetState6'), Instruction('Float.Add')],
    [Instruction('Node.GetState6'), Instruction('Node.GetState0'), Instruction('Float.Add')],
    [Instruction('Node.GetSiblingCount'), Instruction('Float.FromInt'), Instruction('Node.GetDistance'), Instruction('Float.Add')],
    [Instruction('Graph.MapReduceSiblings', code_block=[Instruction('Node.GetScore')]), Instruction('Float.ConstHalf'), Instruction('Float.Mul'), Instruction('Node.GetDistance'), Instruction('Float.Add')],
]


def random_instruction(max_block_depth: int = 0, rng: np.random.RandomState = None) -> Instruction:
    """Generate a random instruction. Control flow instructions get random sub-blocks."""
    if rng is None:
        rng = np.random.RandomState()

    # Bias towards primitives; control flow less likely and only if depth allows
    if max_block_depth > 0 and rng.random() < 0.15:
        # Generate a control flow instruction
        ctrl = rng.choice(CONTROL_INSTRUCTIONS)
        block1 = random_block(rng.randint(1, 4), max_block_depth - 1, rng)
        if ctrl == 'Exec.If':
            block2 = random_block(rng.randint(1, 4), max_block_depth - 1, rng)
            return Instruction(ctrl, code_block=block1, code_block2=block2)
        else:
            return Instruction(ctrl, code_block=block1)
    else:
        name = rng.choice(PRIMITIVE_INSTRUCTIONS)
        return Instruction(name)


def random_block(size: int, max_depth: int = 2, rng: np.random.RandomState = None) -> List[Instruction]:
    """Generate a random block of instructions."""
    return [random_instruction(max_depth, rng) for _ in range(size)]


def random_program(min_size: int = 5, max_size: int = 30, max_depth: int = 2,
                   rng: np.random.RandomState = None) -> List[Instruction]:
    """Generate a random Push program."""
    if rng is None:
        rng = np.random.RandomState()
    size = rng.randint(min_size, max_size + 1)
    return random_block(size, max_depth, rng)


def seeded_programs() -> List[List[Instruction]]:
    """Primitive-only seed programs that produce valid Float outputs."""
    return [deep_copy_program(program) for program in SEED_PROGRAMS]


def program_length(program: List[Instruction]) -> int:
    """Count total instructions in a program (recursively counting blocks)."""
    count = 0
    for ins in program:
        count += 1
        if ins.code_block:
            count += program_length(ins.code_block)
        if ins.code_block2:
            count += program_length(ins.code_block2)
    return count


def program_relational_ratio(program: List[Instruction]) -> float:
    """Measure how much a program relies on dynamic graph-relative state."""
    relational_ops = {
        'Graph.GetLastExpanded', 'Graph.OpenAt',
        'Graph.MapReduce', 'Graph.MapReduceOpen', 'Graph.MapReduceSiblings', 'Graph.MapReduceAncestors',
        'Node.GetScore', 'Node.GetExpansionCount',
        'Node.GetSubtreeSize', 'Node.GetOpenDescendants', 'Node.GetCompleteDescendants',
        'Node.GetSiblingCount', 'Node.SiblingAt',
        'Node.GetState0', 'Node.GetState1', 'Node.GetState2', 'Node.GetState3',
        'Node.GetState4', 'Node.GetState5', 'Node.GetState6', 'Node.GetState7',
    }

    def _count(block: List[Instruction]) -> int:
        total = 0
        for ins in block:
            if ins.name in relational_ops:
                total += 1
            if ins.code_block:
                total += _count(ins.code_block)
            if ins.code_block2:
                total += _count(ins.code_block2)
        return total

    total_length = max(1, program_length(program))
    return _count(program) / float(total_length)


def flatten_program(program: List[Instruction]) -> List[Instruction]:
    """Flatten a program to a list of all instructions (top-level only pointers)."""
    result = []
    for ins in program:
        result.append(ins)
    return result


def deep_copy_program(program: List[Instruction]) -> List[Instruction]:
    """Deep copy a program."""
    return copy.deepcopy(program)


# ============================================================
# Mutation operators
# ============================================================

def mutate_point(program: List[Instruction], rng: np.random.RandomState) -> List[Instruction]:
    """Replace a random instruction with a new random one."""
    prog = deep_copy_program(program)
    if not prog:
        return prog
    idx = rng.randint(0, len(prog))
    prog[idx] = random_instruction(max_block_depth=1, rng=rng)
    return prog


def mutate_insert(program: List[Instruction], rng: np.random.RandomState) -> List[Instruction]:
    """Insert a random instruction at a random position."""
    prog = deep_copy_program(program)
    if program_length(prog) >= 60:  # max program size
        return prog
    idx = rng.randint(0, len(prog) + 1)
    prog.insert(idx, random_instruction(max_block_depth=1, rng=rng))
    return prog


def mutate_delete(program: List[Instruction], rng: np.random.RandomState) -> List[Instruction]:
    """Delete a random instruction."""
    prog = deep_copy_program(program)
    if len(prog) <= 2:
        return prog
    idx = rng.randint(0, len(prog))
    prog.pop(idx)
    return prog


def mutate_swap(program: List[Instruction], rng: np.random.RandomState) -> List[Instruction]:
    """Swap two random instructions."""
    prog = deep_copy_program(program)
    if len(prog) < 2:
        return prog
    i, j = rng.choice(len(prog), 2, replace=False)
    prog[i], prog[j] = prog[j], prog[i]
    return prog


def mutate_block(program: List[Instruction], rng: np.random.RandomState) -> List[Instruction]:
    """Replace a control flow block's body with new random code."""
    prog = deep_copy_program(program)
    # Find control flow instructions
    ctrl_indices = [i for i, ins in enumerate(prog) if ins.name in CONTROL_INSTRUCTIONS]
    if not ctrl_indices:
        # Insert a new control flow instruction
        return mutate_insert(program, rng)
    idx = rng.choice(ctrl_indices)
    prog[idx].code_block = random_block(rng.randint(1, 5), max_depth=1, rng=rng)
    if prog[idx].name == 'Exec.If':
        prog[idx].code_block2 = random_block(rng.randint(1, 5), max_depth=1, rng=rng)
    return prog


MUTATION_OPS = [mutate_point, mutate_insert, mutate_delete, mutate_swap, mutate_block]


def mutate(program: List[Instruction], rng: np.random.RandomState,
           n_mutations: int = 1) -> List[Instruction]:
    """Apply n random mutations to a program."""
    prog = program
    for _ in range(n_mutations):
        op = rng.choice(MUTATION_OPS)
        prog = op(prog, rng)
    return prog


def crossover(parent1: List[Instruction], parent2: List[Instruction],
              rng: np.random.RandomState) -> List[Instruction]:
    """Single-point crossover between two programs."""
    p1 = deep_copy_program(parent1)
    p2 = deep_copy_program(parent2)
    if not p1 or not p2:
        return p1 or p2
    cut1 = rng.randint(0, len(p1))
    cut2 = rng.randint(0, len(p2))
    child = p1[:cut1] + p2[cut2:]
    # Limit size
    if program_length(child) > 60:
        child = child[:40]
    return child


# ============================================================
# Fitness evaluation
# ============================================================

class FitnessResult:
    """Multi-objective fitness result."""
    def __init__(self, ber: float, mse: float, avg_flops: float,
                 code_length: int, frac_faults: float,
                 baseline_ber: float = 1.0, ber_ratio: float = 1.0,
                 dynamic_delta: float = 0.0, relational_score: float = 0.0,
                 generalization_gap: float = 0.0):
        self.ber = ber
        self.mse = mse
        self.avg_flops = avg_flops
        self.code_length = code_length
        self.frac_faults = frac_faults  # fraction of evaluations that hit HardwareFault
        self.baseline_ber = baseline_ber
        self.ber_ratio = ber_ratio
        self.dynamic_delta = dynamic_delta
        self.relational_score = relational_score
        self.generalization_gap = generalization_gap

    def dominates(self, other: 'FitnessResult') -> bool:
        """Pareto dominance check."""
        return (
            self.ber <= other.ber and
            self.ber_ratio <= other.ber_ratio and
            self.avg_flops <= other.avg_flops and
            self.generalization_gap <= other.generalization_gap and
            self.dynamic_delta >= other.dynamic_delta and
            (
                self.ber < other.ber or
                self.ber_ratio < other.ber_ratio or
                self.avg_flops < other.avg_flops or
                self.generalization_gap < other.generalization_gap
            )
        )

    def composite_score(self) -> float:
        """Single composite score for ranking. Lower is better."""
        fault_penalty = 1000.0 * self.frac_faults
        baseline_penalty = 8.0 * max(0.0, self.ber - 0.85 * self.baseline_ber)
        ratio_penalty = 12.0 * max(0.0, self.ber_ratio - 0.75)
        dynamic_penalty = 2.0 * max(0.0, 0.05 - self.dynamic_delta)
        relational_penalty = 1.5 * max(0.0, 0.20 - self.relational_score)
        generalization_penalty = 6.0 * self.generalization_gap
        complexity_penalty = 5e-6 * self.avg_flops
        return (
            self.ber +
            fault_penalty +
            baseline_penalty +
            ratio_penalty +
            dynamic_penalty +
            relational_penalty +
            generalization_penalty +
            complexity_penalty +
            1e-6 * self.code_length
        )

    def __repr__(self):
        return (f"Fitness(BER={self.ber:.4f}, MSE={self.mse:.6f}, "
                f"BER/LMMSE={self.ber_ratio:.3f}, dyn={self.dynamic_delta:.3f}, "
                f"rel={self.relational_score:.2f}, gap={self.generalization_gap:.3f}, FLOPs={self.avg_flops:.0f}, "
                f"len={self.code_length}, faults={self.frac_faults:.2f})")


class Individual:
    """An individual in the population: a Push program + its fitness."""
    def __init__(self, program: List[Instruction], fitness: Optional[FitnessResult] = None):
        self.program = program
        self.fitness = fitness
        self.age = 0

    def __repr__(self):
        return f"Individual(len={program_length(self.program)}, fitness={self.fitness})"


# ============================================================
# Tournament selection
# ============================================================

def tournament_select(population: List[Individual], tournament_size: int,
                      rng: np.random.RandomState) -> Individual:
    """Tournament selection: pick best from random subset."""
    indices = rng.choice(len(population), min(tournament_size, len(population)), replace=False)
    candidates = [population[i] for i in indices]
    # Sort by composite score (lower is better)
    candidates.sort(key=lambda ind: ind.fitness.composite_score() if ind.fitness else float('inf'))
    return candidates[0]


def lexicase_select(population: List[Individual], rng: np.random.RandomState) -> Individual:
    """Simplified lexicase selection using [BER, FLOPs, code_length] as cases."""
    candidates = [ind for ind in population if ind.fitness is not None]
    if not candidates:
        return population[rng.randint(len(population))]
    fallback_candidates = list(candidates)

    # Randomly order the objectives
    objectives = ['ber', 'ber_ratio', 'avg_flops', 'dynamic_delta', 'code_length']
    rng.shuffle(objectives)

    for obj in objectives:
        if len(candidates) <= 1:
            break
        raw_values = [getattr(c.fitness, obj) for c in candidates]
        finite_pairs = [(c, v) for c, v in zip(candidates, raw_values) if np.isfinite(v)]
        if not finite_pairs:
            candidates = list(fallback_candidates)
            break
        filtered_candidates = [candidate for candidate, _ in finite_pairs]
        values = [value for _, value in finite_pairs]
        if obj == 'dynamic_delta':
            best_val = max(values)
            epsilon = 0.01 * (max(values) - min(values) + 1e-10)
            next_candidates = [c for c, v in zip(filtered_candidates, values) if v >= best_val - epsilon]
        else:
            best_val = min(values)
            epsilon = 0.01 * (max(values) - min(values) + 1e-10)
            next_candidates = [c for c, v in zip(filtered_candidates, values) if v <= best_val + epsilon]

        if next_candidates:
            candidates = next_candidates
        else:
            candidates = list(fallback_candidates)
            break

    if not candidates:
        if fallback_candidates:
            fallback_candidates.sort(key=lambda ind: ind.fitness.composite_score())
            top_k = fallback_candidates[:max(1, min(4, len(fallback_candidates)))]
            return top_k[rng.randint(len(top_k))]
        return population[rng.randint(len(population))]

    return candidates[rng.randint(len(candidates))]
