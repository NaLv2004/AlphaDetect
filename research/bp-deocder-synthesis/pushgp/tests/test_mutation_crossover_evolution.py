"""PR4 tests: mutation closure, crossover closure, selection, evolution loop."""

from __future__ import annotations

import numpy as np
import pytest

from pushgp.crossover import crossover_genome, crossover_program
from pushgp.evolution import EvolutionConfig, evolve
from pushgp.genome import Genome, MAX_PROG_LEN, N_EVO_CONSTS
from pushgp.mutation import (
    mutate_genome,
    mutate_log_constants,
    mutate_program,
    mut_block,
    mut_delete,
    mut_grow,
    mut_insert,
    mut_point,
    mut_segment,
    mut_swap,
)
from pushgp.program import Instruction, program_length
from pushgp.random_program import C2V_INSTR, RandomProgramGenerator, V2C_INSTR
from pushgp.selection import lexicase_select, tournament_select
from pushgp.validators import _make_vm, _run, validate_genome
from pushgp.vm import VM


def I(name, *, b1=None, b2=None):
    return Instruction(name=name, code_block=b1, code_block2=b2)


def _hand_v2c():
    return [
        I("Env.GetChannelLLR"),
        I("Env.GetIncomingVec"),
        I("FVec.Len"),
        I("Exec.DoTimes", b1=[I("FVec.At"), I("Float.Add")]),
    ]


def _hand_c2v():
    return [
        I("Bool.False"),
        I("Env.GetIncomingVec"),
        I("FVec.Len"),
        I("Exec.DoTimes", b1=[I("FVec.At"), I("Float.Const0"),
                              I("Float.LT"), I("Bool.Xor")]),
        I("Float.Const1"),
        I("Float.Exp"),
        I("Env.GetIncomingVec"),
        I("FVec.Len"),
        I("Exec.DoTimes", b1=[I("FVec.At"), I("Float.Abs"), I("Float.Min")]),
        I("Float.EvoConst0"),
        I("Float.Sub"),
        I("Float.Const0"),
        I("Float.Max"),
        I("Exec.If", b1=[I("Float.Neg")], b2=[]),
    ]


# =============================================================== Mutation


@pytest.mark.parametrize("op", [mut_point, mut_insert, mut_delete, mut_swap,
                                mut_block, mut_segment, mut_grow])
def test_mutation_closure_v2c(op):
    rng = np.random.default_rng(0)
    gen = RandomProgramGenerator(rng=rng)
    prog = _hand_v2c()
    for _ in range(40):
        out = op(prog, rng, gen, V2C_INSTR)
        assert isinstance(out, list)
        for ins in out:
            assert isinstance(ins, Instruction)
        assert program_length(out) <= MAX_PROG_LEN


def test_mutate_program_chain():
    rng = np.random.default_rng(1)
    gen = RandomProgramGenerator(rng=rng)
    prog = _hand_v2c()
    p = mutate_program(prog, rng, gen, V2C_INSTR, n_mutations=10)
    assert isinstance(p, list)
    assert program_length(p) <= MAX_PROG_LEN


def test_mutate_log_constants_in_range():
    rng = np.random.default_rng(0)
    c = np.zeros(N_EVO_CONSTS)
    out = mutate_log_constants(c, rng, n_tweaks=20, sigma=5.0)  # large sigma
    assert (out >= -3.0 - 1e-9).all()
    assert (out <= 3.0 + 1e-9).all()


def test_mutate_genome_runs_validator_input_independent():
    """mutate_genome must produce a Genome with same structure even if
    no validator is run."""
    rng = np.random.default_rng(2)
    gen = RandomProgramGenerator(rng=rng)
    g = Genome(prog_v2c=_hand_v2c(), prog_c2v=_hand_c2v(),
               log_constants=np.full(N_EVO_CONSTS, np.log10(0.25)))
    g2 = mutate_genome(g, rng, gen, n_mutations=2, p_const_tweak=1.0)
    assert isinstance(g2, Genome)
    # Original untouched.
    assert len(g.prog_v2c) == 4


# =============================================================== Crossover


def test_crossover_program_yields_list():
    rng = np.random.default_rng(0)
    p1 = _hand_v2c()
    p2 = [I("Float.Const1"), I("Float.Const2"), I("Float.Add")]
    for _ in range(40):
        c = crossover_program(p1, p2, rng)
        assert isinstance(c, list)
        for ins in c:
            assert isinstance(ins, Instruction)
        assert program_length(c) <= MAX_PROG_LEN


def test_crossover_genome_blends_constants():
    rng = np.random.default_rng(0)
    g1 = Genome(prog_v2c=_hand_v2c(), prog_c2v=_hand_c2v(),
                log_constants=np.full(N_EVO_CONSTS, -3.0))
    g2 = Genome(prog_v2c=_hand_v2c(), prog_c2v=_hand_c2v(),
                log_constants=np.full(N_EVO_CONSTS, 3.0))
    c = crossover_genome(g1, g2, rng)
    assert (c.log_constants >= -3.0 - 1e-9).all()
    assert (c.log_constants <= 3.0 + 1e-9).all()


# =============================================================== Selection


def test_tournament_select_picks_min():
    pop = ["a", "b", "c", "d"]
    fits = [3.0, 1.0, 2.0, 4.0]
    rng = np.random.default_rng(0)
    counts = {p: 0 for p in pop}
    for _ in range(200):
        winner = tournament_select(pop, fits, k=4, rng=rng)
        counts[winner] += 1
    # k = n forces all 4 in tournament; "b" should always win.
    assert counts["b"] == 200


def test_tournament_select_handles_inf():
    rng = np.random.default_rng(0)
    w = tournament_select(["x", "y"], [float("inf"), 1.0], k=2, rng=rng)
    assert w == "y"


def test_lexicase_select_basic():
    rng = np.random.default_rng(0)
    # Two individuals, three cases. Ind 0 wins 2/3, ind 1 wins 1/3.
    cf = np.array([[0.0, 0.0, 1.0],
                   [1.0, 1.0, 0.0]])
    counts = [0, 0]
    for _ in range(300):
        w = lexicase_select([0, 1], cf, rng)
        counts[w] += 1
    assert counts[0] > counts[1]


# =============================================================== Evolution


def _toy_fitness(genome: Genome) -> float:
    """Toy fitness: how close V2C output is to 5.7 on a fixed input."""
    target = 5.7
    incoming = np.array([0.5, -0.3, 1.1, -0.7, 0.9, 0.2, -0.4])
    vm = _make_vm(incoming, channel_llr=0.0, deg=8, evo_consts=genome.evo_const_values())
    out = _run(genome.prog_v2c, vm, "v2c")
    if out is None:
        return 1e6
    return float((out - target) ** 2)


def test_evolution_smoke_improves_or_plateaus():
    seed_genome = Genome(
        prog_v2c=_hand_v2c(), prog_c2v=_hand_c2v(),
        log_constants=np.full(N_EVO_CONSTS, np.log10(0.25)),
    )
    cfg = EvolutionConfig(
        pop_size=12,
        generations=4,
        elitism=2,
        tournament_k=3,
        n_mutations=2,
        max_retries=10,
        seed=42,
    )
    res = evolve(_toy_fitness, [seed_genome], cfg)

    assert len(res.history) == cfg.generations
    # Elitism guarantees monotonic non-increase of best fitness.
    best_curve = [g.best_fit for g in res.history]
    for i in range(1, len(best_curve)):
        assert best_curve[i] <= best_curve[i - 1] + 1e-12, \
            f"elitism violated: {best_curve}"
    assert isinstance(res.best_genome, Genome)
    # The seed produces L_v + sum(incoming) = 0 + 1.3 = 1.3, target = 5.7,
    # initial fitness ~ (5.7 - 1.3)^2 = 19.36.  Evolution should not be
    # WORSE than the seed.
    assert res.best_fitness <= 19.5


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
