"""Tests for `pushgp.serialize` round-trip and tree metrics."""
from __future__ import annotations

import numpy as np

from pushgp.program import Instruction
from pushgp.random_program import RandomProgramGenerator
from pushgp.serialize import (
    dict_to_instr,
    dict_to_program,
    genome_to_dict,
    instr_to_dict,
    program_to_dict,
    tree_max_depth,
    tree_size,
)
from pushgp_ldpc.adapter import oms_seed_genome


def _eq_instr(a: Instruction, b: Instruction) -> bool:
    if a.name != b.name:
        return False
    for attr in ("code_block", "code_block2"):
        ax, bx = getattr(a, attr), getattr(b, attr)
        if (ax is None) != (bx is None):
            return False
        if ax is None:
            continue
        if len(ax) != len(bx):
            return False
        for x, y in zip(ax, bx):
            if not _eq_instr(x, y):
                return False
    return True


def test_atomic_round_trip():
    ins = Instruction(name="Float.Add")
    d = instr_to_dict(ins)
    assert d == {"name": "Float.Add"}
    assert _eq_instr(dict_to_instr(d), ins)


def test_nested_round_trip():
    inner = Instruction(name="Exec.DoTimes",
                        code_block=[Instruction("Float.Mul"),
                                    Instruction("Float.ConstPi")])
    outer = Instruction(name="Exec.If",
                        code_block=[inner, Instruction("Bool.Not")],
                        code_block2=[Instruction("Float.Sub")])
    d = instr_to_dict(outer)
    assert "code_block" in d and "code_block2" in d
    assert _eq_instr(dict_to_instr(d), outer)


def test_random_program_round_trip():
    gen = RandomProgramGenerator(rng=np.random.default_rng(7), max_recur_depth=2)
    from pushgp.random_program import V2C_INSTR
    prog = gen.random_program(V2C_INSTR, 5, 15)
    d = program_to_dict(prog)
    p2 = dict_to_program(d)
    assert len(p2) == len(prog)
    for a, b in zip(prog, p2):
        assert _eq_instr(a, b)


def test_genome_round_trip():
    g = oms_seed_genome()
    d = genome_to_dict(g)
    assert "prog_v2c" in d and "prog_c2v" in d and "log_constants" in d
    v2c = dict_to_program(d["prog_v2c"])
    c2v = dict_to_program(d["prog_c2v"])
    assert len(v2c) == len(g.prog_v2c)
    assert len(c2v) == len(g.prog_c2v)


def test_tree_size_atomic():
    assert tree_size([Instruction("X"), Instruction("Y")]) == 2


def test_tree_size_nested():
    p = [Instruction("Exec.DoTimes",
                     code_block=[Instruction("a"), Instruction("b")])]
    assert tree_size(p) == 3


def test_tree_max_depth_atomic_zero():
    assert tree_max_depth([Instruction("a"), Instruction("b")]) == 0


def test_tree_max_depth_nested():
    inner = Instruction("Exec.DoTimes", code_block=[Instruction("x")])
    outer = Instruction("Exec.DoTimes", code_block=[inner])
    assert tree_max_depth([outer]) == 2
