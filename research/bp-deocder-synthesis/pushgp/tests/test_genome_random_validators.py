"""Tests for genome JSON I/O, random program generation, and validators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pushgp.genome import (
    Genome,
    MAX_PROG_LEN,
    N_EVO_CONSTS,
    instruction_to_dict,
    dict_to_instruction,
)
from pushgp.program import Instruction, program_length
from pushgp.random_program import (
    C2V_INSTR,
    RandomProgramGenerator,
    V2C_INSTR,
    truncate_program_to_max,
)
from pushgp.validators import validate_c2v, validate_genome, validate_v2c
from pushgp.vm import VM


def I(name, *, b1=None, b2=None):
    return Instruction(name=name, code_block=b1, code_block2=b2)


# =================================================================== Genome IO


def test_instruction_roundtrip():
    ins = I("Exec.If", b1=[I("Float.Const1")], b2=[I("Float.Const2"), I("Float.Add")])
    d = instruction_to_dict(ins)
    back = dict_to_instruction(d)
    assert back.name == ins.name
    assert back.code_block[0].name == "Float.Const1"
    assert back.code_block2[1].name == "Float.Add"


def test_genome_save_load_roundtrip(tmp_path: Path):
    g = Genome(
        prog_v2c=[I("Env.GetChannelLLR")],
        prog_c2v=[I("Env.GetIncomingVec")],
        log_constants=np.array([0.1, 0.2, -0.3, 1.5, -1.7, 0.0, 0.5, -0.5]),
    )
    p = tmp_path / "g.json"
    g.save(p)
    g2 = Genome.load(p)
    assert g2.prog_v2c[0].name == "Env.GetChannelLLR"
    assert g2.prog_c2v[0].name == "Env.GetIncomingVec"
    assert np.allclose(g2.log_constants, g.log_constants)


def test_evo_const_values_clamped():
    g = Genome(log_constants=np.array([10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    v = g.evo_const_values()
    assert v[0] == 10**3
    assert v[1] == 10**-3


# ================================================================== Subsets


def test_v2c_subset_includes_channel_llr():
    assert "Env.GetChannelLLR" in V2C_INSTR
    assert "Env.GetChannelLLR" not in C2V_INSTR


def test_subsets_have_essential_atoms():
    # Sanity: arithmetic, vec ops, control flow, env all present.
    for needed in ("Float.Add", "FVec.Len", "FVec.At", "Exec.DoTimes",
                   "Env.GetIncomingVec", "Env.GetDeg"):
        assert needed in V2C_INSTR
        assert needed in C2V_INSTR


# ============================================================= RandomGenerator


def test_random_program_length_within_bounds():
    g = RandomProgramGenerator(rng=np.random.default_rng(0))
    p = g.random_v2c(min_size=4, max_size=10)
    # Top-level count is in [4, 10]; nested-control may push total higher.
    assert 4 <= len(p) <= 10


def test_random_genome_produces_valid_dataclass():
    g = RandomProgramGenerator(rng=np.random.default_rng(123))
    genome = g.random_genome(min_size=4, max_size=12)
    assert isinstance(genome, Genome)
    assert genome.log_constants.shape == (N_EVO_CONSTS,)
    assert (genome.log_constants >= -3.0 - 1e-9).all()
    assert (genome.log_constants <= 3.0 + 1e-9).all()


def test_random_program_runs_without_crash():
    """Smoke-test: 200 random programs all run to completion (or fault) without raising."""
    gen = RandomProgramGenerator(rng=np.random.default_rng(7))
    vm = VM()
    for _ in range(200):
        prog = gen.random_v2c(min_size=4, max_size=20)
        vm.reset()
        vm.state.ctx_channel_llr = 0.5
        vm.state.ctx_incoming = np.array([0.1, -0.2, 0.3, -0.4])
        vm.state.ctx_deg = 5
        vm.state.floats.push(0.5)
        vm.state.fvecs.push(vm.state.ctx_incoming.copy())
        vm.run(prog)  # may fault, must never raise


def test_truncate_program_to_max():
    longp = [I("Float.Const1") for _ in range(MAX_PROG_LEN + 50)]
    short = truncate_program_to_max(longp)
    assert program_length(short) <= MAX_PROG_LEN


# ============================================================== Validators


def _hand_v2c_oms_like():
    """Hand-written V2C: out = L_v + sum(incoming).  Should pass all checks."""
    return [
        I("Env.GetChannelLLR"),
        I("Env.GetIncomingVec"),
        I("FVec.Len"),
        I(
            "Exec.DoTimes",
            b1=[I("FVec.At"), I("Float.Add")],
        ),
    ]


def _hand_c2v_signprod_minabs_like():
    """Hand-written C2V: out = (Π sign) * min(|incoming|) - β.

    Implemented atomically:
      sign-product:
          Bool.True
          Env.GetIncomingVec, FVec.Len, DoTimes {
              FVec.At, Float.Const0, Float.LT,    # bool: m_i < 0
              Bool.Xor                            # toggle accumulator
          }
      Now bool stack top = (number of negatives is odd)
      Then -1 if True else 1, multiplied by min|m_i|, minus β=EvoConst0.
    """
    return [
        # --- product of signs as bool (True = negative) ---
        I("Bool.False"),
        I("Env.GetIncomingVec"),
        I("FVec.Len"),
        I(
            "Exec.DoTimes",
            b1=[
                I("FVec.At"),
                I("Float.Const0"),
                I("Float.LT"),  # bool: v[i] < 0
                I("Bool.Xor"),
            ],
        ),
        # --- min absolute value ---
        I("Float.Exp"),                           # filler so program isn't trivial
        I("Float.Pop"),
        I("Float.Const1"),
        I("Float.Exp"),                           # ~2.718, big positive seed
        I("Env.GetIncomingVec"),
        I("FVec.Len"),
        I(
            "Exec.DoTimes",
            b1=[
                I("FVec.At"),
                I("Float.Abs"),
                I("Float.Min"),
            ],
        ),
        # --- subtract beta = EvoConst0 ---
        I("Float.EvoConst0"),
        I("Float.Sub"),
        # --- max(0, .) for the OMS clamp ---
        I("Float.Const0"),
        I("Float.Max"),
        # --- apply sign: if Bool.True (negative count odd), negate ---
        I(
            "Exec.If",
            b1=[I("Float.Neg")],
            b2=[],
        ),
    ]


def test_validator_accepts_hand_v2c():
    prog = _hand_v2c_oms_like()
    ok, why = validate_v2c(prog, rng=np.random.default_rng(0))
    assert ok, f"hand V2C should validate, got: {why}"


def test_validator_accepts_hand_c2v():
    prog = _hand_c2v_signprod_minabs_like()
    ok, why = validate_c2v(prog, rng=np.random.default_rng(0), evo_consts=np.array([0.25]*8))
    assert ok, f"hand C2V should validate, got: {why}"


def test_validator_rejects_constant_v2c():
    prog = [I("Float.Const1")]
    ok, why = validate_v2c(prog, rng=np.random.default_rng(0))
    assert not ok
    assert "independent" in why or "faulty" in why


def test_validator_rejects_non_permutation_invariant():
    """V2C that uses incoming[0] only: depends on incoming but not invariant."""
    prog = [
        I("Env.GetChannelLLR"),
        I("Env.GetIncomingVec"),
        I("Int.Const0"),
        I("FVec.At"),
        I("Float.Add"),
    ]
    ok, why = validate_v2c(prog, rng=np.random.default_rng(0))
    assert not ok
    assert "permutation" in why or "independent" in why


def test_validator_rejects_v2c_ignoring_channel_llr():
    """Sum incoming, ignore L_v."""
    prog = [
        I("Env.GetIncomingVec"),
        I("FVec.Len"),
        I("Exec.DoTimes", b1=[I("FVec.At"), I("Float.Add")]),
    ]
    # The seeded float stack will have L_v on it but the program ignores it
    # (it pushes 0 from sum + L_v will be present and the final answer's value
    # depends on L_v because sum is added to it on the float stack).
    # So this actually DOES depend on L_v through the seeded value.  Replace
    # with a program that explicitly clears L_v:
    prog2 = [
        I("Float.Pop"),  # discard the seeded L_v
        I("Float.Const0"),
        I("Env.GetIncomingVec"),
        I("FVec.Len"),
        I("Exec.DoTimes", b1=[I("FVec.At"), I("Float.Add")]),
    ]
    ok, why = validate_v2c(prog2, rng=np.random.default_rng(0))
    assert not ok
    assert "L_v" in why or "faulty" in why or "independent" in why


def test_validate_genome_calls_both_sides():
    g = Genome(
        prog_v2c=_hand_v2c_oms_like(),
        prog_c2v=_hand_c2v_signprod_minabs_like(),
        log_constants=np.full(N_EVO_CONSTS, np.log10(0.25)),
    )
    ok, why = validate_genome(g, rng=np.random.default_rng(0))
    assert ok, why


# ===================== Random programs sometimes pass the validator =========


def test_random_genome_pass_rate_smoke():
    """Random genomes almost never satisfy the strict permutation-invariance
    constraint — that is by design and the GA solves it via seeding from
    OMS plus targeted mutation.  This test only checks that validation
    runs to completion on random genomes without raising."""
    gen = RandomProgramGenerator(rng=np.random.default_rng(2024))
    rng = np.random.default_rng(0)
    for _ in range(40):
        g = gen.random_genome(min_size=6, max_size=15)
        ok, why = validate_genome(g, rng=rng)
        assert isinstance(ok, bool)
        assert isinstance(why, str)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
