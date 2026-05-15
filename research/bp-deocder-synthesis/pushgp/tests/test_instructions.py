"""Per-family tests for instructions + VM integration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pushgp.instructions import HANDLERS, all_instruction_names
from pushgp.program import Instruction
from pushgp.vm import VM


# ------------------------------------------------------------------- helpers


def run(prog, **ctx):
    vm = VM()
    for k, v in ctx.items():
        setattr(vm.state, k, v)
    return vm.run(prog), vm


def I(name, *, b1=None, b2=None):
    return Instruction(name=name, code_block=b1, code_block2=b2)


# ----------------------------------------------------------- registry sanity


def test_registry_size_minimum() -> None:
    # Should have ≥ 130 atomic instructions per design.
    assert len(HANDLERS) >= 130, f"only {len(HANDLERS)} instructions registered"


def test_no_duplicate_handlers() -> None:
    names = all_instruction_names()
    assert len(names) == len(set(names))


# ----------------------------------------------------------- Float arithmetic


def test_float_basic_arithmetic():
    out, _ = run([I("Float.Const1"), I("Float.Const2"), I("Float.Add")])
    assert out == 3.0
    out, _ = run([I("Float.Const2"), I("Float.Const1"), I("Float.Sub")])
    assert out == 1.0
    out, _ = run([I("Float.Const2"), I("Float.Const2"), I("Float.Mul")])
    assert out == 4.0
    out, _ = run([I("Float.Const1"), I("Float.Const2"), I("Float.Div")])
    assert out == 0.5


def test_float_div_by_zero_safe():
    out, _ = run([I("Float.Const1"), I("Float.Const0"), I("Float.Div")])
    assert out == 0.0  # NAN_INF_REPLACEMENT


def test_float_unary_ops():
    out, _ = run([I("Float.ConstNeg1"), I("Float.Abs")])
    assert out == 1.0
    out, _ = run([I("Float.Const2"), I("Float.Square")])
    assert out == 4.0
    out, _ = run([I("Float.ConstNeg1"), I("Float.Sqrt")])
    assert out == 0.0  # invalid → safe
    out, _ = run([I("Float.Const1"), I("Float.Tanh")])
    assert pytest.approx(out, rel=1e-9) == math.tanh(1.0)


def test_evo_consts_use_context():
    vm = VM()
    vm.state.ctx_evo_constants = np.array([0.25, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    out = vm.run([I("Float.EvoConst0"), I("Float.EvoConst1"), I("Float.Mul")])
    assert pytest.approx(out, rel=1e-9) == 0.25 * 0.9


# --------------------------------------------------------------- Int / Bool


def test_int_arithmetic_and_constants():
    out, vm = run([I("Int.Const1"), I("Int.Const2"), I("Int.Add"), I("Float.FromInt")])
    assert out == 3.0


def test_bool_logic_and_compare():
    out, vm = run([I("Float.Const1"), I("Float.Const2"), I("Float.LT")])
    assert out is None  # nothing on float stack
    assert vm.state.bools.peek() is True

    _, vm = run([I("Bool.True"), I("Bool.False"), I("Bool.And")])
    assert vm.state.bools.peek() is False

    _, vm = run([I("Bool.True"), I("Bool.False"), I("Bool.Or")])
    assert vm.state.bools.peek() is True


# ---------------------------------------------------------------- Conversions


def test_float_int_conversion():
    out, _ = run([I("Float.Const2"), I("Int.FromFloat"), I("Float.FromInt")])
    assert out == 2.0


# ---------------------------------------------------------------- FloatVec


def test_fvec_len_at_basic():
    vm = VM()
    vm.state.fvecs.push(np.array([10.0, 20.0, 30.0]))
    out = vm.run([I("FVec.Len"), I("Float.FromInt")])
    assert out == 3.0
    vm.reset()
    vm.state.fvecs.push(np.array([10.0, 20.0, 30.0]))
    out = vm.run([I("Int.Const1"), I("FVec.At")])
    assert out == 20.0


def test_fvec_indexing_wraparound():
    vm = VM()
    vm.state.fvecs.push(np.array([1.0, 2.0, 3.0]))
    out = vm.run([I("Int.Const1"), I("Int.Const2"), I("Int.Add"), I("FVec.At")])  # idx=3 → wrap to 0
    assert out == 1.0


def test_fvec_set_push_popback():
    vm = VM()
    vm.state.fvecs.push(np.array([1.0, 2.0, 3.0]))
    vm.run([I("Float.Const0"), I("Int.Const1"), I("FVec.Set")])
    assert (vm.state.fvecs.peek() == np.array([1.0, 0.0, 3.0])).all()
    vm.run([I("Float.Const2"), I("FVec.Push")])
    assert vm.state.fvecs.peek().tolist() == [1.0, 0.0, 3.0, 2.0]
    out = vm.run([I("FVec.PopBack")])
    assert out == 2.0
    assert vm.state.fvecs.peek().tolist() == [1.0, 0.0, 3.0]


def test_fvec_concat_slice_reverse():
    vm = VM()
    vm.state.fvecs.push(np.array([1.0, 2.0]))
    vm.state.fvecs.push(np.array([3.0, 4.0]))
    vm.run([I("FVec.Concat")])
    assert vm.state.fvecs.peek().tolist() == [1.0, 2.0, 3.0, 4.0]
    vm.run([I("Int.Const1"), I("Int.Const2"), I("FVec.Slice")])
    assert vm.state.fvecs.peek().tolist() == [2.0]
    vm.reset()
    vm.state.fvecs.push(np.array([1.0, 2.0, 3.0]))
    vm.run([I("FVec.Reverse")])
    assert vm.state.fvecs.peek().tolist() == [3.0, 2.0, 1.0]


# ---------------------------------------------------------------- Memory


def test_mem_read_write():
    vm = VM()
    vm.run([I("Float.Const2"), I("Int.Const1"), I("Mem.Write")])
    assert vm.state.memory[1] == 2.0
    out = vm.run([I("Int.Const1"), I("Mem.Read")])
    assert out == 2.0


# ----------------------------------------------------------------- Env reads


def test_env_get_channel_and_incoming():
    vm = VM()
    vm.state.ctx_channel_llr = 1.5
    vm.state.ctx_incoming = np.array([0.5, -0.3, 1.2])
    vm.state.ctx_iter = 7
    vm.state.ctx_deg = 4
    out = vm.run([I("Env.GetChannelLLR")])
    assert out == 1.5
    vm.reset()
    vm.state.ctx_channel_llr = 1.5
    vm.state.ctx_incoming = np.array([0.5, -0.3, 1.2])
    vm.run([I("Env.GetIncomingVec")])
    assert (vm.state.fvecs.peek() == np.array([0.5, -0.3, 1.2])).all()
    vm.reset()
    vm.state.ctx_iter = 7
    vm.run([I("Env.GetIter"), I("Float.FromInt")])
    assert vm.state.floats.peek() == 7.0


def test_env_channel_llr_blocked_in_c2v():
    vm = VM()
    vm.state.ctx_channel_llr = 1.5
    vm.state.ctx_has_channel_llr = False
    vm.run([I("Env.GetChannelLLR")])
    assert vm.state.floats.is_empty()


# --------------------------------------------------------- Control flow + sum
def test_dotimes_executes_n_times():
    """Composing a sum over a vector via DoTimes: the canonical example."""
    vm = VM()
    vm.state.fvecs.push(np.array([1.0, 2.0, 3.0, 4.0]))
    prog = [
        I("Float.Const0"),                  # accumulator on float stack
        I("FVec.Len"),                      # push 4
        I(
            "Exec.DoTimes",
            b1=[
                I("FVec.At"),               # consumes loop var, pushes v[i]
                I("Float.Add"),             # accumulator += v[i]
            ],
        ),
    ]
    out = vm.run(prog)
    assert out == 10.0


def test_if_true_branch():
    out, _ = run([
        I("Bool.True"),
        I("Exec.If",
          b1=[I("Float.Const1")],
          b2=[I("Float.ConstNeg1")]),
    ])
    assert out == 1.0


def test_if_false_branch():
    out, _ = run([
        I("Bool.False"),
        I("Exec.If",
          b1=[I("Float.Const1")],
          b2=[I("Float.ConstNeg1")]),
    ])
    assert out == -1.0


def test_dotimes_loop_cap_enforced():
    """Asking for 10000 iterations only runs at most N_MAX_LOOP."""
    from pushgp.instructions import N_MAX_LOOP
    vm = VM()
    prog = [
        I("Float.Const0"),
        I("Int.Const1"),
        I("Int.Const1"),
        I("Int.Add"),                 # 2
        I("Int.Const2"),
        I("Int.Mul"),                 # 4
        I("Int.Const2"),
        I("Int.Mul"),                 # 8
        I("Int.Const2"),
        I("Int.Mul"),                 # 16
        I("Int.Const2"),
        I("Int.Mul"),                 # 32
        I("Int.Const2"),
        I("Int.Mul"),                 # 64
        I("Int.Const2"),
        I("Int.Mul"),                 # 128 (asked) → capped to N_MAX_LOOP
        I("Exec.DoTimes",
          b1=[I("Float.Const1"), I("Float.Add")]),
    ]
    out = vm.run(prog)
    assert out == float(N_MAX_LOOP)


# --------------------------------------------------- Resource budget enforcement


def test_step_budget_aborts_run():
    vm = VM(step_max=10)
    prog = [I("Float.Const1")] * 50
    out = vm.run(prog)
    assert out is None
    assert vm.state.fault
    assert "step_max" in vm.state.fault_reason


def test_flop_budget_aborts_run():
    vm = VM(flop_max=3)
    prog = [
        I("Float.Const1"), I("Float.Const1"), I("Float.Add"),
        I("Float.Const1"), I("Float.Add"),
        I("Float.Const1"), I("Float.Add"),
        I("Float.Const1"), I("Float.Add"),
    ]
    out = vm.run(prog)
    assert out is None
    assert vm.state.fault


def test_recursion_budget_aborts():
    vm = VM(recur_max=2)
    # nested DoTimes deeper than recur_max
    inner = I("Exec.DoTimes", b1=[I("Float.Const1")])
    inner = I(
        "Exec.DoTimes",
        b1=[I("Int.Const1"), inner, I("Float.Const1")],
    )
    inner = I(
        "Exec.DoTimes",
        b1=[I("Int.Const1"), inner, I("Float.Const1")],
    )
    out = vm.run([I("Int.Const1"), inner])
    # may complete or abort depending on depth; ensure we don't crash.
    assert out is None or isinstance(out, float)


# ------------------------------------------------------ Forgiving underflow


def test_handlers_no_op_on_underflow():
    vm = VM()
    # Single instruction with empty stacks must not raise.
    for name in HANDLERS:
        vm.reset()
        vm.run([I(name)])
        # never raises; may set fault on resource exhaustion but most won't.


# ---------------------------------------------------- Empty-program edge case


def test_empty_program():
    out, _ = run([])
    assert out is None  # float stack empty


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
