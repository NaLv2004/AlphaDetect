"""All atomic Push-GP instruction handlers, plus the registry.

Design rules (locked in by user during the v2 design discussion):

* **Atomic only** — every handler implements one minimum-grain operation.
  No reductions or aggregations are baked into vector / loop operations.
  Sums, products, mins-over-vector, sign-products etc. must be composed
  by the evolved program out of `Float.*` / `Bool.*` primitives plus
  `Exec.DoTimes` over `FVec.At`.

* **Forgiving semantics** — every operation tolerates stack underflow,
  out-of-range indices, NaN/Inf operands, etc., by treating the
  offending case as a no-op and continuing.  This is critical for
  randomly generated Push programs to ever finish executing.

* **No Code stack** — control instructions (`Exec.If`, `Exec.When`,
  `Exec.DoTimes`, `Exec.DoRange`, `Exec.While`) carry their nested
  blocks as syntactic children of the `Instruction` object.  Per the
  user's explicit request we do NOT introduce first-class code values
  / S-K-Y combinators.

Each handler has signature `(vm, ins) -> None` where `vm` is the
`VM` instance (gives access to `vm.state` and `vm.execute_block`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Dict, List

import numpy as np

from .program import Instruction
from .types import FLOAT_ABS_MAX, NAN_INF_REPLACEMENT, Type

if TYPE_CHECKING:  # pragma: no cover
    from .vm import VM


# ============================================================ numerical helpers


def _safe_float(x: float) -> float:
    if not math.isfinite(x):
        return NAN_INF_REPLACEMENT
    if x > FLOAT_ABS_MAX:
        return FLOAT_ABS_MAX
    if x < -FLOAT_ABS_MAX:
        return -FLOAT_ABS_MAX
    return x


def _safe_array(a: np.ndarray) -> np.ndarray:
    """Sanitize an ndarray: replace NaN/Inf, clamp magnitude."""
    a = np.where(np.isfinite(a), a, NAN_INF_REPLACEMENT).astype(np.float64, copy=False)
    np.clip(a, -FLOAT_ABS_MAX, FLOAT_ABS_MAX, out=a)
    return a


# ===================================================================== Registry

# A handler may also charge extra FLOPs via `vm.charge_flops(n)`.
HANDLERS: Dict[str, Callable[["VM", Instruction], None]] = {}


def _reg(name: str):
    def deco(fn):
        if name in HANDLERS:
            raise ValueError(f"duplicate instruction registration: {name}")
        HANDLERS[name] = fn
        fn.__pushgp_name__ = name  # type: ignore[attr-defined]
        return fn

    return deco


# =========================================================== Float arithmetic
def _float_binop(vm: "VM", op: Callable[[float, float], float]) -> None:
    s = vm.state.floats
    if s.depth() < 2:
        return
    b = s.pop()
    a = s.pop()
    try:
        r = op(float(a), float(b))  # type: ignore[arg-type]
    except (ZeroDivisionError, OverflowError, ValueError):
        r = NAN_INF_REPLACEMENT
    s.push(_safe_float(r))
    vm.charge_flops(1)


def _float_unop(vm: "VM", op: Callable[[float], float]) -> None:
    s = vm.state.floats
    if s.depth() < 1:
        return
    a = s.pop()
    try:
        r = op(float(a))  # type: ignore[arg-type]
    except (ValueError, OverflowError):
        r = NAN_INF_REPLACEMENT
    s.push(_safe_float(r))
    vm.charge_flops(1)


@_reg("Float.Add")
def _f_add(vm, ins): _float_binop(vm, lambda a, b: a + b)
@_reg("Float.Sub")
def _f_sub(vm, ins): _float_binop(vm, lambda a, b: a - b)
@_reg("Float.Mul")
def _f_mul(vm, ins): _float_binop(vm, lambda a, b: a * b)
@_reg("Float.Div")
def _f_div(vm, ins): _float_binop(vm, lambda a, b: a / b if b != 0.0 else NAN_INF_REPLACEMENT)
@_reg("Float.Mod")
def _f_mod(vm, ins): _float_binop(vm, lambda a, b: math.fmod(a, b) if b != 0.0 else NAN_INF_REPLACEMENT)
@_reg("Float.Min")
def _f_min(vm, ins): _float_binop(vm, min)
@_reg("Float.Max")
def _f_max(vm, ins): _float_binop(vm, max)

@_reg("Float.Abs")
def _f_abs(vm, ins): _float_unop(vm, abs)
@_reg("Float.Neg")
def _f_neg(vm, ins): _float_unop(vm, lambda a: -a)
@_reg("Float.Inv")
def _f_inv(vm, ins): _float_unop(vm, lambda a: 1.0 / a if a != 0.0 else NAN_INF_REPLACEMENT)
@_reg("Float.Sqrt")
def _f_sqrt(vm, ins): _float_unop(vm, lambda a: math.sqrt(a) if a >= 0.0 else NAN_INF_REPLACEMENT)
@_reg("Float.Square")
def _f_square(vm, ins): _float_unop(vm, lambda a: a * a)
@_reg("Float.Exp")
def _f_exp(vm, ins): _float_unop(vm, lambda a: math.exp(a) if a < 80.0 else NAN_INF_REPLACEMENT)
@_reg("Float.Log")
def _f_log(vm, ins): _float_unop(vm, lambda a: math.log(a) if a > 0.0 else NAN_INF_REPLACEMENT)
@_reg("Float.Tanh")
def _f_tanh(vm, ins): _float_unop(vm, math.tanh)
@_reg("Float.Atanh")
def _f_atanh(vm, ins): _float_unop(vm, lambda a: math.atanh(a) if -1.0 < a < 1.0 else NAN_INF_REPLACEMENT)
@_reg("Float.Sign")
def _f_sign(vm, ins): _float_unop(vm, lambda a: 1.0 if a > 0 else (-1.0 if a < 0 else 0.0))
@_reg("Float.Floor")
def _f_floor(vm, ins): _float_unop(vm, math.floor)
@_reg("Float.Ceil")
def _f_ceil(vm, ins): _float_unop(vm, math.ceil)


# =========================================================== Float constants
def _push_float_const(vm: "VM", value: float) -> None:
    vm.state.floats.push(_safe_float(value))


@_reg("Float.Const0")
def _fc_0(vm, ins): _push_float_const(vm, 0.0)
@_reg("Float.Const1")
def _fc_1(vm, ins): _push_float_const(vm, 1.0)
@_reg("Float.ConstNeg1")
def _fc_n1(vm, ins): _push_float_const(vm, -1.0)
@_reg("Float.ConstHalf")
def _fc_h(vm, ins): _push_float_const(vm, 0.5)
@_reg("Float.Const2")
def _fc_2(vm, ins): _push_float_const(vm, 2.0)
@_reg("Float.Const0_1")
def _fc_01(vm, ins): _push_float_const(vm, 0.1)
@_reg("Float.ConstPi")
def _fc_pi(vm, ins): _push_float_const(vm, math.pi)
@_reg("Float.Const1e-6")
def _fc_eps(vm, ins): _push_float_const(vm, 1e-6)


# =========================================== Float evolved constants (8 of them)
def _make_evo(idx: int):
    def _h(vm, ins):
        ec = vm.state.ctx_evo_constants
        if 0 <= idx < ec.shape[0]:
            _push_float_const(vm, float(ec[idx]))

    _h.__name__ = f"_evo_{idx}"
    return _h


for _i in range(8):
    _reg(f"Float.EvoConst{_i}")(_make_evo(_i))


# ============================================================ Float stack ops
def _stack_pop(vm, t):  vm.state.stack_for(t).pop()
def _stack_dup(vm, t):  vm.state.stack_for(t).dup()
def _stack_swap(vm, t): vm.state.stack_for(t).swap()
def _stack_rot(vm, t):  vm.state.stack_for(t).rot()


def _stack_yank(vm, t):
    n_stk = vm.state.ints
    if n_stk.is_empty():
        return
    n = n_stk.pop()
    if n is None:
        return
    vm.state.stack_for(t).yank(int(n))


def _stack_shove(vm, t):
    n_stk = vm.state.ints
    if n_stk.is_empty():
        return
    n = n_stk.pop()
    if n is None:
        return
    vm.state.stack_for(t).shove(int(n))


@_reg("Float.Pop")
def _fp_pop(vm, ins): _stack_pop(vm, Type.FLOAT)
@_reg("Float.Dup")
def _fp_dup(vm, ins): _stack_dup(vm, Type.FLOAT)
@_reg("Float.Swap")
def _fp_swap(vm, ins): _stack_swap(vm, Type.FLOAT)
@_reg("Float.Rot")
def _fp_rot(vm, ins): _stack_rot(vm, Type.FLOAT)
@_reg("Float.Yank")
def _fp_yank(vm, ins): _stack_yank(vm, Type.FLOAT)
@_reg("Float.Shove")
def _fp_shove(vm, ins): _stack_shove(vm, Type.FLOAT)


# ============================================================== Int arithmetic
def _int_binop(vm, op):
    s = vm.state.ints
    if s.depth() < 2:
        return
    b = s.pop()
    a = s.pop()
    try:
        r = int(op(int(a), int(b)))  # type: ignore[arg-type]
    except (ZeroDivisionError, OverflowError, ValueError):
        r = 0
    # Bound the integer to avoid astronomical loop counters elsewhere.
    if r > 10**9:
        r = 10**9
    elif r < -(10**9):
        r = -(10**9)
    s.push(r)
    vm.charge_flops(1)


def _int_unop(vm, op):
    s = vm.state.ints
    if s.depth() < 1:
        return
    a = s.pop()
    try:
        r = int(op(int(a)))  # type: ignore[arg-type]
    except (OverflowError, ValueError):
        r = 0
    s.push(r)
    vm.charge_flops(1)


@_reg("Int.Add")
def _i_add(vm, ins): _int_binop(vm, lambda a, b: a + b)
@_reg("Int.Sub")
def _i_sub(vm, ins): _int_binop(vm, lambda a, b: a - b)
@_reg("Int.Mul")
def _i_mul(vm, ins): _int_binop(vm, lambda a, b: a * b)
@_reg("Int.Div")
def _i_div(vm, ins): _int_binop(vm, lambda a, b: a // b if b != 0 else 0)
@_reg("Int.Mod")
def _i_mod(vm, ins): _int_binop(vm, lambda a, b: a % b if b != 0 else 0)
@_reg("Int.Min")
def _i_min(vm, ins): _int_binop(vm, min)
@_reg("Int.Max")
def _i_max(vm, ins): _int_binop(vm, max)
@_reg("Int.Inc")
def _i_inc(vm, ins): _int_unop(vm, lambda a: a + 1)
@_reg("Int.Dec")
def _i_dec(vm, ins): _int_unop(vm, lambda a: a - 1)
@_reg("Int.Neg")
def _i_neg(vm, ins): _int_unop(vm, lambda a: -a)


@_reg("Int.Const0")
def _ic_0(vm, ins): vm.state.ints.push(0)
@_reg("Int.Const1")
def _ic_1(vm, ins): vm.state.ints.push(1)
@_reg("Int.Const2")
def _ic_2(vm, ins): vm.state.ints.push(2)
@_reg("Int.ConstNeg1")
def _ic_n1(vm, ins): vm.state.ints.push(-1)


@_reg("Int.Pop")
def _ip_pop(vm, ins): _stack_pop(vm, Type.INT)
@_reg("Int.Dup")
def _ip_dup(vm, ins): _stack_dup(vm, Type.INT)
@_reg("Int.Swap")
def _ip_swap(vm, ins): _stack_swap(vm, Type.INT)
@_reg("Int.Rot")
def _ip_rot(vm, ins): _stack_rot(vm, Type.INT)


# ===================================================================== Bool ops
def _bool_binop(vm, op):
    s = vm.state.bools
    if s.depth() < 2:
        return
    b = s.pop()
    a = s.pop()
    s.push(bool(op(bool(a), bool(b))))


@_reg("Bool.And")
def _b_and(vm, ins): _bool_binop(vm, lambda a, b: a and b)
@_reg("Bool.Or")
def _b_or(vm, ins): _bool_binop(vm, lambda a, b: a or b)
@_reg("Bool.Xor")
def _b_xor(vm, ins): _bool_binop(vm, lambda a, b: a ^ b)


@_reg("Bool.Not")
def _b_not(vm, ins):
    s = vm.state.bools
    if s.depth() < 1:
        return
    s.push(not bool(s.pop()))


@_reg("Bool.True")
def _b_t(vm, ins): vm.state.bools.push(True)
@_reg("Bool.False")
def _b_f(vm, ins): vm.state.bools.push(False)
@_reg("Bool.Pop")
def _b_pop(vm, ins): _stack_pop(vm, Type.BOOL)
@_reg("Bool.Dup")
def _b_dup(vm, ins): _stack_dup(vm, Type.BOOL)
@_reg("Bool.Swap")
def _b_swap(vm, ins): _stack_swap(vm, Type.BOOL)


# ================================================================== Comparisons
def _cmp_float(vm, op):
    s = vm.state.floats
    if s.depth() < 2:
        return
    b = s.pop()
    a = s.pop()
    vm.state.bools.push(bool(op(float(a), float(b))))  # type: ignore[arg-type]


def _cmp_int(vm, op):
    s = vm.state.ints
    if s.depth() < 2:
        return
    b = s.pop()
    a = s.pop()
    vm.state.bools.push(bool(op(int(a), int(b))))  # type: ignore[arg-type]


@_reg("Float.LT")
def _flt(vm, ins): _cmp_float(vm, lambda a, b: a < b)
@_reg("Float.GT")
def _fgt(vm, ins): _cmp_float(vm, lambda a, b: a > b)
@_reg("Float.EQ")
def _feq(vm, ins): _cmp_float(vm, lambda a, b: abs(a - b) < 1e-12)
@_reg("Int.LT")
def _ilt(vm, ins): _cmp_int(vm, lambda a, b: a < b)
@_reg("Int.GT")
def _igt(vm, ins): _cmp_int(vm, lambda a, b: a > b)
@_reg("Int.EQ")
def _ieq(vm, ins): _cmp_int(vm, lambda a, b: a == b)


# ==================================================================== Conversions
@_reg("Float.FromInt")
def _conv_fi(vm, ins):
    s = vm.state.ints
    if s.depth() < 1:
        return
    v = s.pop()
    vm.state.floats.push(_safe_float(float(v)))  # type: ignore[arg-type]


@_reg("Int.FromFloat")
def _conv_if(vm, ins):
    s = vm.state.floats
    if s.depth() < 1:
        return
    v = s.pop()
    try:
        vm.state.ints.push(int(v))  # type: ignore[arg-type]
    except (ValueError, OverflowError):
        vm.state.ints.push(0)


@_reg("Int.FromBool")
def _conv_ib(vm, ins):
    s = vm.state.bools
    if s.depth() < 1:
        return
    v = s.pop()
    vm.state.ints.push(1 if v else 0)


@_reg("Bool.FromFloat")
def _conv_bf(vm, ins):
    s = vm.state.floats
    if s.depth() < 1:
        return
    v = s.pop()
    vm.state.bools.push(bool(float(v) > 0.0))  # type: ignore[arg-type]


@_reg("Bool.FromInt")
def _conv_bi(vm, ins):
    s = vm.state.ints
    if s.depth() < 1:
        return
    v = s.pop()
    vm.state.bools.push(bool(int(v) != 0))  # type: ignore[arg-type]


# ====================================================== Vector size guard
MAX_VEC_LEN = 1024


def _truncate_vec(a: np.ndarray) -> np.ndarray:
    if a.size > MAX_VEC_LEN:
        a = a[:MAX_VEC_LEN].copy()
    return a


# ============================================================ FloatVec ops
@_reg("FVec.Len")
def _fv_len(vm, ins):
    v = vm.state.fvecs.peek()
    if v is None:
        return
    vm.state.ints.push(int(v.size))


@_reg("FVec.At")
def _fv_at(vm, ins):
    v = vm.state.fvecs.peek()
    if v is None or v.size == 0:
        return
    if vm.state.ints.depth() < 1:
        return
    idx = vm.state.ints.pop()
    if idx is None:
        return
    i = int(idx) % v.size  # wrap-around for forgiving indexing
    vm.state.floats.push(_safe_float(float(v[i])))


@_reg("FVec.Set")
def _fv_set(vm, ins):
    v = vm.state.fvecs.peek()
    if v is None or v.size == 0:
        return
    if vm.state.ints.depth() < 1 or vm.state.floats.depth() < 1:
        return
    idx = vm.state.ints.pop()
    val = vm.state.floats.pop()
    new_v = v.copy()
    i = int(idx) % v.size  # type: ignore[arg-type]
    new_v[i] = _safe_float(float(val))  # type: ignore[arg-type]
    # replace top
    vm.state.fvecs.pop()
    vm.state.fvecs.push(new_v)


@_reg("FVec.Push")
def _fv_push(vm, ins):
    v = vm.state.fvecs.peek()
    if v is None or vm.state.floats.depth() < 1:
        return
    val = vm.state.floats.pop()
    new_v = np.append(v, _safe_float(float(val)))  # type: ignore[arg-type]
    new_v = _truncate_vec(new_v)
    vm.state.fvecs.pop()
    vm.state.fvecs.push(new_v)


@_reg("FVec.PopBack")
def _fv_popback(vm, ins):
    v = vm.state.fvecs.peek()
    if v is None or v.size == 0:
        return
    val = float(v[-1])
    new_v = v[:-1].copy()
    vm.state.fvecs.pop()
    vm.state.fvecs.push(new_v)
    vm.state.floats.push(_safe_float(val))


@_reg("FVec.New")
def _fv_new(vm, ins):
    if vm.state.ints.depth() < 1:
        return
    n = vm.state.ints.pop()
    nn = max(0, min(int(n), MAX_VEC_LEN))  # type: ignore[arg-type]
    vm.state.fvecs.push(np.zeros(nn, dtype=np.float64))


@_reg("FVec.FromFloat")
def _fv_fromf(vm, ins):
    if vm.state.floats.depth() < 1:
        return
    val = vm.state.floats.pop()
    vm.state.fvecs.push(np.array([_safe_float(float(val))]))  # type: ignore[arg-type]


@_reg("FVec.Concat")
def _fv_concat(vm, ins):
    if vm.state.fvecs.depth() < 2:
        return
    b = vm.state.fvecs.pop()
    a = vm.state.fvecs.pop()
    vm.state.fvecs.push(_truncate_vec(np.concatenate([a, b])))


@_reg("FVec.Slice")
def _fv_slice(vm, ins):
    v = vm.state.fvecs.peek()
    if v is None or vm.state.ints.depth() < 2:
        return
    end = vm.state.ints.pop()
    start = vm.state.ints.pop()
    s = max(0, min(int(start), v.size))  # type: ignore[arg-type]
    e = max(s, min(int(end), v.size))  # type: ignore[arg-type]
    vm.state.fvecs.pop()
    vm.state.fvecs.push(v[s:e].copy())


@_reg("FVec.Reverse")
def _fv_rev(vm, ins):
    v = vm.state.fvecs.peek()
    if v is None:
        return
    vm.state.fvecs.pop()
    vm.state.fvecs.push(v[::-1].copy())


@_reg("FVec.Roll")
def _fv_roll(vm, ins):
    v = vm.state.fvecs.peek()
    if v is None or vm.state.ints.depth() < 1:
        return
    k = vm.state.ints.pop()
    if v.size == 0:
        return
    vm.state.fvecs.pop()
    vm.state.fvecs.push(np.roll(v, int(k) % v.size))  # type: ignore[arg-type]


@_reg("FVec.Pop")
def _fv_pop(vm, ins): _stack_pop(vm, Type.FLOATVEC)
@_reg("FVec.Dup")
def _fv_dup(vm, ins): _stack_dup(vm, Type.FLOATVEC)
@_reg("FVec.Swap")
def _fv_swap(vm, ins): _stack_swap(vm, Type.FLOATVEC)
@_reg("FVec.Rot")
def _fv_rot(vm, ins): _stack_rot(vm, Type.FLOATVEC)


# =================================================================== BoolVec
@_reg("BVec.Len")
def _bv_len(vm, ins):
    v = vm.state.bvecs.peek()
    if v is None:
        return
    vm.state.ints.push(int(v.size))


@_reg("BVec.At")
def _bv_at(vm, ins):
    v = vm.state.bvecs.peek()
    if v is None or v.size == 0 or vm.state.ints.depth() < 1:
        return
    idx = vm.state.ints.pop()
    i = int(idx) % v.size  # type: ignore[arg-type]
    vm.state.bools.push(bool(v[i]))


@_reg("BVec.Set")
def _bv_set(vm, ins):
    v = vm.state.bvecs.peek()
    if v is None or v.size == 0:
        return
    if vm.state.ints.depth() < 1 or vm.state.bools.depth() < 1:
        return
    idx = vm.state.ints.pop()
    val = vm.state.bools.pop()
    new_v = v.copy()
    new_v[int(idx) % v.size] = bool(val)  # type: ignore[arg-type]
    vm.state.bvecs.pop()
    vm.state.bvecs.push(new_v)


@_reg("BVec.Push")
def _bv_push(vm, ins):
    v = vm.state.bvecs.peek()
    if v is None or vm.state.bools.depth() < 1:
        return
    val = vm.state.bools.pop()
    new_v = np.append(v, bool(val))  # type: ignore[arg-type]
    new_v = new_v[:MAX_VEC_LEN]
    vm.state.bvecs.pop()
    vm.state.bvecs.push(new_v.astype(bool))


@_reg("BVec.PopBack")
def _bv_popback(vm, ins):
    v = vm.state.bvecs.peek()
    if v is None or v.size == 0:
        return
    val = bool(v[-1])
    new_v = v[:-1].copy()
    vm.state.bvecs.pop()
    vm.state.bvecs.push(new_v)
    vm.state.bools.push(val)


@_reg("BVec.New")
def _bv_new(vm, ins):
    if vm.state.ints.depth() < 1:
        return
    n = vm.state.ints.pop()
    nn = max(0, min(int(n), MAX_VEC_LEN))  # type: ignore[arg-type]
    vm.state.bvecs.push(np.zeros(nn, dtype=bool))


@_reg("BVec.Concat")
def _bv_concat(vm, ins):
    if vm.state.bvecs.depth() < 2:
        return
    b = vm.state.bvecs.pop()
    a = vm.state.bvecs.pop()
    vm.state.bvecs.push(np.concatenate([a, b])[:MAX_VEC_LEN])


@_reg("BVec.Slice")
def _bv_slice(vm, ins):
    v = vm.state.bvecs.peek()
    if v is None or vm.state.ints.depth() < 2:
        return
    end = vm.state.ints.pop()
    start = vm.state.ints.pop()
    s = max(0, min(int(start), v.size))  # type: ignore[arg-type]
    e = max(s, min(int(end), v.size))  # type: ignore[arg-type]
    vm.state.bvecs.pop()
    vm.state.bvecs.push(v[s:e].copy())


@_reg("BVec.Reverse")
def _bv_rev(vm, ins):
    v = vm.state.bvecs.peek()
    if v is None:
        return
    vm.state.bvecs.pop()
    vm.state.bvecs.push(v[::-1].copy())


@_reg("BVec.Pop")
def _bv_pop(vm, ins): _stack_pop(vm, Type.BOOLVEC)
@_reg("BVec.Dup")
def _bv_dup(vm, ins): _stack_dup(vm, Type.BOOLVEC)
@_reg("BVec.Swap")
def _bv_swap(vm, ins): _stack_swap(vm, Type.BOOLVEC)


# =================================================================== IntVec
@_reg("IVec.Len")
def _iv_len(vm, ins):
    v = vm.state.ivecs.peek()
    if v is None:
        return
    vm.state.ints.push(int(v.size))


@_reg("IVec.At")
def _iv_at(vm, ins):
    v = vm.state.ivecs.peek()
    if v is None or v.size == 0 or vm.state.ints.depth() < 1:
        return
    idx = vm.state.ints.pop()
    i = int(idx) % v.size  # type: ignore[arg-type]
    vm.state.ints.push(int(v[i]))


@_reg("IVec.Set")
def _iv_set(vm, ins):
    v = vm.state.ivecs.peek()
    if v is None or v.size == 0:
        return
    if vm.state.ints.depth() < 2:
        return
    val = vm.state.ints.pop()
    idx = vm.state.ints.pop()
    new_v = v.copy()
    new_v[int(idx) % v.size] = int(val)  # type: ignore[arg-type]
    vm.state.ivecs.pop()
    vm.state.ivecs.push(new_v)


@_reg("IVec.Push")
def _iv_push(vm, ins):
    v = vm.state.ivecs.peek()
    if v is None or vm.state.ints.depth() < 1:
        return
    val = vm.state.ints.pop()
    new_v = np.append(v, int(val))[:MAX_VEC_LEN]  # type: ignore[arg-type]
    vm.state.ivecs.pop()
    vm.state.ivecs.push(new_v.astype(np.int64))


@_reg("IVec.PopBack")
def _iv_popback(vm, ins):
    v = vm.state.ivecs.peek()
    if v is None or v.size == 0:
        return
    val = int(v[-1])
    new_v = v[:-1].copy()
    vm.state.ivecs.pop()
    vm.state.ivecs.push(new_v)
    vm.state.ints.push(val)


@_reg("IVec.New")
def _iv_new(vm, ins):
    if vm.state.ints.depth() < 1:
        return
    n = vm.state.ints.pop()
    nn = max(0, min(int(n), MAX_VEC_LEN))  # type: ignore[arg-type]
    vm.state.ivecs.push(np.zeros(nn, dtype=np.int64))


@_reg("IVec.Concat")
def _iv_concat(vm, ins):
    if vm.state.ivecs.depth() < 2:
        return
    b = vm.state.ivecs.pop()
    a = vm.state.ivecs.pop()
    vm.state.ivecs.push(np.concatenate([a, b])[:MAX_VEC_LEN])


@_reg("IVec.Slice")
def _iv_slice(vm, ins):
    v = vm.state.ivecs.peek()
    if v is None or vm.state.ints.depth() < 2:
        return
    end = vm.state.ints.pop()
    start = vm.state.ints.pop()
    s = max(0, min(int(start), v.size))  # type: ignore[arg-type]
    e = max(s, min(int(end), v.size))  # type: ignore[arg-type]
    vm.state.ivecs.pop()
    vm.state.ivecs.push(v[s:e].copy())


@_reg("IVec.Reverse")
def _iv_rev(vm, ins):
    v = vm.state.ivecs.peek()
    if v is None:
        return
    vm.state.ivecs.pop()
    vm.state.ivecs.push(v[::-1].copy())


@_reg("IVec.Pop")
def _iv_pop(vm, ins): _stack_pop(vm, Type.INTVEC)
@_reg("IVec.Dup")
def _iv_dup(vm, ins): _stack_dup(vm, Type.INTVEC)
@_reg("IVec.Swap")
def _iv_swap(vm, ins): _stack_swap(vm, Type.INTVEC)


# =================================================================== Memory (16 cells)
@_reg("Mem.Read")
def _m_read(vm, ins):
    if vm.state.ints.depth() < 1:
        return
    idx = vm.state.ints.pop()
    n = vm.state.memory.size
    if n == 0:
        return
    i = int(idx) % n  # type: ignore[arg-type]
    vm.state.floats.push(_safe_float(float(vm.state.memory[i])))


@_reg("Mem.Write")
def _m_write(vm, ins):
    if vm.state.ints.depth() < 1 or vm.state.floats.depth() < 1:
        return
    idx = vm.state.ints.pop()
    val = vm.state.floats.pop()
    n = vm.state.memory.size
    if n == 0:
        return
    vm.state.memory[int(idx) % n] = _safe_float(float(val))  # type: ignore[arg-type]


@_reg("Mem.ReadVec")
def _m_readv(vm, ins):
    if vm.state.ints.depth() < 2:
        return
    end = vm.state.ints.pop()
    start = vm.state.ints.pop()
    n = vm.state.memory.size
    s = max(0, min(int(start), n))  # type: ignore[arg-type]
    e = max(s, min(int(end), n))  # type: ignore[arg-type]
    vm.state.fvecs.push(vm.state.memory[s:e].copy())


@_reg("Mem.WriteVec")
def _m_writev(vm, ins):
    if vm.state.ints.depth() < 1 or vm.state.fvecs.depth() < 1:
        return
    start = vm.state.ints.pop()
    v = vm.state.fvecs.pop()
    n = vm.state.memory.size
    s = max(0, min(int(start), n))  # type: ignore[arg-type]
    e = min(n, s + v.size)
    vm.state.memory[s:e] = _safe_array(v[: e - s])


# =================================================================== Env reads
@_reg("Env.GetChannelLLR")
def _env_llr(vm, ins):
    if vm.state.ctx_has_channel_llr:
        vm.state.floats.push(_safe_float(float(vm.state.ctx_channel_llr)))


@_reg("Env.GetIncomingVec")
def _env_inc(vm, ins):
    vm.state.fvecs.push(_safe_array(vm.state.ctx_incoming.copy()))


@_reg("Env.GetNoiseVar")
def _env_nv(vm, ins):
    vm.state.floats.push(_safe_float(float(vm.state.ctx_noise_var)))


@_reg("Env.GetIter")
def _env_it(vm, ins):
    vm.state.ints.push(int(vm.state.ctx_iter))


@_reg("Env.GetMaxIter")
def _env_mit(vm, ins):
    vm.state.ints.push(int(vm.state.ctx_max_iter))


@_reg("Env.GetDeg")
def _env_dg(vm, ins):
    vm.state.ints.push(int(vm.state.ctx_deg))


@_reg("Env.GetEdgeIndex")
def _env_eix(vm, ins):
    vm.state.ints.push(int(vm.state.ctx_edge_index))


@_reg("Env.GetCodeRate")
def _env_cr(vm, ins):
    vm.state.floats.push(_safe_float(float(vm.state.ctx_code_rate)))


# =================================================================== Control flow
N_MAX_LOOP = 64
N_MAX_WHILE = 32


@_reg("Exec.If")
def _ex_if(vm, ins):
    if vm.state.bools.depth() < 1:
        return
    cond = vm.state.bools.pop()
    block = ins.code_block if cond else ins.code_block2
    if block:
        vm.execute_block(block)


@_reg("Exec.When")
def _ex_when(vm, ins):
    if vm.state.bools.depth() < 1:
        return
    cond = vm.state.bools.pop()
    if cond and ins.code_block:
        vm.execute_block(ins.code_block)


@_reg("Exec.DoTimes")
def _ex_dotimes(vm, ins):
    if vm.state.ints.depth() < 1 or not ins.code_block:
        return
    n = vm.state.ints.pop()
    if n is None:
        return
    n = max(0, min(int(n), N_MAX_LOOP))
    for i in range(n):
        if vm.aborted():
            return
        vm.state.ints.push(i)
        vm.execute_block(ins.code_block)


@_reg("Exec.DoRange")
def _ex_dorange(vm, ins):
    if vm.state.ints.depth() < 2 or not ins.code_block:
        return
    end = vm.state.ints.pop()
    start = vm.state.ints.pop()
    s = int(start)  # type: ignore[arg-type]
    e = int(end)    # type: ignore[arg-type]
    if e <= s:
        return
    span = min(N_MAX_LOOP, e - s)
    for i in range(s, s + span):
        if vm.aborted():
            return
        vm.state.ints.push(i)
        vm.execute_block(ins.code_block)


@_reg("Exec.While")
def _ex_while(vm, ins):
    if not ins.code_block:
        return
    for _ in range(N_MAX_WHILE):
        if vm.aborted():
            return
        if vm.state.bools.depth() < 1:
            return
        cond = vm.state.bools.pop()
        if not cond:
            return
        vm.execute_block(ins.code_block)


# ================================================================= Convenience

def all_instruction_names() -> List[str]:
    return list(HANDLERS.keys())


def is_control(name: str) -> bool:
    return name in {"Exec.If", "Exec.When", "Exec.DoTimes", "Exec.DoRange", "Exec.While"}


def has_two_blocks(name: str) -> bool:
    return name == "Exec.If"


__all__ = [
    "HANDLERS",
    "all_instruction_names",
    "is_control",
    "has_two_blocks",
    "MAX_VEC_LEN",
    "N_MAX_LOOP",
    "N_MAX_WHILE",
]
