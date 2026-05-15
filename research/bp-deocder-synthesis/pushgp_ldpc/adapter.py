"""Bridge between an evolved Push-GP Genome and the LDPC decoder.

`make_callables(genome, ...)` returns a `(v2c_fn, c2v_fn)` pair whose
signatures match the contract of `ldpc_5g.decode_bp` and the
stack-seeding contract of `pushgp.validators`.

A hand-coded OMS seed genome is provided by `oms_seed_genome()` and
stored on disk under `pushgp_ldpc/seeds/oms.json`.  Plugging that seed
into `make_callables(...)` and decoding with `decode_bp(...)` produces
identical posteriors to `ldpc_5g.default_v2c_oms / default_c2v_oms`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import numpy as np

from pushgp.genome import Genome, N_EVO_CONSTS
from pushgp.program import Instruction
from pushgp.vm import VM


SeedFn = Callable[[float, np.ndarray, int, int, dict], float]
# c2v has no L_v argument
C2VFn = Callable[[np.ndarray, int, int, dict], float]


# ============================================================ Adapter


def _seed_v2c(vm: VM, L_v: float, incoming: np.ndarray, deg: int,
              it: int, max_iter: int, evo_consts: np.ndarray) -> None:
    vm.reset()
    vm.state.ctx_channel_llr = float(L_v)
    vm.state.ctx_has_channel_llr = True
    vm.state.ctx_incoming = incoming.astype(np.float64, copy=True)
    vm.state.ctx_deg = int(deg)
    vm.state.ctx_iter = int(it)
    vm.state.ctx_max_iter = int(max_iter)
    vm.state.ctx_noise_var = 1.0
    vm.state.ctx_edge_index = 0
    vm.state.ctx_code_rate = 0.5
    vm.state.ctx_evo_constants = evo_consts.astype(np.float64, copy=True)

    vm.state.floats.push(float(L_v))
    vm.state.ints.push(0)
    vm.state.ints.push(int(deg))
    vm.state.ints.push(int(it))
    vm.state.ints.push(int(max_iter))
    vm.state.fvecs.push(vm.state.ctx_incoming.copy())


def _seed_c2v(vm: VM, incoming: np.ndarray, deg: int,
              it: int, max_iter: int, evo_consts: np.ndarray) -> None:
    vm.reset()
    vm.state.ctx_channel_llr = 0.0
    vm.state.ctx_has_channel_llr = False
    vm.state.ctx_incoming = incoming.astype(np.float64, copy=True)
    vm.state.ctx_deg = int(deg)
    vm.state.ctx_iter = int(it)
    vm.state.ctx_max_iter = int(max_iter)
    vm.state.ctx_noise_var = 1.0
    vm.state.ctx_edge_index = 0
    vm.state.ctx_code_rate = 0.5
    vm.state.ctx_evo_constants = evo_consts.astype(np.float64, copy=True)

    vm.state.ints.push(0)
    vm.state.ints.push(int(deg))
    vm.state.ints.push(int(it))
    vm.state.ints.push(int(max_iter))
    vm.state.fvecs.push(vm.state.ctx_incoming.copy())


def make_callables(
    genome: Genome,
    *,
    step_max: int = 2000,
    flop_max: int = 50_000,
    recur_max: int = 32,
) -> Tuple[SeedFn, C2VFn]:
    """Return (v2c_fn, c2v_fn) closures driven by `genome`'s programs."""
    evo = genome.evo_const_values()
    # Pre-bind a single VM per closure to avoid per-call construction cost.
    vm_v2c = VM(step_max=step_max, flop_max=flop_max, recur_max=recur_max)
    vm_c2v = VM(step_max=step_max, flop_max=flop_max, recur_max=recur_max)

    def v2c_fn(L_v: float, incoming: np.ndarray, deg: int, it: int, ctx: dict) -> float:
        _seed_v2c(vm_v2c, L_v, incoming, deg, it,
                  int(ctx.get("max_iter", 25)), evo)
        out = vm_v2c.run(genome.prog_v2c)
        return 0.0 if out is None else float(out)

    def c2v_fn(incoming: np.ndarray, deg: int, it: int, ctx: dict) -> float:
        _seed_c2v(vm_c2v, incoming, deg, it,
                  int(ctx.get("max_iter", 25)), evo)
        out = vm_c2v.run(genome.prog_c2v)
        return 0.0 if out is None else float(out)

    return v2c_fn, c2v_fn


# ============================================================ OMS seed


def _I(name: str, *, b1=None, b2=None) -> Instruction:
    return Instruction(name=name, code_block=b1, code_block2=b2)


def _oms_v2c_program() -> list:
    """V2C: L_v + sum(incoming).

    Stack layout on entry (per `_seed_v2c`):
        floats: [L_v]
        ints:   [v_idx, deg, iter, max_iter]
        fvecs:  [incoming]
    """
    return [
        # Discard the implicitly-present max_iter on int top so that
        # FVec.Len's value lands in the right position?  No — DoTimes pops
        # the int top regardless, and we *want* it to pop our len.  So we
        # push len AFTER the existing ints and DoTimes immediately consumes
        # it.  Below works: FVec.Len → ints=[v,deg,it,maxit,len]; DoTimes
        # pops len.
        _I("FVec.Len"),
        _I("Exec.DoTimes", b1=[
            _I("FVec.At"),     # pops i (loop counter), pushes v[i]
            _I("Float.Add"),   # accumulates onto L_v on float stack
        ]),
    ]


def _oms_c2v_program() -> list:
    """C2V: sign_product * max(min|incoming| - β, 0), with β = EvoConst0.

    Stack layout on entry (per `_seed_c2v`):
        floats: []
        ints:   [v_idx, deg, iter, max_iter]
        fvecs:  [incoming]

    The sentinel for the min-fold is EvoConst1 (set to 1e6 in the seed).
    """
    return [
        # ---- Sign-product as bool (True iff #negatives is odd) ----
        _I("Bool.False"),
        _I("FVec.Len"),
        _I("Exec.DoTimes", b1=[
            _I("FVec.At"),         # pushes v[i]
            _I("Float.Const0"),    # pushes 0
            _I("Float.LT"),        # pops 0 then v[i], pushes (v[i] < 0)
            _I("Bool.Xor"),        # XOR into accumulator
        ]),
        # ---- Min |incoming| using EvoConst1 sentinel (= 1e6) ----
        _I("Float.EvoConst1"),
        _I("FVec.Len"),
        _I("Exec.DoTimes", b1=[
            _I("FVec.At"),
            _I("Float.Abs"),
            _I("Float.Min"),
        ]),
        # ---- Subtract β = EvoConst0, then clamp to >= 0 ----
        _I("Float.EvoConst0"),
        _I("Float.Sub"),
        _I("Float.Const0"),
        _I("Float.Max"),
        # ---- Apply sign: if odd #negatives, negate ----
        _I("Exec.If", b1=[_I("Float.Neg")], b2=[]),
    ]


def oms_seed_genome(beta: float = 0.25, sentinel: float = 1e6) -> Genome:
    """Construct the hand-coded OMS Push genome.

    `log_constants[0] = log10(beta)`, `log_constants[1] = log10(sentinel)`,
    other constants set to 0 (= 1.0 multiplier — unused).
    """
    log_consts = np.zeros(N_EVO_CONSTS, dtype=np.float64)
    log_consts[0] = float(np.log10(beta))
    log_consts[1] = float(np.log10(sentinel))
    return Genome(
        prog_v2c=_oms_v2c_program(),
        prog_c2v=_oms_c2v_program(),
        log_constants=log_consts,
    )


# ============================================================ Disk seed


_SEED_DIR = Path(__file__).resolve().parent / "seeds"


def save_oms_seed(path: Path | None = None) -> Path:
    if path is None:
        _SEED_DIR.mkdir(parents=True, exist_ok=True)
        path = _SEED_DIR / "oms.json"
    g = oms_seed_genome()
    g.save(path)
    return path


def load_oms_seed(path: Path | None = None) -> Genome:
    if path is None:
        path = _SEED_DIR / "oms.json"
    return Genome.load(path)


__all__ = [
    "make_callables",
    "oms_seed_genome",
    "save_oms_seed",
    "load_oms_seed",
]
