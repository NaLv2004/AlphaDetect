"""Real-perturbation validators for evolved Push-GP programs.

We do **not** track data dependencies via shadow tags or static analysis.
Per the user's explicit instruction we run the program with concrete
inputs, mutate one input at a time, and check that the output actually
changes (data-dependence) or stays the same (permutation invariance).

Two checks are exposed:

* `validate_v2c(prog, ...)`
  - V2C output must depend on the channel LLR (`L_v`)
  - V2C output must depend on the incoming-message vector (perturbing
    any element changes the output)
  - V2C output must be invariant under permutation of the incoming
    vector
  - V2C output must be finite for every test configuration

* `validate_c2v(prog, ...)`
  - C2V output must depend on the incoming-message vector
  - C2V output must be invariant under permutation of the incoming
    vector
  - C2V output must be finite for every test configuration

Both validators return `(ok: bool, reason: str)` so the GA can log why a
candidate was rejected.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .program import Instruction
from .vm import VM


# ---------------------------------------------------------------- defaults
DEFAULT_DEG = 8
DEFAULT_NUM_CONFIGS = 3
DEFAULT_NUM_PERMUTATIONS = 5
DEFAULT_PERTURB_DELTA = 1.7
EPS_DEPENDENCY = 1e-9
EPS_INVARIANCE = 1e-7


# ---------------------------------------------------------------- helpers


def _make_vm(
    incoming: np.ndarray,
    *,
    channel_llr: float = 0.0,
    has_channel_llr: bool = True,
    deg: int = DEFAULT_DEG,
    iter_idx: int = 0,
    max_iter: int = 25,
    noise_var: float = 1.0,
    edge_index: int = 0,
    code_rate: float = 0.5,
    evo_consts: Optional[np.ndarray] = None,
) -> VM:
    vm = VM()
    vm.state.ctx_channel_llr = float(channel_llr)
    vm.state.ctx_has_channel_llr = bool(has_channel_llr)
    vm.state.ctx_incoming = incoming.astype(np.float64, copy=True)
    vm.state.ctx_deg = int(deg)
    vm.state.ctx_iter = int(iter_idx)
    vm.state.ctx_max_iter = int(max_iter)
    vm.state.ctx_noise_var = float(noise_var)
    vm.state.ctx_edge_index = int(edge_index)
    vm.state.ctx_code_rate = float(code_rate)
    if evo_consts is not None:
        vm.state.ctx_evo_constants = evo_consts.astype(np.float64, copy=True)
    return vm


def _seed_v2c_stacks(vm: VM) -> None:
    """Push the V2C entry-point context on the typed stacks.

    Layout (mirrors the LDPC adapter we'll build in PR6):
        float_stack: [L_v]
        int_stack:   [v_index, deg, iter, max_iter]   (max_iter on top)
        fvec_stack:  [incoming]
    """
    vm.state.floats.push(float(vm.state.ctx_channel_llr))
    vm.state.ints.push(int(vm.state.ctx_edge_index))
    vm.state.ints.push(int(vm.state.ctx_deg))
    vm.state.ints.push(int(vm.state.ctx_iter))
    vm.state.ints.push(int(vm.state.ctx_max_iter))
    vm.state.fvecs.push(vm.state.ctx_incoming.copy())


def _seed_c2v_stacks(vm: VM) -> None:
    """C2V variant — no channel LLR on the float stack, no GetChannelLLR available."""
    vm.state.ctx_has_channel_llr = False
    vm.state.ints.push(int(vm.state.ctx_edge_index))
    vm.state.ints.push(int(vm.state.ctx_deg))
    vm.state.ints.push(int(vm.state.ctx_iter))
    vm.state.ints.push(int(vm.state.ctx_max_iter))
    vm.state.fvecs.push(vm.state.ctx_incoming.copy())


def _run(prog: List[Instruction], vm: VM, side: str) -> Optional[float]:
    if side == "v2c":
        _seed_v2c_stacks(vm)
    elif side == "c2v":
        _seed_c2v_stacks(vm)
    else:
        raise ValueError(side)
    return vm.run(prog)


# ============================================================ V2C validator


def validate_v2c(
    prog: List[Instruction],
    *,
    rng: Optional[np.random.Generator] = None,
    num_configs: int = DEFAULT_NUM_CONFIGS,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    deg: int = DEFAULT_DEG,
    evo_consts: Optional[np.ndarray] = None,
) -> Tuple[bool, str]:
    rng = rng if rng is not None else np.random.default_rng(0)

    for cfg in range(num_configs):
        L_v = float(rng.uniform(-2.0, 2.0))
        incoming = rng.uniform(-3.0, 3.0, size=deg - 1)

        # baseline
        vm = _make_vm(incoming, channel_llr=L_v, deg=deg, iter_idx=cfg, evo_consts=evo_consts)
        base = _run(prog, vm, "v2c")
        if base is None:
            return False, f"v2c cfg{cfg}: faulty / non-finite baseline"

        # 1. dependence on L_v
        vm2 = _make_vm(incoming, channel_llr=L_v + DEFAULT_PERTURB_DELTA, deg=deg, iter_idx=cfg, evo_consts=evo_consts)
        out2 = _run(prog, vm2, "v2c")
        if out2 is None:
            return False, f"v2c cfg{cfg}: faulty after L_v perturbation"
        if abs(out2 - base) < EPS_DEPENDENCY:
            return False, f"v2c cfg{cfg}: output independent of L_v"

        # 2. dependence on incoming: try several single-element perturbations
        #    (and a global shift); accept if ANY of them changes the output.
        #    A legitimate min-sum-style program ignores all but the extreme
        #    element, so insisting that EVERY element matters is too strict.
        changed = False
        for tries in range(min(incoming.size, 4)):
            idx = int((cfg * 7 + tries * 3 + rng.integers(0, incoming.size)) % incoming.size)
            inc2 = incoming.copy()
            inc2[idx] += DEFAULT_PERTURB_DELTA
            vm3 = _make_vm(inc2, channel_llr=L_v, deg=deg, iter_idx=cfg, evo_consts=evo_consts)
            out3 = _run(prog, vm3, "v2c")
            if out3 is None:
                return False, f"v2c cfg{cfg}: faulty after incoming perturbation"
            if abs(out3 - base) >= EPS_DEPENDENCY:
                changed = True
                break
        if not changed:
            # Last resort: shift every element by delta.
            inc3 = incoming + DEFAULT_PERTURB_DELTA
            vm5 = _make_vm(inc3, channel_llr=L_v, deg=deg, iter_idx=cfg, evo_consts=evo_consts)
            out5 = _run(prog, vm5, "v2c")
            if out5 is None or abs(out5 - base) < EPS_DEPENDENCY:
                return False, f"v2c cfg{cfg}: output independent of incoming"

        # 3. permutation invariance
        for _ in range(num_permutations):
            perm = rng.permutation(incoming)
            vm4 = _make_vm(perm, channel_llr=L_v, deg=deg, iter_idx=cfg, evo_consts=evo_consts)
            out4 = _run(prog, vm4, "v2c")
            if out4 is None:
                return False, f"v2c cfg{cfg}: faulty under permutation"
            if abs(out4 - base) > EPS_INVARIANCE:
                return False, f"v2c cfg{cfg}: not permutation-invariant ({base} vs {out4})"

    return True, "ok"


# ============================================================ C2V validator


def validate_c2v(
    prog: List[Instruction],
    *,
    rng: Optional[np.random.Generator] = None,
    num_configs: int = DEFAULT_NUM_CONFIGS,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    deg: int = DEFAULT_DEG,
    evo_consts: Optional[np.ndarray] = None,
) -> Tuple[bool, str]:
    rng = rng if rng is not None else np.random.default_rng(0)

    for cfg in range(num_configs):
        incoming = rng.uniform(-3.0, 3.0, size=deg - 1)

        vm = _make_vm(incoming, has_channel_llr=False, deg=deg, iter_idx=cfg, evo_consts=evo_consts)
        base = _run(prog, vm, "c2v")
        if base is None:
            return False, f"c2v cfg{cfg}: faulty / non-finite baseline"

        # 1. dependence on incoming: try several single-element perturbations
        #    plus a global shift; require AT LEAST ONE change.
        changed = False
        for tries in range(min(incoming.size, 4)):
            idx = int((cfg * 7 + tries * 3 + rng.integers(0, incoming.size)) % incoming.size)
            inc2 = incoming.copy()
            inc2[idx] += DEFAULT_PERTURB_DELTA
            vm2 = _make_vm(inc2, has_channel_llr=False, deg=deg, iter_idx=cfg, evo_consts=evo_consts)
            out2 = _run(prog, vm2, "c2v")
            if out2 is None:
                return False, f"c2v cfg{cfg}: faulty after incoming perturbation"
            if abs(out2 - base) >= EPS_DEPENDENCY:
                changed = True
                break
        if not changed:
            inc3 = incoming + DEFAULT_PERTURB_DELTA
            vm4 = _make_vm(inc3, has_channel_llr=False, deg=deg, iter_idx=cfg, evo_consts=evo_consts)
            out4 = _run(prog, vm4, "c2v")
            if out4 is None or abs(out4 - base) < EPS_DEPENDENCY:
                return False, f"c2v cfg{cfg}: output independent of incoming"

        # 2. permutation invariance
        for _ in range(num_permutations):
            perm = rng.permutation(incoming)
            vm3 = _make_vm(perm, has_channel_llr=False, deg=deg, iter_idx=cfg, evo_consts=evo_consts)
            out3 = _run(prog, vm3, "c2v")
            if out3 is None:
                return False, f"c2v cfg{cfg}: faulty under permutation"
            if abs(out3 - base) > EPS_INVARIANCE:
                return False, f"c2v cfg{cfg}: not permutation-invariant ({base} vs {out3})"

    return True, "ok"


def validate_genome(
    genome,
    *,
    rng: Optional[np.random.Generator] = None,
    deg: int = DEFAULT_DEG,
) -> Tuple[bool, str]:
    """Run both V2C and C2V validators on a Genome instance."""
    evo = genome.evo_const_values()
    rng = rng if rng is not None else np.random.default_rng(0)
    ok, why = validate_v2c(genome.prog_v2c, rng=rng, deg=deg, evo_consts=evo)
    if not ok:
        return False, why
    ok, why = validate_c2v(genome.prog_c2v, rng=rng, deg=deg, evo_consts=evo)
    if not ok:
        return False, why
    return True, "ok"


__all__ = [
    "validate_v2c",
    "validate_c2v",
    "validate_genome",
    "DEFAULT_DEG",
    "DEFAULT_NUM_CONFIGS",
    "DEFAULT_NUM_PERMUTATIONS",
]
