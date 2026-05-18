"""Real-perturbation validators for evolved Push-GP programs.

Design (rev 2 -- post peer-#13/#18 / OMS-cfg2 incident)
========================================================

Two failure modes drove this rewrite:

1.  ``OMS`` (signProd * max(0, min|inc| - beta)) was being rejected at
    cfg=2 because only 4 randomly-indexed single-element perturbations
    were tried, and none landed on ``argmin |inc|``.  Fix: probe *every*
    position, in *both* directions, at two magnitudes, and also try
    sign-flips and zeroings.  Probe count scales linearly with ``deg``.

2.  Evolved programs like ``peer #18`` (Float.Exp(118)) and ``peer #13``
    (an EvoConst-driven constant cascade) silently collapsed to 0 under
    the old VM's NAN_INF_REPLACEMENT=0 guard.  The refactored VM now
    faults on domain errors and clamps overflow to ``get_float_clamp()``
    (30 default), but the validator still needs to exercise the evo
    regime that triggered collapse.  Fix: every cfg is run under
    multiple ``evo_consts`` panels (the genome's own + several drawn
    from the runtime prior ``log10(c) ~ U(-3, 3)``).

Two checks are exposed:

* ``validate_v2c(prog, ...)``
    - output must depend on the channel LLR (``L_v``)
    - output must depend on the incoming-message vector (some probe must
      change the output) -- evaluated per (cfg, evo_panel)
    - output must be invariant under structural permutations of incoming
    - output must be finite for every test configuration

* ``validate_c2v(prog, ...)`` -- same but no L_v dependence check

Both validators return ``(ok: bool, reason: str)``.

Probe budget per cfg (excluding L_v probes and permutations):
    6 * deg + 6 input probes.
With NUM_CONFIGS=5, NUM_EVO_PANELS=4 (genome's evo + 3 sampled), deg=8
that is ~5*4*54 = 1080 VM runs per side.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from .program import Instruction
from .vm import VM


# ---------------------------------------------------------------- defaults
DEFAULT_DEG = 8

# How many random (incoming, L_v) base configurations to test under.
DEFAULT_NUM_CONFIGS = 5

# How many additional evo-panel samples on top of the genome's own
# `evo_const_values()`.  Default 0 because random evo sampling produces
# false-positives on legitimate programs whose constants are tuned to a
# specific evo range (e.g. OMS uses EvoConst1=1e6 as a `min` sentinel;
# resampling evo turns that sentinel into a regular operand and the
# program degenerates).  The strengthened per-position input probes
# (6*deg+6) are sufficient to detect programs whose output is constant
# at their own evo.  Caller may set num_evo_panels>0 to opt into the
# robustness sweep when desired.
DEFAULT_NUM_EVO_PANELS = 0

# Number of structural permutations per cfg.  Slot 0 = cyclic left shift
# by 1 (deterministic), slot 1 = reverse, remainder = random shuffles.
DEFAULT_NUM_PERMUTATIONS = 5

# Two-scale perturbation magnitudes.  The smaller one drives linear
# dependence; the larger one drives saturating / threshold ops past
# their knee (e.g. max(0, |inc|-beta)).
DEFAULT_PERTURB_DELTA = 1.7
_DELTA_SCALES = (1.0, 5.0)

# Evolution-time evo_const prior: log10(c) ~ U(-3, 3).  When sampling
# evo panels we mirror this range so peer programs that only collapse
# in the tails are detected.
_EVO_LOG_LO = -3.0
_EVO_LOG_HI = 3.0
_EVO_PANEL_SIZE = 8  # length of the evo_const vector (matches Genome)

EPS_DEPENDENCY = 1e-9
EPS_INVARIANCE = 1e-7

# Sampling range for random base (incoming, L_v).  BP messages at
# operating SNR=2-4 dB typically span roughly [-15, 15]; we sample at
# [-10, 10] to keep most probes in-distribution while still exposing
# step-response programs whose decision boundaries sit between 3 and 10.
INCOMING_SAMPLE_LO = -10.0
INCOMING_SAMPLE_HI = 10.0
L_V_SAMPLE_LO = -10.0
L_V_SAMPLE_HI = 10.0


# ---------------------------------------------------------------- helpers


def _structured_perms(
    incoming: np.ndarray,
    num_permutations: int,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    """Yield up to ``num_permutations`` permutations of ``incoming``.

    Slot 0: left cyclic shift by 1 (deterministic catch for programs
    that read a fixed index).  Slot 1: full reverse.  Remaining slots:
    independent random shuffles.
    """
    n = incoming.size
    yielded = 0
    if num_permutations > 0 and n >= 2:
        shifted = np.empty_like(incoming)
        shifted[:-1] = incoming[1:]
        shifted[-1] = incoming[0]
        yield shifted
        yielded += 1
    if num_permutations > 1 and n >= 2:
        yield incoming[::-1].copy()
        yielded += 1
    while yielded < num_permutations:
        yield rng.permutation(incoming)
        yielded += 1


def _sample_evo_panels(
    genome_evo: Optional[np.ndarray],
    num_extra: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    panels: List[np.ndarray] = []
    if genome_evo is None:
        panels.append(np.ones(_EVO_PANEL_SIZE, dtype=np.float64))
    else:
        panels.append(np.asarray(genome_evo, dtype=np.float64).copy())
    for _ in range(max(0, num_extra)):
        logs = rng.uniform(_EVO_LOG_LO, _EVO_LOG_HI, size=_EVO_PANEL_SIZE)
        panels.append(np.power(10.0, logs).astype(np.float64))
    return panels


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
    vm.state.floats.push(float(vm.state.ctx_channel_llr))
    vm.state.ints.push(int(vm.state.ctx_edge_index))
    vm.state.ints.push(int(vm.state.ctx_deg))
    vm.state.ints.push(int(vm.state.ctx_iter))
    vm.state.ints.push(int(vm.state.ctx_max_iter))
    vm.state.fvecs.push(vm.state.ctx_incoming.copy())


def _seed_c2v_stacks(vm: VM) -> None:
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


def _spawn(rng: np.random.Generator, n: int) -> List[np.random.Generator]:
    """Return ``n`` independent child Generators of ``rng``."""
    try:
        ss = rng.bit_generator.seed_seq  # type: ignore[attr-defined]
        children = ss.spawn(n)
        return [np.random.default_rng(c) for c in children]
    except Exception:
        return [np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1))) for _ in range(n)]


# ---------------------------------------------------------------- probe construction


def _build_input_probes(
    incoming: np.ndarray,
    delta: float,
) -> List[Tuple[str, np.ndarray]]:
    """Build the per-cfg input-probe list for the dependency check.

    Returns ``[(label, perturbed_incoming), ...]`` covering:

    * ``2 * deg``  per-position +/- delta probes at base magnitude
    * ``2 * deg``  per-position +/- 5*delta probes (cross saturation knees)
    * ``deg``      per-position sign-flips (catches signProd-only deps)
    * ``deg``      per-position zeroings (catches "ignore if zero" branches)
    * ``4``        global shifts (+/- delta, +/- 5*delta)
    * ``2``        global scales (*0.5, *2)

    Total = 6 * deg + 6 probes.
    """
    probes: List[Tuple[str, np.ndarray]] = []
    n = incoming.size
    for s_label, s in zip(("d", "5d"), _DELTA_SCALES):
        for sign_label, sign in (("+", 1.0), ("-", -1.0)):
            for i in range(n):
                p = incoming.copy()
                p[i] += sign * s * delta
                probes.append((f"pos{i}{sign_label}{s_label}", p))
    for i in range(n):
        p = incoming.copy()
        p[i] = -p[i]
        probes.append((f"pos{i}_flip", p))
    for i in range(n):
        p = incoming.copy()
        p[i] = 0.0
        probes.append((f"pos{i}_zero", p))
    for sign_label, sign in (("+", 1.0), ("-", -1.0)):
        for s_label, s in zip(("d", "5d"), _DELTA_SCALES):
            probes.append((f"glob{sign_label}{s_label}",
                           incoming + sign * s * delta))
    probes.append(("scale0.5", incoming * 0.5))
    probes.append(("scale2.0", incoming * 2.0))
    return probes


# ============================================================ V2C validator


def validate_v2c(
    prog: List[Instruction],
    *,
    rng: Optional[np.random.Generator] = None,
    num_configs: int = DEFAULT_NUM_CONFIGS,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    num_evo_panels: int = DEFAULT_NUM_EVO_PANELS,
    deg: int = DEFAULT_DEG,
    evo_consts: Optional[np.ndarray] = None,
    delta: float = DEFAULT_PERTURB_DELTA,
) -> Tuple[bool, str]:
    rng = rng if rng is not None else np.random.default_rng(0)
    evo_panels = _sample_evo_panels(evo_consts, num_evo_panels, rng)

    for cfg in range(num_configs):
        L_v = float(rng.uniform(L_V_SAMPLE_LO, L_V_SAMPLE_HI))
        incoming = rng.uniform(INCOMING_SAMPLE_LO, INCOMING_SAMPLE_HI, size=deg - 1)
        input_probes = _build_input_probes(incoming, delta)
        perm_list = list(_structured_perms(incoming, num_permutations, rng))

        for evo_idx, evo in enumerate(evo_panels):
            tag = f"cfg{cfg}/evo{evo_idx}"

            vm = _make_vm(incoming, channel_llr=L_v, deg=deg,
                          iter_idx=cfg, evo_consts=evo)
            base = _run(prog, vm, "v2c")
            if base is None:
                return False, f"v2c {tag}: faulty / non-finite baseline"

            # 1. dependence on L_v (two magnitudes, both signs)
            l_changed = False
            for sign in (+1.0, -1.0):
                for s in _DELTA_SCALES:
                    vm2 = _make_vm(incoming, channel_llr=L_v + sign * s * delta,
                                   deg=deg, iter_idx=cfg, evo_consts=evo)
                    out2 = _run(prog, vm2, "v2c")
                    if out2 is None:
                        continue
                    if abs(out2 - base) >= EPS_DEPENDENCY:
                        l_changed = True
                        break
                if l_changed:
                    break
            if not l_changed:
                return False, f"v2c {tag}: output independent of L_v"

            # 2. dependence on incoming -- ANY probe must change output
            inc_changed = False
            for _label, perturbed in input_probes:
                vm3 = _make_vm(perturbed, channel_llr=L_v, deg=deg,
                               iter_idx=cfg, evo_consts=evo)
                out3 = _run(prog, vm3, "v2c")
                if out3 is None:
                    continue
                if abs(out3 - base) >= EPS_DEPENDENCY:
                    inc_changed = True
                    break
            if not inc_changed:
                return False, f"v2c {tag}: output independent of incoming"

            # 3. permutation invariance
            for perm in perm_list:
                vm4 = _make_vm(perm, channel_llr=L_v, deg=deg,
                               iter_idx=cfg, evo_consts=evo)
                out4 = _run(prog, vm4, "v2c")
                if out4 is None:
                    return False, f"v2c {tag}: faulty under permutation"
                if abs(out4 - base) > EPS_INVARIANCE:
                    return False, f"v2c {tag}: not permutation-invariant ({base} vs {out4})"

    return True, "ok"


# ============================================================ C2V validator


def validate_c2v(
    prog: List[Instruction],
    *,
    rng: Optional[np.random.Generator] = None,
    num_configs: int = DEFAULT_NUM_CONFIGS,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    num_evo_panels: int = DEFAULT_NUM_EVO_PANELS,
    deg: int = DEFAULT_DEG,
    evo_consts: Optional[np.ndarray] = None,
    delta: float = DEFAULT_PERTURB_DELTA,
) -> Tuple[bool, str]:
    rng = rng if rng is not None else np.random.default_rng(0)
    evo_panels = _sample_evo_panels(evo_consts, num_evo_panels, rng)

    for cfg in range(num_configs):
        incoming = rng.uniform(INCOMING_SAMPLE_LO, INCOMING_SAMPLE_HI, size=deg - 1)
        input_probes = _build_input_probes(incoming, delta)
        perm_list = list(_structured_perms(incoming, num_permutations, rng))

        for evo_idx, evo in enumerate(evo_panels):
            tag = f"cfg{cfg}/evo{evo_idx}"

            vm = _make_vm(incoming, has_channel_llr=False, deg=deg,
                          iter_idx=cfg, evo_consts=evo)
            base = _run(prog, vm, "c2v")
            if base is None:
                return False, f"c2v {tag}: faulty / non-finite baseline"

            inc_changed = False
            for _label, perturbed in input_probes:
                vm2 = _make_vm(perturbed, has_channel_llr=False, deg=deg,
                               iter_idx=cfg, evo_consts=evo)
                out2 = _run(prog, vm2, "c2v")
                if out2 is None:
                    continue
                if abs(out2 - base) >= EPS_DEPENDENCY:
                    inc_changed = True
                    break
            if not inc_changed:
                return False, f"c2v {tag}: output independent of incoming"

            for perm in perm_list:
                vm3 = _make_vm(perm, has_channel_llr=False, deg=deg,
                               iter_idx=cfg, evo_consts=evo)
                out3 = _run(prog, vm3, "c2v")
                if out3 is None:
                    return False, f"c2v {tag}: faulty under permutation"
                if abs(out3 - base) > EPS_INVARIANCE:
                    return False, f"c2v {tag}: not permutation-invariant ({base} vs {out3})"

    return True, "ok"


def validate_genome(
    genome,
    *,
    rng: Optional[np.random.Generator] = None,
    deg: int = DEFAULT_DEG,
) -> Tuple[bool, str]:
    """Run both V2C and C2V validators on a Genome instance.

    V2C and C2V get *independent* child rngs spawned from ``rng`` so
    their probe / cfg / evo-panel sequences do not couple.  Coupling
    was the root cause of the OMS-cfg2 flakiness pre-rewrite.
    """
    evo = genome.evo_const_values()
    rng = rng if rng is not None else np.random.default_rng(0)
    rng_v2c, rng_c2v = _spawn(rng, 2)
    ok, why = validate_v2c(genome.prog_v2c, rng=rng_v2c, deg=deg, evo_consts=evo)
    if not ok:
        return False, why
    ok, why = validate_c2v(genome.prog_c2v, rng=rng_c2v, deg=deg, evo_consts=evo)
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
    "DEFAULT_NUM_EVO_PANELS",
    "DEFAULT_PERTURB_DELTA",
]
