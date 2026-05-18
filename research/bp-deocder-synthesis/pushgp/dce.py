"""Behavioral Dead-Code Elimination (DCE).

This module implements a delta-debugging-style program reducer that uses
the 32-entry behavioral fingerprint (``pushgp.evolution._behav_fingerprint``)
as the *correctness oracle* for what counts as "dead" code.

Why behavioral DCE rather than syntactic DCE
============================================

Push-style programs make naive (provenance-only) DCE unsafe because:

  * Removing an instruction changes the *stack depth*; a "dead" Float.Pop
    that just discards a value silently re-targets which two operands the
    next live op will consume.
  * Cross-stack effects: a "dead" Int.Add consumes two Ints, which may
    have been the loop bound for a "live" Exec.DoTimes nearby.
  * Memory side-effects: ``Mem.Write`` writes a slot that may be read by
    a later iteration's V2C / C2V program, even if the writer's value is
    not on the float stack at end of execution.

The only safe transformation is one that preserves the program's
*observable behavior* on a representative panel of inputs.  The 32-entry
behavioral panel + ``%.8g`` quantization defines exactly that observable
behavior, and is byte-identical between Python and the C++ seeder.

Algorithm
=========

::

    fp0 = behav_fp(side, prog)
    while True:
        for pos in walk_all_positions(prog):           # depth-first
            cand = prog with the instruction at `pos` removed
            if behav_fp(side, cand) == fp0:
                prog = cand                            # safe to remove
                break                                  # restart (positions changed)
        else:
            break  # no further removals
    return prog

Each iteration runs O(N) fingerprint evaluations (each = 32 VM runs);
total cost is O(N^2) FP evaluations in the worst case.  In practice the
search exits early because most "dead" instructions fall out in the
first pass.

Caveats
=======

  * **Single-side fingerprints only**.  V2C panel does not exercise C2V
    behavior, so a Mem.Write in V2C that is dead w.r.t. the V2C output
    *itself* may still influence subsequent C2V execution via shared
    memory.  Test T6 in :mod:`pushgp.tests.test_dce` exercises this risk;
    if it fails, the production DCE should switch to one of:
      (a) a conservative "never delete Mem.Write*" rule, or
      (b) a joint V/C panel that runs full BP iterations.

  * **Panel coverage**.  32 entries is a finite oracle; two programs
    can be fp-equal but BER-different at unsampled inputs.  Test T4
    runs full LDPC BER comparison on reduced vs original to detect this.

Public API
==========

  * :func:`behavioral_reduce` -- reduce a single program for one side.
  * :func:`reduce_seeded_population` -- post-process a list of seeded
    programs (intended as a hook *after* parallel_seed / Python seeder
    returns; not coupled to any seeder implementation).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from .program import Instruction, deep_copy_program, program_length


def _try_import_pushgp_cpp_dce():
    """Import ``pushgp_cpp_dce``; locate sibling ``cpp_dce/`` build dir.

    The ``.pyd`` lives at ``<project_root>/cpp_dce/pushgp_cpp_dce.*.pyd``
    and is not installed to site-packages.  We make it importable by
    prepending that directory to ``sys.path`` on first attempt.  Cached.
    """
    import importlib
    import sys as _sys
    from pathlib import Path as _Path
    try:
        return importlib.import_module("pushgp_cpp_dce")
    except ImportError:
        pass
    here = _Path(__file__).resolve()
    # pushgp/dce.py -> pushgp/ -> bp-deocder-synthesis/ ; cpp_dce/ sibling.
    candidates = [
        here.parent.parent / "cpp_dce",
    ]
    for c in candidates:
        if c.exists() and str(c) not in _sys.path:
            _sys.path.insert(0, str(c))
    try:
        return importlib.import_module("pushgp_cpp_dce")
    except ImportError:
        return None

__all__ = [
    "behavioral_reduce",
    "behavioral_reduce_bp",
    "reduce_seeded_population",
    "reduce_populations_bp",
    "DCEStats",
]


class DCEStats:
    """Collected statistics for a single ``behavioral_reduce`` call."""

    __slots__ = (
        "side",
        "size_before",
        "size_after",
        "passes",
        "fp_evals",
        "removed_positions",
    )

    def __init__(self, side: str, size_before: int) -> None:
        self.side: str = side
        self.size_before: int = size_before
        self.size_after: int = size_before
        self.passes: int = 0
        self.fp_evals: int = 0
        self.removed_positions: List[Tuple[int, ...]] = []

    @property
    def removed(self) -> int:
        return self.size_before - self.size_after

    @property
    def reduction_ratio(self) -> float:
        if self.size_before == 0:
            return 0.0
        return self.removed / float(self.size_before)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"DCEStats(side={self.side!r}, {self.size_before}->{self.size_after} "
            f"({self.reduction_ratio:.1%}), passes={self.passes}, "
            f"fp_evals={self.fp_evals})"
        )


# ---------------------------------------------------------------------------
# Position walk + remove helpers
# ---------------------------------------------------------------------------

# A "position" is a path of integers identifying one Instruction inside a
# nested program.  The path describes which list to descend into at each
# level.  At a top-level list, the path is just (i,).  At a code_block
# inside instruction i, the path is (i, "cb", j).  At a code_block2, the
# path is (i, "cb2", j).  Recursive nesting extends the tuple.

PositionStep = object  # int | "cb" | "cb2"
Position = Tuple[PositionStep, ...]


def _walk_positions(prog: List[Instruction], prefix: Position = ()
                    ) -> List[Position]:
    """Depth-first list of every Instruction position inside ``prog``.

    The order is: top-level instruction, then its code_block contents,
    then its code_block2 contents, then the next top-level instruction.
    Children are listed *after* their parent so a removal of the parent
    automatically discards the children in one step.
    """
    positions: List[Position] = []
    for i, ins in enumerate(prog):
        positions.append(prefix + (i,))
        if ins.code_block is not None:
            positions.extend(
                _walk_positions(ins.code_block, prefix + (i, "cb"))
            )
        if ins.code_block2 is not None:
            positions.extend(
                _walk_positions(ins.code_block2, prefix + (i, "cb2"))
            )
    return positions


def _remove_at(prog: List[Instruction], pos: Position) -> List[Instruction]:
    """Return a deep-copy of ``prog`` with the instruction at ``pos`` removed.

    Removing a control-instruction (with ``code_block``) deletes the
    entire instruction *including* its nested blocks.  Removing an
    instruction inside a code_block leaves the surrounding control
    instruction in place with its child list shortened.
    """
    new = deep_copy_program(prog)
    if not pos:
        raise ValueError("empty position")
    # Walk down to the *parent list* that contains the instruction to
    # delete, then pop the final integer index from it.
    container: List[Instruction] = new
    i = 0
    while i < len(pos) - 1:
        step = pos[i]
        if not isinstance(step, int):
            raise ValueError(f"position step {i} must be int, got {step!r}")
        ins = container[step]
        next_step = pos[i + 1]
        if next_step == "cb":
            assert ins.code_block is not None, "cb step on instr w/o code_block"
            container = ins.code_block
            i += 2
        elif next_step == "cb2":
            assert ins.code_block2 is not None, "cb2 step on instr w/o code_block2"
            container = ins.code_block2
            i += 2
        else:
            # next_step is an int — but that means current step IS the
            # final instruction to remove (i.e. we shouldn't be here).
            raise ValueError(
                f"malformed position {pos!r}: int followed by int at i={i}"
            )
    last = pos[-1]
    if not isinstance(last, int):
        raise ValueError(f"final position step must be int, got {last!r}")
    del container[last]
    return new


# ---------------------------------------------------------------------------
# Public reducer
# ---------------------------------------------------------------------------

# Type of a fingerprint function; deferred import to avoid evolution.py
# pulling DCE on every load.
FpFn = Callable[[str, List[Instruction]], str]


def _default_fp() -> FpFn:
    from .evolution import _behav_fingerprint  # noqa: WPS433
    return _behav_fingerprint


def _zero_incoming_fingerprint(side: str, prog, evo_consts) -> str:
    """Extra panel: BP-realistic ``it=0`` case (incoming = zeros).

    Real BP at iteration 0 always passes an all-zero ``incoming`` vector
    (no c2v history yet).  Three details that make this case *not*
    covered by the default 32-point dedup panel:
      1. the default panel uses NON-zero incoming arrays;
      2. the default panel hard-codes incoming length 7, while real BP
         passes length ``deg - 1`` (the target edge is excluded), and
         programs can branch on ``FVec.Len``;
      3. ``max_iter`` is exposed on the int stack and programs may
         branch on it; the default panel uses the VM's default (25)
         while production BER tests typically use much smaller values
         (4, 8, 25).
    We therefore sweep panel L_v values (32) crossed with all realistic
    degrees ``deg in {2..8}`` (7) and a few realistic ``max_iter``
    values ``{4, 25}`` (2), feeding ``incoming = zeros(deg - 1)``.
    Total: 448 evaluations per side.  See
    ``code_review/debug_dce_divergence.py`` for the diagnostic that
    motivated this.
    """
    import numpy as _np
    from .evolution import (
        _BEHAV_PANEL_V2C_LV, _BEHAV_PANEL_SIZE, _BEHAV_QUANT,
    )
    from .validators import _make_vm, _seed_v2c_stacks, _seed_c2v_stacks
    outs: List[str] = []
    finite = 0
    degs = (2, 3, 4, 5, 6, 7, 8)
    max_iters = (4, 25)
    for i in range(_BEHAV_PANEL_SIZE):
        for deg in degs:
            for max_iter in max_iters:
                inc_len = max(deg - 1, 0)
                zeros = _np.zeros(inc_len, dtype=_np.float64)
                if side == "v2c":
                    L_v = float(_BEHAV_PANEL_V2C_LV[i])
                    vm = _make_vm(zeros, channel_llr=L_v, deg=deg,
                                  iter_idx=0, max_iter=max_iter,
                                  evo_consts=evo_consts)
                    _seed_v2c_stacks(vm)
                else:
                    vm = _make_vm(zeros, has_channel_llr=False,
                                  deg=deg, iter_idx=0, max_iter=max_iter,
                                  evo_consts=evo_consts)
                    _seed_c2v_stacks(vm)
                try:
                    out = vm.run(prog)
                except Exception:
                    out = None
                if out is None or not _np.isfinite(out):
                    outs.append("nan")
                else:
                    finite += 1
                    outs.append(format(float(out), f".{_BEHAV_QUANT}g"))
    if finite == 0:
        return "Z_NAN"
    return "|".join(outs)


def behavioral_reduce(
    prog: List[Instruction],
    side: str,
    *,
    fp_fn: Optional[FpFn] = None,
    evo_consts: Optional["np.ndarray"] = None,
    max_passes: int = 800,
    max_fp_evals: Optional[int] = None,
    stats: Optional[DCEStats] = None,
) -> List[Instruction]:
    """Reduce ``prog`` to a behaviorally-equivalent shorter program.

    Parameters
    ----------
    prog : List[Instruction]
        Program to reduce.  Not mutated in place.
    side : str
        ``"v2c"`` or ``"c2v"`` -- selects which behavioral panel applies.
    fp_fn : callable, optional
        Override fingerprint function (used by tests).  Defaults to the
        production ``pushgp.evolution._behav_fingerprint``.
    evo_consts : np.ndarray, optional
        Genome's actual evolved-constant K values (``10 ** log_constants``).
        When provided, the fingerprint oracle uses these constants instead of
        the default fixed panel (``_BEHAV_DEFAULT_K``), so that instructions
        referencing ``ctx_evo_constants`` are NOT incorrectly pruned.
        Ignored when ``fp_fn`` is explicitly provided.
    max_passes : int
        Hard cap on outer-loop iterations.  Each pass removes at most one
        instruction; the loop exits early when a full sweep finds nothing
        to remove.  Default of 800 comfortably exceeds MAX_PROG_LEN=80
        (all-dead 80-instruction programs need at most 80 passes).
    max_fp_evals : int, optional
        Hard cap on total fingerprint evaluations.  Prevents pathological
        runtime on adversarial inputs.
    stats : DCEStats, optional
        If provided, populated with run statistics in place.

    Returns
    -------
    List[Instruction]
        A new program with the same behavioral fingerprint as ``prog``
        but possibly fewer instructions.  Always a deep copy; original
        is untouched.
    """
    if fp_fn is None:
        if evo_consts is not None:
            from .evolution import _behav_fingerprint as _bfp
            _k = evo_consts
            # Sample the panel at multiple iter values so DCE catches
            # instructions that are dead at it=0 but live at higher
            # iterations.  At it=0 the panel's `incoming` is the only
            # signal; instructions guarded by `it != 0` look dead.
            _iters = (0, 2, 4)
            def _multi_iter_fp(_side, _prog):
                parts = [
                    _bfp(_side, _prog, evo_consts=_k, iter_idx=it)
                    for it in _iters
                ]
                # Crucial: cover BP iter=0 reality (incoming=zeros)
                # across all realistic degrees — the default panel
                # uses non-zero incoming and only one deg (8), so
                # programs that branch on (L_v, deg) at zero
                # incoming would otherwise pass the panel while
                # behaving differently in real BP.
                parts.append(_zero_incoming_fingerprint(_side, _prog, _k))
                return "||".join(parts)
            fp_fn = _multi_iter_fp
        else:
            fp_fn = _default_fp()
    if side not in ("v2c", "c2v"):
        raise ValueError(f"side must be 'v2c' or 'c2v', got {side!r}")

    cur: List[Instruction] = deep_copy_program(prog)
    n0 = program_length(cur)
    if stats is None:
        stats = DCEStats(side=side, size_before=n0)
    else:
        stats.side = side
        stats.size_before = n0
        stats.size_after = n0
        stats.passes = 0
        stats.fp_evals = 0
        stats.removed_positions = []

    fp0 = fp_fn(side, cur)
    stats.fp_evals += 1

    for pass_idx in range(max_passes):
        stats.passes = pass_idx + 1
        removed_in_pass = False
        positions = _walk_positions(cur)
        # Iterate from deepest positions to shallowest so removing a leaf
        # doesn't invalidate positions of other unrelated leaves.  We do
        # one removal then restart anyway, but ordering still helps when
        # the first deeply-nested removal succeeds — fewer wasted FP
        # evaluations than starting from the root.
        for pos in reversed(positions):
            if max_fp_evals is not None and stats.fp_evals >= max_fp_evals:
                break
            try:
                cand = _remove_at(cur, pos)
            except (IndexError, AssertionError, ValueError):
                # Position became invalid (shouldn't happen on a fresh
                # walk but guard against bugs).  Skip.
                continue
            fp = fp_fn(side, cand)
            stats.fp_evals += 1
            if fp == fp0:
                cur = cand
                stats.removed_positions.append(tuple(pos))
                stats.size_after = program_length(cur)
                removed_in_pass = True
                break  # restart pass with fresh positions
        if not removed_in_pass:
            break
        if max_fp_evals is not None and stats.fp_evals >= max_fp_evals:
            break

    return cur


def behavioral_reduce_bp(
    prog: List[Instruction],
    side: str,
    *,
    peer_prog: List[Instruction],
    log_constants: "np.ndarray",
    par,
    rx_llrs: "Sequence[np.ndarray]",
    max_iter: int = 8,
    max_passes: int = 800,
    max_decode_evals: Optional[int] = None,
    decimals: int = 6,
    stats: Optional[DCEStats] = None,
    use_cpp: bool = True,
) -> List[Instruction]:
    """Reduce ``prog`` by direct full-BP equivalence on a frame bank.

    Stronger oracle than the static behavioral panel: for every candidate
    (one instruction removed) we run the entire BP decoder on EACH
    pre-generated ``rx_llr`` frame using ``(prog, peer_prog)`` for the
    two sides and require the resulting post-LLR vector to match the
    baseline (rounded to ``decimals``) on ALL frames.  Removal is
    accepted only if every frame agrees.

    Parameters
    ----------
    prog : List[Instruction]
        Program to reduce (one side).
    side : str
        ``"v2c"`` or ``"c2v"`` -- which side ``prog`` belongs to.
    peer_prog : List[Instruction]
        The other side's program; held fixed during this reduction.
    log_constants : np.ndarray
        Genome's ``log_constants`` (length N_EVO_CONSTS).
    par : LiftedParity
        Parity-check structure (e.g. ``build_parity(2,1,2)``).
    rx_llrs : Sequence[np.ndarray]
        One or more pre-generated channel LLR frames.  Removal must
        preserve BP post-LLR on every frame.  Multi-frame banks are
        recommended (covers multiple SNR / noise realizations).
    max_iter : int
        BP max iterations (default 8).
    max_passes, max_decode_evals, decimals, stats : see ``behavioral_reduce``.

    Returns
    -------
    List[Instruction]
        A new program; deep copy.
    """
    import numpy as _np
    from .genome import Genome
    from pushgp_ldpc.adapter import make_callables
    from ldpc_5g import decode_bp

    if side not in ("v2c", "c2v"):
        raise ValueError(f"side must be 'v2c' or 'c2v', got {side!r}")
    if len(rx_llrs) == 0:
        raise ValueError("rx_llrs must contain at least one frame")

    # ------------------------------------------------------------------
    # Fast path: C++ port (pushgp_cpp_dce.reduce_bp).  Tests assert
    # structural equality to the Python loop below.  NO silent fallback:
    # if cpp is requested and unavailable / errors out, raise loudly so
    # bugs are not masked.  Pass use_cpp=False to force the reference.
    # ------------------------------------------------------------------
    if use_cpp:
        _cdce = _try_import_pushgp_cpp_dce()
        if _cdce is None:
            raise RuntimeError(
                "behavioral_reduce_bp(use_cpp=True) but pushgp_cpp_dce "
                "is not importable; build cpp_dce/ or pass use_cpp=False."
            )
        from .genome import Genome, program_to_list, dict_to_instruction
        import numpy as _np_local
        g = Genome(prog_v2c=prog if side == "v2c" else peer_prog,
                   prog_c2v=peer_prog if side == "v2c" else prog,
                   log_constants=log_constants)
        evo = g.evo_const_values().astype(_np_local.float64)
        parH = _cdce.build_parity_handle(par)
        n0 = program_length(prog)
        red_dict, st = _cdce.reduce_bp(
            program_to_list(prog), side, program_to_list(peer_prog),
            parH, list(rx_llrs), evo,
            int(max_iter), int(max_passes),
            int(max_decode_evals) if max_decode_evals is not None else -1,
            int(decimals),
        )
        reduced = [dict_to_instruction(d) for d in red_dict]
        if stats is not None:
            stats.side = side
            stats.size_before = n0
            stats.size_after  = int(st["size_after"])
            stats.passes      = int(st["passes"])
            stats.fp_evals    = int(st["fp_evals"])
            stats.removed_positions = []  # cpp positions use {idx,desc} tuples
        return reduced

    def _decode_all(p):
        v = p if side == "v2c" else peer_prog
        c = peer_prog if side == "v2c" else p
        try:
            g = Genome(prog_v2c=v, prog_c2v=c, log_constants=log_constants)
            v_fn, c_fn = make_callables(g)
        except Exception:
            return None
        outs = []
        for rx in rx_llrs:
            try:
                outs.append(decode_bp(rx.copy(), par, v2c_fn=v_fn, c2v_fn=c_fn,
                                     max_iter=max_iter, offset=0.25,
                                     code_rate=0.5))
            except Exception:
                return None
        return outs

    def _eq_all(a, b):
        if a is None or b is None:
            return False
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if x.shape != y.shape:
                return False
            if not _np.array_equal(_np.round(x, decimals), _np.round(y, decimals)):
                return False
        return True

    cur: List[Instruction] = deep_copy_program(prog)
    n0 = program_length(cur)
    if stats is None:
        stats = DCEStats(side=side, size_before=n0)
    else:
        stats.side = side
        stats.size_before = n0
        stats.size_after = n0
        stats.passes = 0
        stats.fp_evals = 0
        stats.removed_positions = []

    base = _decode_all(cur)
    stats.fp_evals += 1
    if base is None:
        return cur

    for pass_idx in range(max_passes):
        stats.passes = pass_idx + 1
        removed_in_pass = False
        for pos in reversed(_walk_positions(cur)):
            if max_decode_evals is not None and stats.fp_evals >= max_decode_evals:
                break
            try:
                cand = _remove_at(cur, pos)
            except (IndexError, AssertionError, ValueError):
                continue
            cand_out = _decode_all(cand)
            stats.fp_evals += 1
            if _eq_all(cand_out, base):
                cur = cand
                stats.removed_positions.append(tuple(pos))
                stats.size_after = program_length(cur)
                removed_in_pass = True
                break
        if not removed_in_pass:
            break
        if max_decode_evals is not None and stats.fp_evals >= max_decode_evals:
            break

    return cur


def reduce_seeded_population(
    progs: Sequence[List[Instruction]],
    side: str,
    *,
    min_size: int = 1,
    fp_fn: Optional[FpFn] = None,
    evo_consts_list: Optional[Sequence["np.ndarray"]] = None,
    max_passes_per_prog: int = 800,
    max_fp_evals_per_prog: Optional[int] = 80_000,
    on_progress: Optional[Callable[[int, int, DCEStats], None]] = None,
) -> Tuple[List[List[Instruction]], List[DCEStats]]:
    """Post-process a freshly-seeded population by behavioral DCE.

    Intended to be plugged in *after* the seeder (Python or C++) returns
    its valid programs.  This is **decoupled from the seeder
    implementation** — pass any ``Sequence[Program]`` in, get a parallel
    list of reduced programs out.  The interface deliberately matches
    what both Python ``parallel_fill_random`` and C++ ``parallel_seed``
    yield (lists of native :class:`pushgp.program.Instruction` lists).

    Parameters
    ----------
    progs : sequence of programs
        Seeded population to reduce.
    side : str
        ``"v2c"`` or ``"c2v"``.
    min_size : int
        Programs already at or below this length are passed through
        unchanged (saves work on already-tiny programs).
    fp_fn : callable, optional
        Override fingerprint function.
    evo_consts_list : sequence of np.ndarray, optional
        Per-program genome K values (``10 ** log_constants``).  Must be
        the same length as ``progs`` when provided.  Passed to each
        ``behavioral_reduce`` call so that evo-constant-dependent
        instructions are not incorrectly pruned.
    max_passes_per_prog : int
        Per-program cap on reducer passes.
    max_fp_evals_per_prog : int, optional
        Per-program cap on FP evaluations.  ``None`` disables.
    on_progress : callable, optional
        Called as ``on_progress(i, n, stats)`` after each program.

    Returns
    -------
    reduced_progs, stats_list
        Parallel lists of equal length to ``progs``.
    """
    out_progs: List[List[Instruction]] = []
    out_stats: List[DCEStats] = []
    n = len(progs)
    for i, prog in enumerate(progs):
        if program_length(prog) <= min_size:
            stats = DCEStats(side=side, size_before=program_length(prog))
            stats.size_after = stats.size_before
            out_progs.append(deep_copy_program(prog))
            out_stats.append(stats)
            if on_progress is not None:
                on_progress(i, n, stats)
            continue
        stats = DCEStats(side=side, size_before=program_length(prog))
        prog_evo_consts = (
            evo_consts_list[i] if evo_consts_list is not None else None
        )
        reduced = behavioral_reduce(
            prog,
            side,
            fp_fn=fp_fn,
            evo_consts=prog_evo_consts,
            max_passes=max_passes_per_prog,
            max_fp_evals=max_fp_evals_per_prog,
            stats=stats,
        )
        out_progs.append(reduced)
        out_stats.append(stats)
        if on_progress is not None:
            on_progress(i, n, stats)
    return out_progs, out_stats


def _validator_guard(
    orig_progs: Sequence[List[Instruction]],
    reduced_progs: List[List[Instruction]],
    stats_list: List[DCEStats],
    side: str,
    pop_k: Sequence["np.ndarray"],
    rng_seed: int = 0,
) -> int:
    """Revert any reduced program that fails the validator while the
    original passed.  Mutates ``reduced_progs`` and ``stats_list`` in
    place.  Returns the number of reversions.

    Invariant: ``len(orig_progs) == len(reduced_progs) == len(stats_list)``.
    Per-program evo_consts come from ``pop_k[i]`` (log_constants ->
    10**log).  Each program is validated with a deterministic Generator
    seeded from ``rng_seed + i`` to ensure reproducibility.
    """
    import numpy as _np
    from .genome import Genome
    from .validators import validate_v2c, validate_c2v
    validator = validate_v2c if side == "v2c" else validate_c2v
    n_reverted = 0
    for i in range(len(orig_progs)):
        red = reduced_progs[i]
        orig = orig_progs[i]
        if len(red) == len(orig):
            continue  # nothing was removed; trivially same as orig
        # Compute evo for this individual.
        g = Genome(prog_v2c=orig if side == "v2c" else orig,
                   prog_c2v=orig if side == "c2v" else orig,
                   log_constants=pop_k[i])
        evo = g.evo_const_values().astype(_np.float64)
        rng_red = _np.random.default_rng(rng_seed + i)
        ok_red, _ = validator(red, rng=rng_red, evo_consts=evo)
        if ok_red:
            continue
        # Reduced fails — does the original pass?
        rng_orig = _np.random.default_rng(rng_seed + i)
        ok_orig, _ = validator(orig, rng=rng_orig, evo_consts=evo)
        if not ok_orig:
            continue  # both fail; DCE didn't worsen it
        # Revert.
        reduced_progs[i] = deep_copy_program(orig)
        stats_list[i].size_after = program_length(orig)
        stats_list[i].removed_positions = []
        # Tag the stats so callers can count reversions.
        try:
            stats_list[i].reverted_by_validator = True  # type: ignore[attr-defined]
        except Exception:
            pass
        n_reverted += 1
    return n_reverted


def reduce_populations_bp(
    pop_v: Sequence[List[Instruction]],
    pop_c: Sequence[List[Instruction]],
    pop_k: Sequence["np.ndarray"],
    *,
    par,
    rx_llrs: "Sequence[np.ndarray]",
    max_iter: int = 8,
    max_passes: int = 800,
    max_decode_evals: int = -1,
    decimals: int = 6,
    threads: int = 1,
    use_cpp: bool = True,
    rng: Optional["np.random.Generator"] = None,
    on_progress: Optional[Callable[[str, int, int, float], None]] = None,
    validator_guard: bool = True,
) -> Tuple[List[List[Instruction]], List[List[Instruction]],
           List[DCEStats], List[DCEStats]]:
    """Apply BP-equivalence DCE to a whole two-pop population.

    For each i in [0, n), reduce ``pop_v[i]`` against a randomly-sampled
    peer ``pop_c[π_v(i)]`` and reduce ``pop_c[i]`` against a randomly-
    sampled peer ``pop_v[π_c(i)]``.  Constants come from ``pop_k[i]``.

    When ``use_cpp=True`` (default), the entire batch is submitted to
    ``pushgp_cpp_dce.reduce_bp_batch`` with ``threads`` workers; this
    keeps the top-level orchestration in Python while pushing the
    O(N^2) BP-evaluation work into native std::thread.

    Returns
    -------
    (new_pop_v, new_pop_c, stats_v, stats_c)
        Lists parallel to inputs.  Programs are deep copies; originals
        unchanged.
    """
    import numpy as _np
    from .genome import Genome, program_to_list, dict_to_instruction

    n = len(pop_v)
    if n == 0:
        return [], [], [], []
    if len(pop_c) != n or len(pop_k) != n:
        raise ValueError(
            f"pop_v / pop_c / pop_k must have equal length; "
            f"got {n}, {len(pop_c)}, {len(pop_k)}"
        )
    if rng is None:
        rng = _np.random.default_rng()

    perm_v = _np.asarray(rng.permutation(n))  # peer-C index for each V
    perm_c = _np.asarray(rng.permutation(n))  # peer-V index for each C

    # Precompute evo vectors (10 ** log_constants).  Same length as n.
    evo_all = []
    for i in range(n):
        g = Genome(prog_v2c=pop_v[i], prog_c2v=pop_c[i],
                   log_constants=pop_k[i])
        evo_all.append(g.evo_const_values().astype(_np.float64))

    if use_cpp:
        _cdce = _try_import_pushgp_cpp_dce()
        if _cdce is None:
            raise RuntimeError(
                "reduce_populations_bp(use_cpp=True) but pushgp_cpp_dce "
                "is not importable; build cpp_dce/ or pass use_cpp=False."
            )
        parH = _cdce.build_parity_handle(par)
        rx_list = list(rx_llrs)
        # Build a single job list: first n jobs are V-side, then C.
        jobs = []
        for i in range(n):
            jobs.append({
                "prog": program_to_list(pop_v[i]),
                "side": "v2c",
                "peer_prog": program_to_list(pop_c[int(perm_v[i])]),
                "evo": evo_all[i],
            })
        for i in range(n):
            jobs.append({
                "prog": program_to_list(pop_c[i]),
                "side": "c2v",
                "peer_prog": program_to_list(pop_v[int(perm_c[i])]),
                "evo": evo_all[i],
            })

        cb = None
        if on_progress is not None:
            def _cb(done, total, elapsed):
                on_progress("batch", int(done), int(total), float(elapsed))
            cb = _cb

        results = _cdce.reduce_bp_batch(
            jobs, parH, rx_list,
            int(max_iter), int(max_passes), int(max_decode_evals),
            int(decimals), int(max(1, threads)), cb,
        )
        new_pop_v: List[List[Instruction]] = []
        new_pop_c: List[List[Instruction]] = []
        stats_v: List[DCEStats] = []
        stats_c: List[DCEStats] = []
        for i in range(n):
            r = results[i]
            new_pop_v.append([dict_to_instruction(d) for d in r["prog"]])
            st = DCEStats(side="v2c", size_before=int(r["stats"]["size_before"]))
            st.size_after = int(r["stats"]["size_after"])
            st.passes    = int(r["stats"]["passes"])
            st.fp_evals  = int(r["stats"]["fp_evals"])
            stats_v.append(st)
        for i in range(n):
            r = results[n + i]
            new_pop_c.append([dict_to_instruction(d) for d in r["prog"]])
            st = DCEStats(side="c2v", size_before=int(r["stats"]["size_before"]))
            st.size_after = int(r["stats"]["size_after"])
            st.passes    = int(r["stats"]["passes"])
            st.fp_evals  = int(r["stats"]["fp_evals"])
            stats_c.append(st)
        if validator_guard:
            n_rv = _validator_guard(pop_v, new_pop_v, stats_v, "v2c", pop_k)
            n_rc = _validator_guard(pop_c, new_pop_c, stats_c, "c2v", pop_k)
            if on_progress is not None and (n_rv + n_rc) > 0:
                on_progress("validator_guard", n_rv + n_rc, 2 * n, 0.0)
        return new_pop_v, new_pop_c, stats_v, stats_c

    # ---- Python reference path (serial, slow).  Used to generate
    # golden fixtures; cpp path is asserted structurally equal to this.
    new_pop_v = []
    new_pop_c = []
    stats_v = []
    stats_c = []
    for i in range(n):
        st = DCEStats(side="v2c", size_before=program_length(pop_v[i]))
        red = behavioral_reduce_bp(
            pop_v[i], "v2c",
            peer_prog=pop_c[int(perm_v[i])], log_constants=pop_k[i],
            par=par, rx_llrs=rx_llrs, max_iter=max_iter,
            max_passes=max_passes,
            max_decode_evals=(None if max_decode_evals < 0
                              else max_decode_evals),
            decimals=decimals, stats=st, use_cpp=False,
        )
        new_pop_v.append(red)
        stats_v.append(st)
        if on_progress is not None:
            on_progress("v2c", i + 1, 2 * n, 0.0)
    for i in range(n):
        st = DCEStats(side="c2v", size_before=program_length(pop_c[i]))
        red = behavioral_reduce_bp(
            pop_c[i], "c2v",
            peer_prog=pop_v[int(perm_c[i])], log_constants=pop_k[i],
            par=par, rx_llrs=rx_llrs, max_iter=max_iter,
            max_passes=max_passes,
            max_decode_evals=(None if max_decode_evals < 0
                              else max_decode_evals),
            decimals=decimals, stats=st, use_cpp=False,
        )
        new_pop_c.append(red)
        stats_c.append(st)
        if on_progress is not None:
            on_progress("c2v", n + i + 1, 2 * n, 0.0)
    if validator_guard:
        n_rv = _validator_guard(pop_v, new_pop_v, stats_v, "v2c", pop_k)
        n_rc = _validator_guard(pop_c, new_pop_c, stats_c, "c2v", pop_k)
        if on_progress is not None and (n_rv + n_rc) > 0:
            on_progress("validator_guard", n_rv + n_rc, 2 * n, 0.0)
    return new_pop_v, new_pop_c, stats_v, stats_c
