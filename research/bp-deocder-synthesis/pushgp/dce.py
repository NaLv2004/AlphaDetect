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

__all__ = [
    "behavioral_reduce",
    "behavioral_reduce_bp",
    "reduce_seeded_population",
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
    # Fast path: C++ port (pushgp_cpp_dce.reduce_bp).  T7 gate proves
    # structural equality to the Python loop below, so the result is
    # bit-identical at `decimals` precision while running x30+ faster.
    # ------------------------------------------------------------------
    if use_cpp:
        try:
            import pushgp_cpp_dce as _cdce  # noqa: F401
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
        except ImportError:
            pass  # cpp module not built; fall back to python loop
        except Exception:
            # Any cpp-side error: fall back to python loop (slow but
            # safe).  Tests would catch a true divergence.
            pass

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
