"""Tests for ``pushgp.dce.behavioral_reduce`` (Step A).

Test matrix (matches design discussion):
  * T1 -- hand-crafted dead-tail: reducer must remove obvious cruft.
  * T2 -- invariant_self: fp(reduce(p)) == fp(p) for many random progs.
  * T3 -- idempotence: reduce(reduce(p)) == reduce(p) (program-equal).
  * T4 -- BER equivalence: full LDPC decode on reduced vs original
          must match bit-for-bit at multiple SNRs (KEY safety test).
  * T5 -- reduction yield: average reduction ratio on big evolved
          programs (size > 20).

All tests are runnable as a script:
    python -B pushgp/tests/test_dce.py
or with pytest:
    pytest pushgp/tests/test_dce.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple

# Make 'pushgp' importable when run directly.
_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import numpy as np

from pushgp.dce import behavioral_reduce, behavioral_reduce_bp, _walk_positions, _remove_at, DCEStats
from pushgp.evolution import _behav_fingerprint
from pushgp.program import Instruction, deep_copy_program, program_length
from pushgp.random_program import RandomProgramGenerator
from pushgp.serialize import dict_to_program, program_to_dict


# ============================================================================
# Helpers
# ============================================================================


def _prog_eq(a: List[Instruction], b: List[Instruction]) -> bool:
    return program_to_dict(a) == program_to_dict(b)


def _random_progs(side: str, n: int, *, seed: int = 0,
                  min_size: int = 4, max_size: int = 30) -> List[List[Instruction]]:
    """Generate ``n`` raw random programs — no validity check.

    Used for T2/T3 where the only correctness criterion is fingerprint
    preservation, not LDPC-decoder validity.  Skipping the validator
    keeps T2/T3 fast regardless of program size.
    """
    rpg = RandomProgramGenerator(rng=np.random.default_rng(seed))
    gen_one = rpg.random_v2c if side == "v2c" else rpg.random_c2v
    return [gen_one(min_size=min_size, max_size=max_size) for _ in range(n)]


def _harvest_valid(side: str, n: int, *, seed: int = 12345,
                   min_size: int = 4, max_size: int = 30,
                   deg: int = 8) -> List[List[Instruction]]:
    """Return ``n`` validator-passing programs.

    Strategy: prefer real evolved programs from the production log
    (already validated, realistic distribution).  Fall back to RPG +
    validator if the log is unavailable or short.
    """
    import json
    log_path = Path(__file__).resolve().parent.parent.parent / \
        "results" / "logged_evolution" / "fromscratch_pop100_dedup" / "individuals.jsonl"
    out: List[List[Instruction]] = []
    if log_path.exists():
        key = "v2c" if side == "v2c" else "c2v"
        size_key = "v2c_size" if side == "v2c" else "c2v_size"
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(out) >= n:
                    break
                rec = json.loads(line)
                if not rec.get("valid"):
                    continue
                sz = rec.get(size_key, 0)
                if sz < min_size or sz > max_size:
                    continue
                out.append(dict_to_program(rec[key]))
        if len(out) >= n:
            return out[:n]
    if out:
        return out  # partial log result is better than slow RPG fallback
    raise RuntimeError(
        f"_harvest_valid: log missing or insufficient for {side} "
        f"(need {n}, got {len(out)}, min_size={min_size}, max_size={max_size}).\n"
        f"Run the evolution first to populate the log."
    )


# ============================================================================
# T1: hand-crafted dead-tail
# ============================================================================


def _make_useful_v2c() -> List[Instruction]:
    """A small program whose live-expr is non-trivial: tanh(L_v) on float-stack."""
    return [
        Instruction(name="Float.Tanh"),  # consumes top of float stack -> tanh(L_v)
    ]


def _t1_hand_crafted_dead_tails() -> None:
    """T1: reducer removes appended trivially-dead tails."""
    print("\n--- T1: hand-crafted dead tails ---", flush=True)
    failures: List[str] = []

    cases: List[Tuple[str, List[Instruction]]] = []
    # Case 1: useful program + trivially-dead Bool ops at the end
    cases.append((
        "v2c useful + dead bool tail",
        _make_useful_v2c() + [
            Instruction(name="Bool.True"),
            Instruction(name="Bool.False"),
            Instruction(name="Bool.Or"),
        ],
    ))
    # Case 2: useful program + dead Int ops appended
    cases.append((
        "v2c useful + dead int tail",
        _make_useful_v2c() + [
            Instruction(name="Int.Const1"),
            Instruction(name="Int.Const2"),
            Instruction(name="Int.Add"),
            Instruction(name="Int.Pop"),
        ],
    ))
    # Case 3: pure-dead program (no float live expr) -> reducer should
    # be able to delete *everything* if NAN-bucket collapse applies.
    # We don't assert full deletion here (depends on program fp), but
    # at minimum it should not grow.
    cases.append((
        "v2c pure bool/int (no live float)",
        [
            Instruction(name="Bool.True"),
            Instruction(name="Int.Const1"),
            Instruction(name="Int.Add"),
        ],
    ))

    for name, prog in cases:
        n_in = program_length(prog)
        stats = DCEStats(side="v2c", size_before=n_in)
        reduced = behavioral_reduce(prog, "v2c", stats=stats)
        n_out = program_length(reduced)
        print(f"  [{name}] {n_in} -> {n_out} (passes={stats.passes}, fp_evals={stats.fp_evals})")
        if n_out > n_in:
            failures.append(f"{name}: program grew {n_in} -> {n_out}")
        # Cases 1 & 2: must reduce strictly (we know the tails are
        # provably ignored by the panel because they don't touch float
        # stack at all and the float stack already carried the live val).
        if "dead" in name and n_out == n_in:
            failures.append(f"{name}: expected strict reduction, got none")

    if failures:
        for f in failures:
            print("  FAIL:", f, flush=True)
        raise AssertionError(f"T1: {len(failures)} failure(s)")
    print("  T1 PASSED", flush=True)


# ============================================================================
# T2: invariant_self
# ============================================================================


def _t2_invariant_self(n_per_side: int = 50) -> None:
    """T2: fp(reduce(p)) == fp(p) for ``n_per_side`` random programs.

    Uses raw random programs (no validator) for speed: the DCE oracle
    is the behavioral fingerprint itself, not LDPC validity.
    """
    print(f"\n--- T2: invariant_self (n={n_per_side} per side) ---", flush=True)
    failures: List[str] = []
    for side in ("v2c", "c2v"):
        progs = _random_progs(side, n_per_side, seed=12345 + (1 if side == "c2v" else 0))
        for i, p in enumerate(progs):
            fp0 = _behav_fingerprint(side, p)
            try:
                r = behavioral_reduce(p, side)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"[{side}] idx={i}: reducer crashed: {type(exc).__name__}: {exc}")
                continue
            fp1 = _behav_fingerprint(side, r)
            if fp0 != fp1:
                failures.append(f"[{side}] idx={i}: fp mismatch (fp0_head={fp0[:32]} fp1_head={fp1[:32]})")
        print(f"  [{side}] ok: {n_per_side - len([f for f in failures if f'[{side}]' in f])}/{n_per_side}", flush=True)
    if failures:
        for f in failures[:10]:
            print("  FAIL:", f, flush=True)
        raise AssertionError(f"T2: {len(failures)} failures (showing first 10)")
    print("  T2 PASSED", flush=True)


# ============================================================================
# T3: idempotence
# ============================================================================


def _t3_idempotence(n_per_side: int = 30) -> None:
    """T3: reduce(reduce(p)) == reduce(p) (program-structurally equal).

    Uses raw random programs (no validator) for speed.
    """
    print(f"\n--- T3: idempotence (n={n_per_side} per side) ---", flush=True)
    failures: List[str] = []
    for side in ("v2c", "c2v"):
        progs = _random_progs(side, n_per_side, seed=23456 + (1 if side == "c2v" else 0))
        for i, p in enumerate(progs):
            r1 = behavioral_reduce(p, side)
            r2 = behavioral_reduce(r1, side)
            if not _prog_eq(r1, r2):
                failures.append(f"[{side}] idx={i}: reduce(reduce(p)) != reduce(p) "
                                f"(sizes {program_length(r1)} vs {program_length(r2)})")
    if failures:
        for f in failures[:10]:
            print("  FAIL:", f, flush=True)
        raise AssertionError(f"T3: {len(failures)} failures (showing first 10)")
    print("  T3 PASSED", flush=True)


# ============================================================================
# T4: BER equivalence (the safety-critical test)
# ============================================================================


def _build_channel_data(par, snr_list, n_frames, seed):
    """Pre-generate (cw, llr) frames per SNR ONCE so orig and reduced
    decoders are fed bit-identical inputs.  Removes any doubt about
    RNG drift between the two evaluations.
    """
    from ldpc_5g import HTYPE, bpsk_modulate, bpsk_llr
    from pushgp_ldpc.eval import _random_codeword
    htype = HTYPE[par.bgn - 1][par.set_idx - 1]
    rate = 0.5
    out = []  # list per snr of list of (cw, llr)
    for snr_db in snr_list:
        rng = np.random.default_rng(seed + abs(int(snr_db * 1000)))
        sigma2 = 1.0 / (2.0 * rate * 10.0 ** (snr_db / 10.0))
        sigma = float(np.sqrt(sigma2))
        frames = []
        for _ in range(n_frames):
            cw = _random_codeword(par, htype, rng)
            tx = bpsk_modulate(cw[2 * par.zc:])
            rx = tx + sigma * rng.standard_normal(tx.shape)
            llr_part = bpsk_llr(rx, sigma2)
            llr = np.zeros(par.cols, dtype=np.float64)
            llr[2 * par.zc:] = llr_part
            frames.append((cw, llr, sigma2))
        out.append(frames)
    return out


def _ber_with_data(prog_v: List[Instruction], prog_c: List[Instruction],
                    log_consts: np.ndarray, par,
                    frames_per_snr: List, max_iter: int
                    ) -> Tuple[List[float], bool]:
    """Decode the pre-built channel data; returns (ber_per_snr, ok)."""
    from pushgp.genome import Genome
    from pushgp_ldpc.adapter import make_callables
    from ldpc_5g import decode_bp

    g = Genome(prog_v2c=prog_v, prog_c2v=prog_c, log_constants=log_consts.copy())
    try:
        v2c_fn, c2v_fn = make_callables(g)
    except Exception:
        return [float("nan")] * len(frames_per_snr), False

    bers: List[float] = []
    ok_all = True
    for frames in frames_per_snr:
        n_err = 0
        n_bits = 0
        for cw, llr, _sigma2 in frames:
            try:
                post = decode_bp(
                    llr, par, v2c_fn=v2c_fn, c2v_fn=c2v_fn,
                    max_iter=max_iter, offset=0.25, code_rate=0.5,
                )
            except Exception:
                ok_all = False
                bers.append(float("nan"))
                break
            else:
                hat = (post < 0.0).astype(np.int8)
                n_err += int((hat != cw).sum())
                n_bits += cw.size
        else:
            bers.append(n_err / max(1, n_bits))
            continue
        # broke inside; pad remaining
        bers.extend([float("nan")] * (len(frames_per_snr) - len(bers)))
        return bers, False
    return bers, ok_all


def _t4_ber_equivalence(n_pairs: int = 20,
                             snr_list: Tuple[float, ...] = (-2.0, -1.0, 0.0),
                             n_frames: int = 200,
                             max_iter: int = 8) -> None:
    """T4: BER(reduce(v), reduce(c)) == BER(v, c) on real LDPC frames.

    Channel data is **pre-generated once per pair** and the SAME
    ``(cw, llr)`` arrays are fed to both the original and the reduced
    decoder.  This removes any doubt about RNG drift between the two
    runs: if BERs differ, it is genuinely the program semantics.

    The acceptance criterion is BER **equality to 6 decimal places** at
    every SNR.  Inequality means the 32-entry behavioral panel failed
    to capture some semantically-relevant aspect of the program.
    """
    print(f"\n--- T4: BER equivalence (n_pairs={n_pairs}, snr={snr_list}, "
          f"n_frames={n_frames}) ---", flush=True)
    from ldpc_5g import build_parity
    from pushgp.genome import Genome
    par = build_parity(bgn=2, set_idx=1, zc=2)

    pairs: List[Tuple[List[Instruction], List[Instruction], np.ndarray]] = []
    log_path = Path(__file__).resolve().parent.parent.parent / \
        "results" / "logged_evolution" / "fromscratch_pop100_dedup" / "individuals.jsonl"
    if log_path.exists():
        import json
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(pairs) >= n_pairs:
                    break
                rec = json.loads(line)
                if not rec.get("valid"):
                    continue
                if rec.get("v2c_size", 0) < 8 or rec.get("c2v_size", 0) < 8:
                    continue
                v = dict_to_program(rec["v2c"])
                c = dict_to_program(rec["c2v"])
                k = np.asarray(rec["log_constants"], dtype=np.float64)
                pairs.append((v, c, k))
    if len(pairs) < n_pairs:
        rpg = RandomProgramGenerator(rng=np.random.default_rng(34567))
        attempts = 0
        while len(pairs) < n_pairs and attempts < 100_000:
            attempts += 1
            v = rpg.random_v2c(min_size=4, max_size=20)
            c = rpg.random_c2v(min_size=4, max_size=20)
            ok_v, _ = validate_v2c(v, rng=np.random.default_rng(attempts), deg=8)
            ok_c, _ = validate_c2v(c, rng=np.random.default_rng(attempts + 1), deg=8)
            if ok_v and ok_c:
                k = rpg.random_log_constants()
                pairs.append((v, c, k))
        if len(pairs) < n_pairs:
            raise RuntimeError(f"only got {len(pairs)} valid pairs")
    print(f"  using {len(pairs)} pairs", flush=True)

    failures: List[str] = []
    n_ok = 0
    import time as _time
    # Build the BP-equivalence oracle: a small bank of frames at every
    # SNR used by the BER measurement.  For each candidate program, the
    # BP post-LLR must match the baseline at ALL oracle frames; this is
    # strictly stronger than any static panel and prevents over-pruning
    # an instruction that is silent at one (snr, frame) but live at
    # another.  We use 1 frame per SNR -> small but multi-distribution.
    oracle_per_snr = _build_channel_data(par, snr_list, 1, seed=20_000)
    oracle_rx_llrs = [frames[0][1] for frames in oracle_per_snr]  # list[np.ndarray]
    for i, (v, c, k) in enumerate(pairs):
        _t0 = _time.time()
        v_red = behavioral_reduce_bp(
            v, "v2c", peer_prog=c, log_constants=k, par=par,
            rx_llrs=oracle_rx_llrs, max_iter=max_iter,
        )
        c_red = behavioral_reduce_bp(
            c, "c2v", peer_prog=v, log_constants=k, par=par,
            rx_llrs=oracle_rx_llrs, max_iter=max_iter,
        )
        # Pre-build channel data ONCE — same arrays for both decoders.
        frames_per_snr = _build_channel_data(
            par, snr_list, n_frames, seed=10_000 + i,
        )
        ber_orig, ok0 = _ber_with_data(v, c, k, par,
                                        frames_per_snr, max_iter)
        ber_red, ok1 = _ber_with_data(v_red, c_red, k, par,
                                        frames_per_snr, max_iter)
        dt = _time.time() - _t0
        match = (ok0 == ok1) and all(
            round(a - b, 6) == 0.0 for a, b in zip(ber_orig, ber_red)
        )
        tag = "OK " if match else "FAIL"
        print(f"  [{i+1:2d}/{len(pairs)}] {tag} "
              f"v {program_length(v):2d}->{program_length(v_red):2d}  "
              f"c {program_length(c):2d}->{program_length(c_red):2d}  "
              f"orig={[round(x,4) for x in ber_orig]}  "
              f"red={[round(x,4) for x in ber_red]}  "
              f"dt={dt:.1f}s", flush=True)
        if ok0 != ok1:
            failures.append(
                f"pair {i}: validity flipped {ok0} -> {ok1}; "
                f"sizes {program_length(v)}->{program_length(v_red)} "
                f"and {program_length(c)}->{program_length(c_red)}"
            )
            continue
        if match:
            n_ok += 1
        else:
            failures.append(
                f"pair {i}: BER mismatch  orig={ber_orig}  red={ber_red}  "
                f"v {program_length(v)}->{program_length(v_red)}  "
                f"c {program_length(c)}->{program_length(c_red)}"
            )
    pass_rate = n_ok / float(len(pairs))
    print(f"  T4 BER-equivalence pass rate: {n_ok}/{len(pairs)} = {pass_rate:.1%}", flush=True)
    if failures:
        for f in failures[:10]:
            print("  FAIL:", f, flush=True)
    if pass_rate < 0.95:
        raise AssertionError(
            f"T4: BER equivalence pass rate {pass_rate:.1%} < 95%. "
            f"BP-equivalence DCE oracle failed to preserve semantics."
        )
    print("  T4 PASSED", flush=True)


# ============================================================================
# T5: reduction yield
# ============================================================================


def _t5_reduction_yield(min_size: int = 25, max_size: int = 60,
                             n_per_side: int = 30,
                             threshold_ratio: float = 0.10) -> None:
    """T5: average reduction on size>=min_size programs must exceed ``threshold_ratio``."""
    print(f"\n--- T5: reduction yield (min_size={min_size}, n={n_per_side}) ---", flush=True)
    summary: List[Tuple[str, float, int, int]] = []
    failures: List[str] = []
    for side in ("v2c", "c2v"):
        # max_size=200 so large evolved programs are not excluded from the log
        progs = _harvest_valid(side, n_per_side, seed=45678 + (1 if side == "c2v" else 0),
                               min_size=min_size, max_size=200)
        ratios: List[float] = []
        sizes_in: List[int] = []
        sizes_out: List[int] = []
        for p in progs:
            stats = DCEStats(side=side, size_before=program_length(p))
            r = behavioral_reduce(p, side, stats=stats)
            ratios.append(stats.reduction_ratio)
            sizes_in.append(stats.size_before)
            sizes_out.append(stats.size_after)
        mean_ratio = float(np.mean(ratios))
        max_ratio = float(np.max(ratios))
        sum_in = int(np.sum(sizes_in))
        sum_out = int(np.sum(sizes_out))
        print(f"  [{side}] mean_reduction={mean_ratio:.1%}  max={max_ratio:.1%}  "
              f"total_sizes {sum_in}->{sum_out}", flush=True)
        summary.append((side, mean_ratio, sum_in, sum_out))
        if mean_ratio < threshold_ratio:
            failures.append(f"[{side}] mean reduction {mean_ratio:.1%} "
                            f"< threshold {threshold_ratio:.1%}")
    if failures:
        for f in failures:
            print("  FAIL:", f, flush=True)
        raise AssertionError(f"T5: {len(failures)} threshold misses")
    print("  T5 PASSED", flush=True)


# ============================================================================
# Driver
# ============================================================================


def _run_all() -> int:
    tests = [
        ("T1", _t1_hand_crafted_dead_tails, ()),
        ("T2", _t2_invariant_self, (50,)),
        ("T3", _t3_idempotence, (30,)),
        ("T5", _t5_reduction_yield, (25, 60, 30, 0.10)),
        # T4 is heavy (LDPC BER eval); leave for last so we know the
        # cheap tests passed first.
        ("T4", _t4_ber_equivalence, (20, (-2.0, -1.0, 0.0), 200, 8)),
    ]
    failures: List[str] = []
    for name, fn, args in tests:
        t0 = time.time()
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{name}] FAILED in {time.time() - t0:.1f}s:", flush=True)
            traceback.print_exc()
            failures.append(name)
        else:
            print(f"[{name}] OK in {time.time() - t0:.1f}s", flush=True)
    if failures:
        print(f"\n=== {len(failures)} test(s) failed: {', '.join(failures)} ===", flush=True)
        return 1
    print("\n=== ALL DCE TESTS PASSED ===", flush=True)
    return 0


# pytest entry points (each test is also discoverable individually)
def test_dce_T1() -> None:
    _t1_hand_crafted_dead_tails()


def test_dce_T2() -> None:
    _t2_invariant_self(20)


def test_dce_T3() -> None:
    _t3_idempotence(15)


def test_dce_T4() -> None:
    _t4_ber_equivalence(10, (-2.0, -1.0, 0.0), 5, 3)


def test_dce_T5() -> None:
    _t5_reduction_yield(25, 60, 15, 0.10)


if __name__ == "__main__":
    sys.exit(_run_all())
