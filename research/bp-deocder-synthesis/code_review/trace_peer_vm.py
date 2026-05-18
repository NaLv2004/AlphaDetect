"""Per-instruction VM tracer for the size-0 peer programs.

Loads `_dce_dump.{tag}.pkl`, picks the V-side size-0 individuals, fetches
the paired peer C2V program, runs it under the same C2V seeding the
adapter uses, and prints the stack tops after EVERY instruction for two
parallel runs: seeder-default evo (all 1.0) vs. genome evo (10**log).

Same input vector, same incoming vector, same deg/it.  We just change
`ctx_evo_constants`.  This lets us see exactly which instruction's
behaviour diverges and how the divergence propagates to the final
output."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pushgp.program import Instruction  # noqa: E402
from pushgp.vm import VM  # noqa: E402
from pushgp.genome import list_to_program  # noqa: E402
from pushgp_ldpc.adapter import _seed_c2v  # noqa: E402


# --------------------------------------------------------------- formatting
def _fmt_floats(xs: List[float], k: int = 5) -> str:
    tail = xs[-k:] if len(xs) > k else xs
    body = ", ".join(f"{v:+.4g}" for v in tail)
    head = f"[{len(xs)}] " if len(xs) > k else f"[{len(xs)}] "
    return f"{head}top→ {body}"


def _fmt_fvecs(vs: List[np.ndarray], k: int = 2) -> str:
    tail = vs[-k:] if len(vs) > k else vs
    parts = []
    for v in tail:
        v = np.asarray(v)
        if v.size == 0:
            parts.append("[]")
        elif v.size <= 4:
            parts.append("[" + ", ".join(f"{x:+.3g}" for x in v) + "]")
        else:
            parts.append(f"[len={v.size}, head={v[0]:+.3g},{v[1]:+.3g},...]")
    body = " | ".join(parts)
    return f"[{len(vs)}] top→ {body}"


def _snapshot(state) -> dict:
    return {
        "F": list(state.floats._data),
        "I": list(state.ints._data),
        "B": list(state.bools._data),
        "FV": [np.asarray(v).copy() for v in state.fvecs._data],
        "IV": [np.asarray(v).copy() for v in state.ivecs._data],
        "fault": state.fault,
        "fault_reason": state.fault_reason,
    }


def _diff_line(label: str, before: dict, after: dict) -> str:
    parts = []
    if before["F"] != after["F"]:
        parts.append(f"F: {_fmt_floats(before['F'])}  →  {_fmt_floats(after['F'])}")
    if before["I"] != after["I"]:
        parts.append(f"I: top={before['I'][-3:]} → {after['I'][-3:]}")
    if [v.tolist() for v in before["FV"]] != [v.tolist() for v in after["FV"]]:
        parts.append(f"FV: {_fmt_fvecs(before['FV'])}  →  {_fmt_fvecs(after['FV'])}")
    if after["fault"] and not before["fault"]:
        parts.append(f"!! FAULT: {after['fault_reason']}")
    return f"  {label}\n     " + "\n     ".join(parts) if parts else f"  {label}\n     (no stack change)"


# --------------------------------------------------------------- tracer
def trace_program(prog: List[Instruction], evo: np.ndarray, incoming: np.ndarray,
                  deg: int, it: int, max_iter: int, label: str) -> List[dict]:
    """Run prog under C2V seeding, capturing a snapshot AFTER each top-level
    instruction.  Nested blocks (DoRange, DoTimes, While) execute atomically
    from the tracer's POV; we just record the cumulative effect."""
    vm = VM(step_max=20_000, flop_max=500_000, recur_max=64)
    _seed_c2v(vm, incoming, deg, it, max_iter, evo)

    snapshots = [("<initial>", _snapshot(vm.state))]
    # Drive instructions one at a time at the top level.
    for idx, ins in enumerate(prog):
        if vm.state.fault:
            snapshots.append((f"#{idx} {ins.name} (skipped: fault)", _snapshot(vm.state)))
            continue
        try:
            vm._step(ins)
        except Exception as exc:
            vm.state.fault = True
            vm.state.fault_reason = f"trace exc: {type(exc).__name__}: {exc}"
        snapshots.append((f"#{idx} {ins.name}", _snapshot(vm.state)))

    final = vm.run.__self__ if False else None  # noop
    top = vm.state.floats.peek()
    if top is None:
        out = None
    else:
        try:
            v = float(top)
            out = v if np.isfinite(v) else None
        except Exception:
            out = None
    print(f"[{label}] FINAL out = {out!r}  (fault={vm.state.fault}, reason={vm.state.fault_reason!r})")
    return snapshots


# --------------------------------------------------------------- driver
def main() -> None:
    here = Path(__file__).resolve().parent
    candidates = sorted(here.glob("_dce_dump.*.pkl"))
    if not candidates:
        print("No _dce_dump.*.pkl files found in", here)
        return

    for pkl_path in candidates:
        print(f"\n{'='*78}\nDUMP: {pkl_path.name}")
        with open(pkl_path, "rb") as fh:
            d = pickle.load(fh)

        pre_v = d["pre_pop_v"]
        pre_c = d["pre_pop_c"]
        post_v = d["post_pop_v"]
        perm_v = d["perm_v"]
        pop_k = d["pop_k"]
        max_iter = d["max_iter"]

        # First size-0 V-side individual
        zero_idx = [i for i, p in enumerate(post_v) if len(p) == 0]
        if not zero_idx:
            print("  no size-0 V individuals in this dump")
            continue
        i = zero_idx[0]
        peer_raw = pre_c[perm_v[i]]
        peer = list_to_program(peer_raw) if peer_raw and isinstance(peer_raw[0], dict) else peer_raw
        log_consts = pop_k[i]
        deg = 8  # seeder default
        evo_seed = np.ones(8, dtype=np.float64)
        evo_genome = (10.0 ** np.asarray(log_consts)).astype(np.float64)

        print(f"  V #{i} (pre size {len(pre_v[i])}) paired with C #{perm_v[i]} (size {len(peer)})")
        print(f"  log_consts = {np.asarray(log_consts).tolist()}")
        print(f"  evo_seed   = {evo_seed.tolist()}")
        print(f"  evo_genome = {[float(x) for x in evo_genome]}")

        # Pick a single incoming vector that elicits non-trivial outputs
        rng = np.random.default_rng(20251119)
        incoming = rng.uniform(-3.0, 3.0, size=deg - 1).astype(np.float64)
        print(f"  incoming   = {incoming.tolist()}")

        snap_seed = trace_program(peer, evo_seed, incoming, deg, 0, max_iter,
                                  label="evo=SEED")
        snap_gen  = trace_program(peer, evo_genome, incoming, deg, 0, max_iter,
                                  label="evo=GENOME")

        print("\n  Per-instruction side-by-side (top of float stack):")
        n = min(len(snap_seed), len(snap_gen))
        for k in range(n):
            lab = snap_seed[k][0]
            s = snap_seed[k][1]; g = snap_gen[k][1]
            f_s = s['F'][-1] if s['F'] else None
            f_g = g['F'][-1] if g['F'] else None
            fv_s = [len(v) for v in s['FV'][-2:]]
            fv_g = [len(v) for v in g['FV'][-2:]]
            mark = "  "
            if (f_s != f_g) or (s['fault'] != g['fault']):
                mark = "**"
            print(f"  {mark} {lab:<40}  SEED  F.top={f_s!r:>22} FV.lens={fv_s}  "
                  f"fault={'Y' if s['fault'] else '.'}")
            print(f"  {mark} {'':<40}  GEN   F.top={f_g!r:>22} FV.lens={fv_g}  "
                  f"fault={'Y' if g['fault'] else '.'}")


if __name__ == "__main__":
    main()
