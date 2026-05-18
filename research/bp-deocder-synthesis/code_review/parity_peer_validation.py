"""Python<->C++ validator parity check on the dumped peer programs.

Loads `_dce_dump.gen0.pkl`, runs both Python `validate_v2c`/`validate_c2v`
and the C++ `validate_with_panels` (driven by the SAME panels generated
in Python so the comparison is deterministic), and asserts agreement on
(ok, reason-bucket) for every peer.

We compare "reason buckets" rather than exact strings because the two
implementations format error messages slightly differently.
"""
from __future__ import annotations

import os, sys, pickle, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp_seeder"))

import numpy as np
import pushgp_cpp_seeder as M  # type: ignore[import]

from pushgp.genome import list_to_program, program_to_list
from pushgp.validators import (
    DEFAULT_DEG, DEFAULT_NUM_CONFIGS, DEFAULT_NUM_PERMUTATIONS,
    DEFAULT_PERTURB_DELTA,
    _build_input_probes, _structured_perms,
)


BUCKETS = (
    "faulty / non-finite baseline",
    "output independent of L_v",
    "output independent of incoming",
    "faulty under permutation",
    "not permutation-invariant",
    "ok",
)


def bucket(reason: str) -> str:
    for b in BUCKETS:
        if b in reason:
            return b
    return reason


def make_panels(side: str, deg: int, num_configs: int, num_permutations: int,
                rng: np.random.Generator):
    """Build a list of panel dicts compatible with the new C++ binding.
    Panels are deterministic from `rng` and identical to what Python's
    validators would draw (same uniform calls, same order)."""
    panels = []
    for cfg in range(num_configs):
        L_v = float(rng.uniform(-2.0, 2.0)) if side == "v2c" else 0.0
        incoming = rng.uniform(-3.0, 3.0, size=deg - 1)
        perms = [p.astype(np.float64) for p in _structured_perms(incoming, num_permutations, rng)]
        panels.append({
            "L_v": L_v,
            "incoming": incoming.astype(np.float64),
            "permutations": perms,
        })
    return panels


def run_one(prog_list, side, peer_idx):
    rng = np.random.default_rng(peer_idx)
    # Python side uses its own RNG; for parity we don't need same RNG —
    # we just need to drive both with the SAME panels (built once here).
    deg = DEFAULT_DEG
    panels = make_panels(side, deg, DEFAULT_NUM_CONFIGS, DEFAULT_NUM_PERMUTATIONS, rng)
    evo = np.ones(8, dtype=np.float64)

    # ---- Python: run validator manually against these panels ----
    from pushgp.validators import _make_vm, _run
    py_ok, py_why = True, "ok"
    delta = DEFAULT_PERTURB_DELTA
    for cfg_i, P in enumerate(panels):
        tag = f"cfg{cfg_i}/evo0"
        vm = _make_vm(P["incoming"], channel_llr=P["L_v"],
                      has_channel_llr=(side == "v2c"), deg=deg,
                      iter_idx=cfg_i, evo_consts=evo)
        base = _run(prog_list, vm, side)
        if base is None:
            py_ok, py_why = False, f"{side} {tag}: faulty / non-finite baseline"; break
        if side == "v2c":
            l_ok = False
            for sign in (+1.0, -1.0):
                for s in (1.0, 5.0):
                    vm2 = _make_vm(P["incoming"], channel_llr=P["L_v"] + sign*s*delta,
                                   has_channel_llr=True, deg=deg, iter_idx=cfg_i,
                                   evo_consts=evo)
                    o = _run(prog_list, vm2, side)
                    if o is None: continue
                    if abs(o - base) >= 1e-9: l_ok = True; break
                if l_ok: break
            if not l_ok:
                py_ok, py_why = False, f"v2c {tag}: output independent of L_v"; break
        inc_ok = False
        for _lbl, p in _build_input_probes(P["incoming"], delta):
            vm3 = _make_vm(p, channel_llr=P["L_v"],
                           has_channel_llr=(side == "v2c"), deg=deg,
                           iter_idx=cfg_i, evo_consts=evo)
            o = _run(prog_list, vm3, side)
            if o is None: continue
            if abs(o - base) >= 1e-9: inc_ok = True; break
        if not inc_ok:
            py_ok, py_why = False, f"{side} {tag}: output independent of incoming"; break
        for perm in P["permutations"]:
            vm4 = _make_vm(perm, channel_llr=P["L_v"],
                           has_channel_llr=(side == "v2c"), deg=deg,
                           iter_idx=cfg_i, evo_consts=evo)
            o = _run(prog_list, vm4, side)
            if o is None:
                py_ok, py_why = False, f"{side} {tag}: faulty under permutation"; break
            if abs(o - base) > 1e-7:
                py_ok, py_why = False, f"{side} {tag}: not permutation-invariant"; break
        if not py_ok: break

    # ---- C++: same panels, same evo ----
    h = M.build_program(program_to_list(prog_list))
    cpp_ok, cpp_why = M.validate_with_panels(h, side, panels, evo, deg)
    return (py_ok, bucket(py_why)), (cpp_ok, bucket(cpp_why))


def main():
    with open(os.path.join(os.path.dirname(__file__), "_dce_dump.gen0.pkl"), "rb") as f:
        data = pickle.load(f)
    prev = [list_to_program(p) for p in data["pre_pop_v"]]
    prec = [list_to_program(p) for p in data["pre_pop_c"]]
    mismatches = 0
    for side, pop in (("v2c", prev), ("c2v", prec)):
        print(f"\n=== {side} ({len(pop)} peers) ===")
        for i, p in enumerate(pop):
            (po, pr), (co, cr) = run_one(p, side, i)
            agree = (po == co) and (pr == cr)
            mark = "OK " if agree else "MIS"
            if not agree: mismatches += 1
            print(f"  {mark} {side}#{i:2d}  py={po} ({pr})   cpp={co} ({cr})")
    print(f"\nTotal mismatches: {mismatches}")
    sys.exit(0 if mismatches == 0 else 1)


if __name__ == "__main__":
    main()
