"""Audit gen0 post-DCE programs for output collapse.

Loads individuals.jsonl from the last evo run, runs every post-DCE
v2c/c2v program through both Python and C++ VMs on 20 random
(incoming, L_v) probes, and reports:
  - how many programs produce only 0 / NaN / constant (= "collapsed")
  - how many produce finite + varying output (= "alive")
  - Py vs C++ numeric disagreement count
"""
from __future__ import annotations
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cpp_seeder"))

import numpy as np
import pushgp_cpp_seeder as M  # type: ignore[import]

from pushgp.genome import list_to_program, program_to_list
from pushgp.validators import _make_vm, _run

RUN = "run_20260518_161605"
PATH = os.path.join(ROOT, "results", "logged_evolution", RUN, "individuals.jsonl")

def probes(rng, deg=8, n=20):
    return [(rng.uniform(-3, 3, size=deg-1), float(rng.uniform(-2, 2))) for _ in range(n)]

def classify(outs):
    """outs: list of (float|None)."""
    finite = [o for o in outs if o is not None and np.isfinite(o)]
    nones = sum(1 for o in outs if o is None)
    if not finite: return "fault-all", nones, 0
    if max(finite) == min(finite):
        if abs(finite[0]) < 1e-12: return "zero-const", nones, len(finite)
        return f"const({finite[0]:.4g})", nones, len(finite)
    return "alive", nones, len(finite)

def main():
    rows = [json.loads(l) for l in open(PATH, "r", encoding="utf-8") if l.strip()]
    # individuals records "side" + "program" (list-of-dict) post-DCE
    print(f"Loaded {len(rows)} individuals from {RUN}")
    # show what keys we have
    if rows: print(f"Keys: {sorted(rows[0].keys())[:20]}")

    rng = np.random.default_rng(0)
    probes_ = probes(rng)

    stats = {"v2c": {}, "c2v": {}}
    mismatch = 0
    sizes = {"v2c": [], "c2v": []}
    for r in rows:
        for side in ("v2c", "c2v"):
            prog_list = r.get(side)
            if not prog_list: continue
            sizes[side].append(len(prog_list))
            prog = list_to_program(prog_list)
            h = M.build_program(prog_list)
            evo = np.ones(8, dtype=np.float64)
            py_outs, cpp_outs = [], []
            for inc, L_v in probes_:
                vm = _make_vm(inc, channel_llr=L_v, has_channel_llr=(side=="v2c"), evo_consts=evo)
                py_outs.append(_run(prog, vm, side))
                o = M.run_program(h, side, evo, inc, L_v if side=="v2c" else 0.0, 8, 0)
                cpp_outs.append(o)
            cls_py, _, _ = classify(py_outs)
            stats[side][cls_py] = stats[side].get(cls_py, 0) + 1
            for a, b in zip(py_outs, cpp_outs):
                if (a is None) != (b is None): mismatch += 1
                elif a is not None and b is not None and abs(a - b) > 1e-6: mismatch += 1

    for side in ("v2c", "c2v"):
        total = sum(stats[side].values())
        sz = sizes[side]
        print(f"\n{side} buckets ({total} progs, post-DCE size mean={np.mean(sz):.1f} min={min(sz)} max={max(sz)}):")
        for k, v in sorted(stats[side].items(), key=lambda x: -x[1]):
            print(f"  {k:14s}  {v}")
    print(f"\nPy<->Cpp numeric disagreements (out of {len(rows)*len(probes_)*2} probes): {mismatch}")

if __name__ == "__main__":
    main()
