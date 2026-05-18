"""Like audit_postdce_outputs.py but probes match validator range [-10,10]."""
from __future__ import annotations
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cpp_seeder"))

import numpy as np
import pushgp_cpp_seeder as M  # type: ignore[import]

from pushgp.genome import list_to_program
from pushgp.validators import _make_vm, _run

RUN = "run_20260518_161605"
PATH = os.path.join(ROOT, "results", "logged_evolution", RUN, "individuals.jsonl")

def probes(rng, deg=8, n=60, lo=-10.0, hi=10.0):
    return [(rng.uniform(lo, hi, size=deg-1), float(rng.uniform(lo, hi))) for _ in range(n)]

def classify(outs):
    finite = [o for o in outs if o is not None and np.isfinite(o)]
    if not finite: return "fault-all"
    if max(finite) == min(finite):
        return "zero-const" if abs(finite[0]) < 1e-12 else f"const({finite[0]:.4g})"
    spread = max(finite) - min(finite)
    return f"alive(spread={spread:.3g})"

def main():
    rows = [json.loads(l) for l in open(PATH, "r", encoding="utf-8") if l.strip()]
    print(f"Loaded {len(rows)} individuals from {RUN} | probes [-10,10] n=60")
    rng = np.random.default_rng(0)
    probes_ = probes(rng)
    stats = {"v2c": {}, "c2v": {}}
    for r in rows:
        log_consts = np.asarray(r.get("log_constants", [0.0]*8), dtype=np.float64)
        evo = (10.0 ** log_consts).astype(np.float64)
        for side in ("v2c", "c2v"):
            prog_list = r.get(side)
            if not prog_list: continue
            prog = list_to_program(prog_list)
            py_outs = []
            for inc, L_v in probes_:
                vm = _make_vm(inc, channel_llr=L_v, has_channel_llr=(side=="v2c"), evo_consts=evo)
                py_outs.append(_run(prog, vm, side))
            # bucket by alive vs collapse
            cls = classify(py_outs)
            key = "alive" if cls.startswith("alive") else cls
            stats[side][key] = stats[side].get(key, 0) + 1

    for side in ("v2c", "c2v"):
        total = sum(stats[side].values())
        good = stats[side].get("alive", 0)
        bad = total - good
        print(f"\n{side}: total={total} alive={good} collapsed={bad}")
        for k, v in sorted(stats[side].items(), key=lambda x: -x[1]):
            print(f"  {k:18s}  {v}")

if __name__ == "__main__":
    main()
