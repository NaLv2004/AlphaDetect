"""Diagnose why constant-output programs slip through.

For each post-DCE program classified 'const' or 'fault-all', run the
Python validator on the POST-DCE program. If validator rejects it ->
DCE breaks invariants (need post-DCE re-validation). If validator
accepts it -> validator bug (probe set is insufficient).
"""
from __future__ import annotations
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from pushgp.genome import list_to_program
from pushgp.validators import validate_v2c, validate_c2v, _make_vm, _run

RUN = "post_refactor_pop20_g1_20260518-155701"
PATH = os.path.join(ROOT, "results", "logged_evolution", RUN, "individuals.jsonl")

def probe_outputs(prog, side, n=20):
    rng = np.random.default_rng(0)
    outs = []
    for _ in range(n):
        inc = rng.uniform(-3, 3, size=7)
        L_v = float(rng.uniform(-2, 2))
        vm = _make_vm(inc, channel_llr=L_v, has_channel_llr=(side=="v2c"),
                      evo_consts=np.ones(8))
        outs.append(_run(prog, vm, side))
    return outs

def is_const_or_fault(outs):
    finite = [o for o in outs if o is not None and np.isfinite(o)]
    if not finite: return ("fault-all", None)
    if max(finite) == min(finite): return ("const", finite[0])
    return ("alive", None)

rows = [json.loads(l) for l in open(PATH, "r", encoding="utf-8") if l.strip()]
print(f"{'ind':<4} {'side':<4} {'class':<10} {'val_ok':<7} {'val_why':<45} {'size':<4}")
print("-" * 80)
n_dce_bug = 0
n_val_bug = 0
n_ok = 0
for i, r in enumerate(rows):
    for side in ("v2c", "c2v"):
        prog_list = r.get(side)
        if not prog_list: continue
        prog = list_to_program(prog_list)
        cls, val = is_const_or_fault(probe_outputs(prog, side))
        if cls == "alive": continue
        # re-validate post-DCE program
        validator = validate_v2c if side == "v2c" else validate_c2v
        ok, why = validator(prog, rng=np.random.default_rng(42))
        if cls == "fault-all":
            tag = "DCE-bug" if ok else "ok-rejected"
        else:  # const
            tag = "VAL-BUG" if ok else "DCE-leak"
        if ok and cls in ("const", "fault-all"): n_val_bug += 1
        elif not ok and cls in ("const", "fault-all"): n_dce_bug += 1
        else: n_ok += 1
        print(f"{i:<4} {side:<4} {cls:<10} {str(ok):<7} {why[:42]:<45} {len(prog_list):<4} [{tag}]")

print("-" * 80)
print(f"VAL-BUG (validator accepts a broken post-DCE prog): {n_val_bug}")
print(f"DCE-leak (validator rejects post-DCE prog -- DCE-caused regression): {n_dce_bug}")
