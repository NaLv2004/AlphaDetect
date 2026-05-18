"""Test whether widening the validator's random-base range from
[-3, 3] to [-10, 10] would catch the step-response programs.

We re-classify the 20 individuals' post-DCE programs under:
  R3 : 20 random incomings uniform[-3, 3]     (current validator range)
  R10: 20 random incomings uniform[-10, 10]   (proposed wider range)

A program is 'caught as varied' if it produces >= 2 unique outputs
across the 20 probes.
"""
from __future__ import annotations
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from pushgp.genome import list_to_program
from pushgp.validators import _make_vm, _run

RUN = "post_refactor_pop20_g1_20260518-155701"
PATH = os.path.join(ROOT, "results", "logged_evolution", RUN, "individuals.jsonl")
rows = [json.loads(l) for l in open(PATH, "r", encoding="utf-8") if l.strip()]

def n_unique(prog, side, lo, hi, n=20, seed=0):
    rng = np.random.default_rng(seed)
    outs = []
    for _ in range(n):
        inc = rng.uniform(lo, hi, size=7)
        L_v = float(rng.uniform(-abs(lo)/2, abs(lo)/2)) if side == "v2c" else 0.0
        vm = _make_vm(inc, channel_llr=L_v, has_channel_llr=(side=="v2c"), evo_consts=np.ones(8))
        outs.append(_run(prog, vm, side))
    finite = [o for o in outs if o is not None and np.isfinite(o)]
    return len(set(round(o, 9) for o in finite)), len(finite), len(outs)

print(f"{'i':<3} {'side':<4} {'R3_u':<5} {'R10_u':<6} {'R3_finite':<10} {'R10_finite':<11} {'verdict'}")
print("-" * 70)
fixed_by_widening = 0
caught_under_3 = 0
caught_under_10 = 0
total = 0
for i, r in enumerate(rows):
    for side in ("v2c", "c2v"):
        pl = r.get(side)
        if not pl: continue
        total += 1
        prog = list_to_program(pl)
        u3, f3, _ = n_unique(prog, side, -3, 3, n=20, seed=0)
        u10, f10, _ = n_unique(prog, side, -10, 10, n=20, seed=0)
        if u3 >= 2: caught_under_3 += 1
        if u10 >= 2: caught_under_10 += 1
        verdict = ""
        if u3 < 2 and u10 >= 2: verdict = "FIXED by widening"; fixed_by_widening += 1
        elif u3 < 2 and u10 < 2: verdict = "still collapsed"
        elif u3 >= 2: verdict = "already varied"
        print(f"{i:<3} {side:<4} {u3:<5} {u10:<6} {f3:<10} {f10:<11} {verdict}")
print("-" * 70)
print(f"Total programs: {total}")
print(f"Varied under [-3,3]:  {caught_under_3}/{total}")
print(f"Varied under [-10,10]: {caught_under_10}/{total}")
print(f"NEWLY caught by widening to [-10,10]: {fixed_by_widening}")
