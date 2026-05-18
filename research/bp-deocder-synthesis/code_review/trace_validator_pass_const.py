"""Trace WHY validate_c2v passes a globally-constant program.

Picks idx=10 (constant 2.718) from the latest run and walks validate_c2v
step by step, printing base/perturbation outputs and which check passes.
"""
from __future__ import annotations
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from pushgp.genome import list_to_program
from pushgp.validators import (
    _make_vm, _run, _build_input_probes, _structured_perms, _sample_evo_panels,
    INCOMING_SAMPLE_LO, INCOMING_SAMPLE_HI,
    DEFAULT_NUM_CONFIGS, DEFAULT_NUM_PERMUTATIONS, DEFAULT_NUM_EVO_PANELS,
    DEFAULT_DEG, DEFAULT_PERTURB_DELTA,
    EPS_DEPENDENCY, EPS_INVARIANCE,
)

RUN = "run_20260518_161605"
PATH = os.path.join(ROOT, "results", "logged_evolution", RUN, "individuals.jsonl")

print(f"EPS_DEPENDENCY={EPS_DEPENDENCY} EPS_INVARIANCE={EPS_INVARIANCE} "
      f"PERTURB_DELTA={DEFAULT_PERTURB_DELTA}")
print(f"INCOMING_RANGE=[{INCOMING_SAMPLE_LO},{INCOMING_SAMPLE_HI}]  "
      f"NUM_CONFIGS={DEFAULT_NUM_CONFIGS} NUM_PERMS={DEFAULT_NUM_PERMUTATIONS} "
      f"NUM_EVO_PANELS={DEFAULT_NUM_EVO_PANELS}")
print()

rows = [json.loads(l) for l in open(PATH, "r", encoding="utf-8") if l.strip()]
TARGET_IDX = 10  # const(2.718)
r = rows[TARGET_IDX]
prog_list = r["c2v"]
prog = list_to_program(prog_list)
log_c = np.asarray(r.get("log_constants", [0.0]*8), dtype=np.float64)
evo = (10.0 ** log_c).astype(np.float64)
print(f"idx={TARGET_IDX} side=c2v size={len(prog_list)} log_constants={log_c}")
print(f"evo (10^log)={evo}")
print(f"Program first 10 ins:")
for ins in prog_list[:10]:
    print(f"  {ins}")
print()

# Replicate validate_c2v with the same rng seed used by guard (0 + TARGET_IDX).
rng = np.random.default_rng(0 + TARGET_IDX)
evo_panels = _sample_evo_panels(evo, DEFAULT_NUM_EVO_PANELS, rng)
print(f"evo_panels count = {len(evo_panels)}")
for i, p in enumerate(evo_panels):
    print(f"  panel[{i}] = {p}")
print()

for cfg in range(DEFAULT_NUM_CONFIGS):
    incoming = rng.uniform(INCOMING_SAMPLE_LO, INCOMING_SAMPLE_HI, size=DEFAULT_DEG-1)
    input_probes = _build_input_probes(incoming, DEFAULT_PERTURB_DELTA)
    perm_list = list(_structured_perms(incoming, DEFAULT_NUM_PERMUTATIONS, rng))
    print(f"=== cfg={cfg} base incoming = {incoming}")
    print(f"  #input_probes = {len(input_probes)}, #perms = {len(perm_list)}")
    for evo_idx, ev in enumerate(evo_panels):
        vm = _make_vm(incoming, has_channel_llr=False, deg=DEFAULT_DEG,
                      iter_idx=cfg, evo_consts=ev)
        base = _run(prog, vm, "c2v")
        print(f"  evo{evo_idx}: base = {base}")
        if base is None:
            print("    -> base faulty, would return False")
            continue
        # dependence check
        outs_perturb = []
        for lbl, perturbed in input_probes:
            vm2 = _make_vm(perturbed, has_channel_llr=False, deg=DEFAULT_DEG,
                           iter_idx=cfg, evo_consts=ev)
            o = _run(prog, vm2, "c2v")
            outs_perturb.append((lbl, o))
        finite_perturb = [(l, o) for l, o in outs_perturb if o is not None]
        none_perturb = [l for l, o in outs_perturb if o is None]
        diffs = [(l, o, abs(o - base)) for l, o in finite_perturb]
        diffs_above = [(l, o, d) for l, o, d in diffs if d >= EPS_DEPENDENCY]
        print(f"    perturb: total={len(outs_perturb)} finite={len(finite_perturb)} "
              f"None={len(none_perturb)} diff>=EPS={len(diffs_above)}")
        if diffs_above:
            print(f"      sample diff>=EPS: {diffs_above[:5]}")
        else:
            print(f"      ALL perturbations within EPS of base -> would FAIL dependence")
        # Examine the actual perturbed values
        finite_vals = [o for _, o in finite_perturb]
        if finite_vals:
            print(f"      perturb out range: min={min(finite_vals)} max={max(finite_vals)}")
        # perm invariance check
        bad_perm = []
        for perm in perm_list:
            vm3 = _make_vm(perm, has_channel_llr=False, deg=DEFAULT_DEG,
                           iter_idx=cfg, evo_consts=ev)
            o = _run(prog, vm3, "c2v")
            if o is None:
                bad_perm.append(("None", perm))
            elif abs(o - base) > EPS_INVARIANCE:
                bad_perm.append((o, perm))
        print(f"    perm: {len(perm_list)} perms, violations = {len(bad_perm)}")
    print()
