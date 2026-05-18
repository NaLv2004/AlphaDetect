"""Trace why validator accepts a constant-output post-DCE program."""
from __future__ import annotations
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from pushgp.genome import list_to_program
from pushgp.validators import _make_vm, _run, _build_input_probes, validate_c2v, DEFAULT_PERTURB_DELTA

RUN = "post_refactor_pop20_g1_20260518-155701"
PATH = os.path.join(ROOT, "results", "logged_evolution", RUN, "individuals.jsonl")

rows = [json.loads(l) for l in open(PATH, "r", encoding="utf-8") if l.strip()]
r = rows[2]
prog_list = r["c2v"]
prog = list_to_program(prog_list)
print(f"Post-DCE c2v program (size={len(prog_list)}):")
for ins in prog_list[:30]:
    print(f"  {ins}")

# 1. validator says ok
ok, why = validate_c2v(prog, rng=np.random.default_rng(42))
print(f"\nValidator(seed=42): ok={ok}  why={why}")

# 2. our random probe says const
rng = np.random.default_rng(0)
outs = []
for _ in range(20):
    inc = rng.uniform(-3, 3, size=7)
    vm = _make_vm(inc, channel_llr=0.0, has_channel_llr=False, evo_consts=np.ones(8))
    o = _run(prog, vm, "c2v")
    outs.append(o)
print(f"\nOutputs on 20 random incomings: min={min(outs)} max={max(outs)}")
print(f"  first 5: {outs[:5]}")

# 3. show the actual probes the validator used
rng2 = np.random.default_rng(42)
# replicate first cfg
L_v = float(rng2.uniform(-2, 2))  # consumed for c2v too?
print(f"\nFirst rng draw L_v={L_v}")
incoming = rng2.uniform(-3, 3, size=7)
print(f"Validator's base incoming: {incoming}")
probes = _build_input_probes(incoming, DEFAULT_PERTURB_DELTA)
print(f"# probes generated: {len(probes)}")
vm = _make_vm(incoming, channel_llr=0.0, has_channel_llr=False, evo_consts=np.ones(8))
base_out = _run(prog, vm, "c2v")
print(f"baseline output on validator's incoming: {base_out}")
finite_outs = [base_out]
for label, p in probes[:20]:
    vm = _make_vm(p, channel_llr=0.0, has_channel_llr=False, evo_consts=np.ones(8))
    o = _run(prog, vm, "c2v")
    if o is None:
        print(f"  {label:30s} -> FAULT")
    else:
        finite_outs.append(o)
        diff = abs(o - base_out) if base_out is not None else float('inf')
        flag = " <-- DIFFERS" if diff > 1e-9 else ""
        print(f"  {label:30s} -> {o:+.6g}  delta={diff:.3g}{flag}")
print(f"\nfinite outs unique: {len(set(round(o,9) for o in finite_outs))}")
