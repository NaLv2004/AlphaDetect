"""Step-by-step trace of ind#2 c2v with proper VM API."""
from __future__ import annotations
import os, sys, json, time as _t
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from pushgp.genome import list_to_program
from pushgp.validators import _make_vm, _seed_c2v_stacks, DEFAULT_PERTURB_DELTA

RUN = "post_refactor_pop20_g1_20260518-155701"
PATH = os.path.join(ROOT, "results", "logged_evolution", RUN, "individuals.jsonl")
rows = [json.loads(l) for l in open(PATH, "r", encoding="utf-8") if l.strip()]
prog_list = rows[2]["c2v"]
prog = list_to_program(prog_list)

rng = np.random.default_rng(42)
_ = float(rng.uniform(-2, 2))
base = rng.uniform(-3, 3, size=7).copy()
perturbed = base.copy()
perturbed[0] += 5 * DEFAULT_PERTURB_DELTA

print("Base    :", base)
print("Perturbed:", perturbed)
print(f"\nProgram size: {len(prog_list)} top-level instructions")
for i, ins in enumerate(prog_list):
    nested = ""
    if "code_block" in ins:
        nested = f" cb={[c['name'] for c in ins['code_block']]}"
    if "code_block2" in ins:
        nested += f" cb2={[c['name'] for c in ins['code_block2']]}"
    print(f"  [{i:02d}] {ins['name']}{nested}")

def snap(vm):
    s = vm.state
    vt = s.fvecs.peek()
    return {
        "ftop": s.floats.peek(),
        "itop": s.ints.peek(),
        "btop": s.bools.peek(),
        "vtop": (vt.tolist() if vt is not None and len(vt) <= 6 else (f"len{len(vt)}={vt[:5].tolist()}..." if vt is not None else None)),
        "lens": (s.floats.depth(), s.ints.depth(), s.bools.depth(), s.fvecs.depth()),
        "fault": s.fault,
        "freason": s.fault_reason,
    }

def trace(prog, incoming, label):
    vm = _make_vm(incoming, channel_llr=0.0, has_channel_llr=False, evo_consts=np.ones(8))
    _seed_c2v_stacks(vm)
    print(f"\n=== {label} ===")
    snaps = []
    vm._t_start = _t.perf_counter()
    for i, ins in enumerate(prog):
        if vm.state.fault:
            print(f"  [{i:02d}] <abort {vm.state.fault_reason}>")
            snaps.append(("ABORT", None))
            break
        vm._step(ins)
        sn = snap(vm)
        snaps.append((ins.name, sn))
        print(f"  [{i:02d}] {ins.name:24s} F={sn['ftop']} I={sn['itop']} B={sn['btop']} V={sn['vtop']} |F,I,B,V|={sn['lens']}")
    out = vm.state.floats.peek()
    print(f"  --> final float top = {out}")
    return snaps, out

snap_a, out_a = trace(prog, base, "BASE")
snap_b, out_b = trace(prog, perturbed, "PERT (pos0+5d)")

print(f"\n>>> BASE output={out_a}   PERT output={out_b}")
print("\nDiverging steps:")
for i in range(min(len(snap_a), len(snap_b))):
    a_name, a_sn = snap_a[i]
    b_name, b_sn = snap_b[i]
    if a_sn != b_sn:
        print(f"  [{i:02d}] {a_name}")
        if a_sn is not None and b_sn is not None:
            for k in a_sn:
                if a_sn[k] != b_sn[k]:
                    print(f"        {k:6s}: BASE={a_sn[k]!r}  PERT={b_sn[k]!r}")
