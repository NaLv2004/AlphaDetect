"""For each individual, re-run validator on the post-DCE program and
report (a) does it pass? (b) is the program constant on wide probes?

If a program passes validator AND is constant on wide probes — that's a
validator bug, not a guard bug.
"""
from __future__ import annotations
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from pushgp.genome import list_to_program
from pushgp.validators import validate_v2c, validate_c2v, _make_vm, _run

RUN = "run_20260518_161605"
PATH = os.path.join(ROOT, "results", "logged_evolution", RUN, "individuals.jsonl")

def is_constant(prog, side, evo, n=60, lo=-10.0, hi=10.0, deg=8):
    rng = np.random.default_rng(7777)
    outs = []
    for _ in range(n):
        inc = rng.uniform(lo, hi, size=deg-1)
        L_v = float(rng.uniform(lo, hi))
        vm = _make_vm(inc, channel_llr=L_v, has_channel_llr=(side=="v2c"), evo_consts=evo)
        outs.append(_run(prog, vm, side))
    finite = [o for o in outs if o is not None and np.isfinite(o)]
    if not finite:
        return "fault-all", None
    if max(finite) == min(finite):
        return "const", finite[0]
    return "alive", max(finite) - min(finite)

def main():
    rows = [json.loads(l) for l in open(PATH, "r", encoding="utf-8") if l.strip()]
    print(f"{'idx':>3} {'side':<4} {'guard_seed_pass':>5} {'wide_pass':>5} {'class':<12} {'val_msg'}")
    n_bug = 0
    for i, r in enumerate(rows):
        log_c = np.asarray(r.get("log_constants", [0.0]*8), dtype=np.float64)
        evo = (10.0 ** log_c).astype(np.float64)
        for side in ("v2c", "c2v"):
            prog_list = r.get(side)
            if not prog_list: continue
            prog = list_to_program(prog_list)
            validator = validate_v2c if side == "v2c" else validate_c2v
            # Guard seed = 0 + i  (same as _validator_guard with rng_seed=0)
            rng_g = np.random.default_rng(0 + i)
            ok_g, msg_g = validator(prog, rng=rng_g, evo_consts=evo)
            # Try with different seed (wide independent test)
            rng_w = np.random.default_rng(42 + i*97)
            ok_w, msg_w = validator(prog, rng=rng_w, evo_consts=evo)
            cls, val = is_constant(prog, side, evo)
            tag = ""
            if cls == "const" and ok_g and ok_w:
                tag = " <-- BUG: validator passes constant"
                n_bug += 1
            elif cls == "const" and ok_g and not ok_w:
                tag = " <-- guard seed lucky"
            elif cls == "const" and not ok_g:
                tag = " <-- guard should have reverted!"
            print(f"{i:>3} {side:<4} {str(ok_g):<5} {str(ok_w):<5} {cls:<12} val={val} {tag}")
    print(f"\nTotal validator-bug constants: {n_bug}")

if __name__ == "__main__":
    main()
