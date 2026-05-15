"""Diagnostic: load a specific (gen, v_idx, c_idx) pair from a run's
individuals.jsonl, then:
  1. Run the C2V program through validate_c2v with several seeds.
  2. Run the C2V program through TraceVM with several different input
     configurations to see whether the live expression VARIES with input.
This separates the question "does the program actually emit a constant"
from "does the tracer's single-shot path mislead us".
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from pushgp.serialize import dict_to_program
from pushgp.trace import trace_program
from pushgp.validators import (
    DEFAULT_DEG, validate_c2v, validate_v2c,
)
from pushgp.vm import VM


def find_record(jsonl: Path, gen: int, c_idx: int):
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("gen") == gen and r.get("c_idx") == c_idx:
            return r
    return None


def run_concrete(prog, *, side: str, incoming, llr=0.0):
    """Run program with concrete inputs, return float top-of-stack."""
    vm = VM()
    vm.state.ctx_channel_llr = float(llr)
    vm.state.ctx_has_channel_llr = (side == "v2c")
    vm.state.ctx_incoming = np.asarray(incoming, dtype=np.float64)
    vm.state.ctx_deg = len(incoming) + 1
    vm.state.ctx_iter = 0
    vm.state.ctx_max_iter = 8
    vm.state.ctx_edge_index = 0
    if side == "v2c":
        vm.state.floats.push(float(llr))
    vm.state.ints.push(0)
    vm.state.ints.push(vm.state.ctx_deg)
    vm.state.ints.push(0)
    vm.state.ints.push(8)
    vm.state.fvecs.push(vm.state.ctx_incoming.copy())
    return vm.run(prog)


def main(argv):
    run = argv[1] if len(argv) > 1 else "fromscratch1"
    gen = int(argv[2]) if len(argv) > 2 else 4
    c_idx = int(argv[3]) if len(argv) > 3 else 22

    candidates = [
        Path("results") / "logged_evolution" / run / "individuals.jsonl",
        Path(_PROJ) / "results" / "logged_evolution" / run / "individuals.jsonl",
        Path(_PROJ).parent.parent / "results" / "logged_evolution" / run / "individuals.jsonl",
    ]
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        print("No individuals.jsonl found", file=sys.stderr)
        return 2

    rec = find_record(path, gen=gen, c_idx=c_idx)
    if rec is None:
        print(f"No record for gen={gen} c_idx={c_idx}", file=sys.stderr)
        return 3

    print(f"=== run={run} gen={gen} c_idx={c_idx} ===")
    print(f"recorded fitness = {rec.get('fitness')}")
    print(f"recorded valid   = {rec.get('valid')}")
    k = rec.get("log_constants", [])
    evo = np.array([10.0 ** float(x) for x in k], dtype=np.float64)
    print(f"K (log10)        = {k}")
    print(f"evo_consts       = {[f'{x:.3f}' for x in evo]}")

    c2v_prog = dict_to_program(rec["c2v"])
    v2c_prog = dict_to_program(rec["v2c"])
    print(f"\nC2V tree size={len(rec['c2v'])} (top-level)  V2C tree size={len(rec['v2c'])}")

    # ==== A) Validator round-trip ====
    print("\n--- (A) Validator: 5 different seeds ---")
    for seed in [0, 1, 2, 7, 42]:
        ok, why = validate_c2v(c2v_prog, rng=np.random.default_rng(seed),
                                deg=DEFAULT_DEG, evo_consts=evo)
        print(f"  c2v seed={seed:>2}: {'PASS' if ok else 'FAIL'}  ({why})")
    for seed in [0, 1, 2, 7, 42]:
        ok, why = validate_v2c(v2c_prog, rng=np.random.default_rng(seed),
                                deg=DEFAULT_DEG, evo_consts=evo)
        print(f"  v2c seed={seed:>2}: {'PASS' if ok else 'FAIL'}  ({why})")

    # ==== B) Concrete-input variation: does C2V output change w/ input? ====
    print("\n--- (B) C2V concrete-input variation (deg=8) ---")
    rng = np.random.default_rng(123)
    for trial in range(8):
        inc = rng.uniform(-3.0, 3.0, size=DEFAULT_DEG - 1)
        vm = VM()
        vm.state.ctx_evo_constants = evo.copy()
        vm.state.ctx_has_channel_llr = False
        vm.state.ctx_incoming = inc.copy()
        vm.state.ctx_deg = DEFAULT_DEG
        vm.state.ctx_iter = 0
        vm.state.ctx_max_iter = 8
        vm.state.ctx_edge_index = 0
        vm.state.ints.push(0)
        vm.state.ints.push(DEFAULT_DEG)
        vm.state.ints.push(0)
        vm.state.ints.push(8)
        vm.state.fvecs.push(inc.copy())
        out = vm.run(c2v_prog)
        print(f"  trial {trial}: incoming={np.round(inc,2).tolist()}  ->  out={out}")

    # ==== C) Trace under different inputs to compare live expressions ====
    print("\n--- (C) TraceVM under varied inputs (does live expr change?) ---")
    rng = np.random.default_rng(99)
    seen_exprs = set()
    for trial in range(8):
        inc = rng.uniform(-3.0, 3.0, size=DEFAULT_DEG - 1)
        r = trace_program(
            c2v_prog,
            ctx_channel_llr=0.0,
            ctx_incoming=inc,
            ctx_noise_var=1.0,
            ctx_iter=0,
            ctx_max_iter=8,
            ctx_deg=DEFAULT_DEG,
            ctx_edge_index=0,
            ctx_evo_constants=evo,
            ctx_has_channel_llr=False,
        )
        expr = r["live_expr"]
        seen_exprs.add(expr)
        head = (expr or "<empty>")[:120]
        print(f"  trial {trial}: out={r['value']}   live_head={head}")
    print(f"\n  distinct live exprs across 8 random inputs: {len(seen_exprs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
