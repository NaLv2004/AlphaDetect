"""Validator-equivalence test: Python validate_*_with_panel == C++ validate_with_panels.

Pre-generates random panels (L_v, incoming, perturb_indices, permutations) in
Python, then runs both the Python panel-driven validator (mirror of the
algorithm in pushgp/validators.py minus the inline RNG calls) and the C++
validate_with_panels. Asserts (ok, reason) agree on every program.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # cpp_seeder
PROJ = ROOT.parent                              # bp-deocder-synthesis
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(ROOT))

import pushgp_cpp_seeder as M
from pushgp.random_program import RandomProgramGenerator, V2C_INSTR, C2V_INSTR
from pushgp.program import Instruction
from pushgp import validators as V
from pushgp.vm import VM


# ---------- panel-driven Python validator (mirror of validator.hpp) ----------


def _make_vm_v2c(incoming, L_v, deg, iter_idx, evo):
    vm = VM()
    vm.state.ctx_has_channel_llr = True
    vm.state.ctx_channel_llr = float(L_v)
    vm.state.ctx_incoming = np.asarray(incoming, dtype=np.float64).copy()
    vm.state.ctx_deg = int(deg)
    vm.state.ctx_iter = int(iter_idx)
    vm.state.ctx_max_iter = 25
    vm.state.ctx_noise_var = 1.0
    vm.state.ctx_edge_index = 0
    vm.state.ctx_code_rate = 0.5
    vm.state.ctx_evo_constants = np.asarray(evo, dtype=np.float64).copy()
    return vm


def _make_vm_c2v(incoming, deg, iter_idx, evo):
    vm = VM()
    vm.state.ctx_has_channel_llr = False
    vm.state.ctx_channel_llr = 0.0
    vm.state.ctx_incoming = np.asarray(incoming, dtype=np.float64).copy()
    vm.state.ctx_deg = int(deg)
    vm.state.ctx_iter = int(iter_idx)
    vm.state.ctx_max_iter = 25
    vm.state.ctx_noise_var = 1.0
    vm.state.ctx_edge_index = 0
    vm.state.ctx_code_rate = 0.5
    vm.state.ctx_evo_constants = np.asarray(evo, dtype=np.float64).copy()
    return vm


def _run(prog, vm, side):
    if side == "v2c":
        V._seed_v2c_stacks(vm)
    else:
        V._seed_c2v_stacks(vm)
    return vm.run(prog)


def py_validate_v2c_with_panels(prog, panels, evo, deg=8):
    for cfg, P in enumerate(panels):
        L_v = P["L_v"]
        incoming = P["incoming"]
        # baseline
        vm = _make_vm_v2c(incoming, L_v, deg, cfg, evo)
        base = _run(prog, vm, "v2c")
        if base is None:
            return False, f"v2c cfg{cfg}: faulty / non-finite baseline"
        # 1. L_v dependence
        vm2 = _make_vm_v2c(incoming, L_v + V.DEFAULT_PERTURB_DELTA, deg, cfg, evo)
        out2 = _run(prog, vm2, "v2c")
        if out2 is None:
            return False, f"v2c cfg{cfg}: faulty after L_v perturbation"
        if abs(out2 - base) < V.EPS_DEPENDENCY:
            return False, f"v2c cfg{cfg}: output independent of L_v"
        # 2. incoming dependence
        changed = False
        tries_max = min(len(incoming), 4)
        for t in range(tries_max):
            idx = P["perturb_indices"][t]
            inc2 = np.array(incoming, dtype=np.float64)
            inc2[idx] += V.DEFAULT_PERTURB_DELTA
            vm3 = _make_vm_v2c(inc2, L_v, deg, cfg, evo)
            out3 = _run(prog, vm3, "v2c")
            if out3 is None:
                return False, f"v2c cfg{cfg}: faulty after incoming perturbation"
            if abs(out3 - base) >= V.EPS_DEPENDENCY:
                changed = True
                break
        if not changed:
            inc3 = np.asarray(incoming, dtype=np.float64) + V.DEFAULT_PERTURB_DELTA
            vm5 = _make_vm_v2c(inc3, L_v, deg, cfg, evo)
            out5 = _run(prog, vm5, "v2c")
            if out5 is None or abs(out5 - base) < V.EPS_DEPENDENCY:
                return False, f"v2c cfg{cfg}: output independent of incoming"
        # 3. permutation invariance
        for perm in P["permutations"]:
            vm4 = _make_vm_v2c(perm, L_v, deg, cfg, evo)
            out4 = _run(prog, vm4, "v2c")
            if out4 is None:
                return False, f"v2c cfg{cfg}: faulty under permutation"
            if abs(out4 - base) > V.EPS_INVARIANCE:
                return False, f"v2c cfg{cfg}: not permutation-invariant"
    return True, "ok"


def py_validate_c2v_with_panels(prog, panels, evo, deg=8):
    for cfg, P in enumerate(panels):
        incoming = P["incoming"]
        vm = _make_vm_c2v(incoming, deg, cfg, evo)
        base = _run(prog, vm, "c2v")
        if base is None:
            return False, f"c2v cfg{cfg}: faulty / non-finite baseline"
        changed = False
        tries_max = min(len(incoming), 4)
        for t in range(tries_max):
            idx = P["perturb_indices"][t]
            inc2 = np.array(incoming, dtype=np.float64)
            inc2[idx] += V.DEFAULT_PERTURB_DELTA
            vm2 = _make_vm_c2v(inc2, deg, cfg, evo)
            out2 = _run(prog, vm2, "c2v")
            if out2 is None:
                return False, f"c2v cfg{cfg}: faulty after incoming perturbation"
            if abs(out2 - base) >= V.EPS_DEPENDENCY:
                changed = True
                break
        if not changed:
            inc3 = np.asarray(incoming, dtype=np.float64) + V.DEFAULT_PERTURB_DELTA
            vm4 = _make_vm_c2v(inc3, deg, cfg, evo)
            out4 = _run(prog, vm4, "c2v")
            if out4 is None or abs(out4 - base) < V.EPS_DEPENDENCY:
                return False, f"c2v cfg{cfg}: output independent of incoming"
        for perm in P["permutations"]:
            vm3 = _make_vm_c2v(perm, deg, cfg, evo)
            out3 = _run(prog, vm3, "c2v")
            if out3 is None:
                return False, f"c2v cfg{cfg}: faulty under permutation"
            if abs(out3 - base) > V.EPS_INVARIANCE:
                return False, f"c2v cfg{cfg}: not permutation-invariant"
    return True, "ok"


# ---------- panel generation ----------


def make_panels(rng, deg=8, num_configs=3, num_permutations=5):
    panels = []
    for cfg in range(num_configs):
        L_v = float(rng.uniform(-2.0, 2.0))
        incoming = rng.uniform(-3.0, 3.0, size=deg - 1).astype(np.float64)
        perturb_indices = []
        n = incoming.size
        tries_max = min(n, 4)
        for t in range(tries_max):
            idx = int((cfg * 7 + t * 3 + rng.integers(0, n)) % n)
            perturb_indices.append(idx)
        perms = [rng.permutation(incoming).astype(np.float64) for _ in range(num_permutations)]
        panels.append({
            "L_v": L_v,
            "incoming": incoming,
            "perturb_indices": perturb_indices,
            "permutations": perms,
        })
    return panels


def panels_for_cpp(panels):
    """C++ binding accepts the same dict shape; just ensure float64 arrays."""
    out = []
    for P in panels:
        out.append({
            "L_v": float(P["L_v"]),
            "incoming": np.ascontiguousarray(P["incoming"], dtype=np.float64),
            "perturb_indices": list(P["perturb_indices"]),
            "permutations": [np.ascontiguousarray(p, dtype=np.float64) for p in P["permutations"]],
        })
    return out


# ---------- program serialization ----------


def program_to_list(prog):
    out = []
    for ins in prog:
        d = {"name": ins.name}
        if ins.code_block is not None:
            d["code_block"] = program_to_list(ins.code_block)
        if ins.code_block2 is not None:
            d["code_block2"] = program_to_list(ins.code_block2)
        out.append(d)
    return out


def _list_to_prog(lst):
    out = []
    for d in lst:
        cb = _list_to_prog(d["code_block"]) if "code_block" in d else None
        cb2 = _list_to_prog(d["code_block2"]) if "code_block2" in d else None
        out.append(Instruction(d["name"], code_block=cb, code_block2=cb2))
    return out


# ---------- main test ----------


def main(n_progs=200, seed=42, deg=8, n_seeded=20):
    rng = np.random.default_rng(seed)
    rpg = RandomProgramGenerator(rng, max_recur_depth=2)
    instr_sets = {"v2c": V2C_INSTR, "c2v": C2V_INSTR}
    evo = np.ones(8, dtype=np.float64)
    evo_const_arr = np.ascontiguousarray(evo, dtype=np.float64)

    # ---- Phase 1: harvest some valid programs from C++ seeder so we cover pass branch.
    seeded_progs = []  # list of (side, prog_list)
    if n_seeded > 0:
        print(f"[harvest] requesting {n_seeded} valid v2c + {n_seeded} c2v programs...")
        for side in ("v2c", "c2v"):
            handles, _ = M.parallel_seed(
                side=side, n_target=n_seeded, max_attempts=2_000_000,
                threads=8, chunk_attempts=2000, min_size=4, max_size=30,
                deg=deg, num_configs=3, num_permutations=5, base_seed=seed * 31,
            )
            for h in handles:
                seeded_progs.append((side, h.to_dict()))
        print(f"[harvest] got {len(seeded_progs)} programs")

    n_match = 0
    n_disagree = 0
    n_both_pass = 0
    n_both_fail = 0
    disagreements = []

    def run_one(idx, side, prog_obj_or_list):
        nonlocal n_match, n_disagree, n_both_pass, n_both_fail
        # Build the Python prog (Instruction list) from list-of-dict.
        if isinstance(prog_obj_or_list, list) and prog_obj_or_list and isinstance(prog_obj_or_list[0], dict):
            prog = _list_to_prog(prog_obj_or_list)
            prog_list = prog_obj_or_list
        else:
            prog = prog_obj_or_list
            prog_list = program_to_list(prog)
        panel_rng = np.random.default_rng(seed * 0x9E3779B9 + idx)
        panels = make_panels(panel_rng, deg=deg)
        if side == "v2c":
            py_ok, py_reason = py_validate_v2c_with_panels(prog, panels, evo, deg=deg)
        else:
            py_ok, py_reason = py_validate_c2v_with_panels(prog, panels, evo, deg=deg)
        h = M.build_program(prog_list)
        cpp_ok, cpp_reason = M.validate_with_panels(
            h, side, panels_for_cpp(panels), evo_const_arr, deg)
        if py_ok == cpp_ok:
            n_match += 1
            if py_ok:
                n_both_pass += 1
            else:
                n_both_fail += 1
        else:
            n_disagree += 1
            if len(disagreements) < 5:
                disagreements.append({
                    "i": idx, "side": side, "size": len(prog),
                    "py": (py_ok, py_reason), "cpp": (cpp_ok, cpp_reason),
                })

    # ---- Phase 2: random programs (mostly fail).
    for i in range(n_progs):
        side = "v2c" if (i % 2 == 0) else "c2v"
        size = int(rng.integers(4, 26))
        prog = rpg.random_program(instr_sets[side], min_size=4, max_size=size + 5)
        run_one(i, side, prog)

    # ---- Phase 3: seeded programs (mostly pass).
    for j, (side, plist) in enumerate(seeded_progs):
        run_one(n_progs + j, side, plist)

    total = n_progs + len(seeded_progs)
    print(f"total={total} match={n_match} (both_pass={n_both_pass} both_fail={n_both_fail}) "
          f"disagree={n_disagree}")
    if disagreements:
        print("\nFirst disagreements:")
        for d in disagreements:
            print(f"  i={d['i']} side={d['side']} size={d['size']}")
            print(f"    py:  ok={d['py'][0]}  reason={d['py'][1]}")
            print(f"    cpp: ok={d['cpp'][0]}  reason={d['cpp'][1]}")
    if n_disagree != 0:
        sys.exit(1)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(n)
