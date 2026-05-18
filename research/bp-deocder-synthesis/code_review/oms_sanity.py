"""OMS sanity: Python validator, C++ validator, both VMs run OMS and agree.

Verifies that after the rev-2 validator + DomainError clamp 30 refactor,
the OMS genome is STILL accepted by both Python and C++ validators on
randomized panels, and that both VMs produce the same numerical output
on a fixed (incoming, L_v) probe.
"""
from __future__ import annotations
import os, sys, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cpp_seeder"))

import numpy as np
import pushgp_cpp_seeder as M  # type: ignore[import]

from pushgp_ldpc.adapter import oms_seed_genome
from pushgp.validators import validate_v2c, validate_c2v
from pushgp.genome import program_to_list
from pushgp.validators import _make_vm, _run

oms = oms_seed_genome()
prog_v = oms.prog_v2c
prog_c = oms.prog_c2v
evo = np.power(10.0, np.asarray(oms.log_constants, dtype=np.float64))

print(f"OMS evo[0:4] = {evo[:4]}")

# ---- Python validation ----
n_trials = 20
py_v_ok = sum(1 for i in range(n_trials)
              if validate_v2c(prog_v, rng=np.random.default_rng(i), evo_consts=evo)[0])
py_c_ok = sum(1 for i in range(n_trials)
              if validate_c2v(prog_c, rng=np.random.default_rng(i), evo_consts=evo)[0])
print(f"PY validate: v2c {py_v_ok}/{n_trials},  c2v {py_c_ok}/{n_trials}")

# ---- C++ validation ----
hv = M.build_program(program_to_list(prog_v))
hc = M.build_program(program_to_list(prog_c))
cpp_v_ok = sum(1 for i in range(n_trials)
               if M.validate_random(hv, "v2c", evo, 8, 5, 5, 0, int(np.random.default_rng(i).integers(0, 2**63 - 1)))[0])
cpp_c_ok = sum(1 for i in range(n_trials)
               if M.validate_random(hc, "c2v", evo, 8, 5, 5, 0, int(np.random.default_rng(i).integers(0, 2**63 - 1)))[0])
print(f"C++ validate: v2c {cpp_v_ok}/{n_trials},  c2v {cpp_c_ok}/{n_trials}")

# ---- Numerical agreement on one probe ----
rng = np.random.default_rng(42)
deg = 8
incoming = rng.uniform(-3.0, 3.0, size=deg - 1)
L_v = float(rng.uniform(-2.0, 2.0))

vm = _make_vm(incoming, channel_llr=L_v, has_channel_llr=True, deg=deg, iter_idx=0, evo_consts=evo)
py_v_out = _run(prog_v, vm, "v2c")

vm = _make_vm(incoming, channel_llr=0.0, has_channel_llr=False, deg=deg, iter_idx=0, evo_consts=evo)
py_c_out = _run(prog_c, vm, "c2v")

cpp_v_out = M.run_program(hv, "v2c", evo, incoming, L_v, 8, 0)
cpp_c_out = M.run_program(hc, "c2v", evo, incoming, 0.0, 8, 0)

print(f"PY  v2c -> {py_v_out!r}")
print(f"CPP v2c -> {cpp_v_out!r}")
print(f"PY  c2v -> {py_c_out!r}")
print(f"CPP c2v -> {cpp_c_out!r}")

ok = True
if py_v_ok < 18 or cpp_v_ok < 18 or py_c_ok < 18 or cpp_c_ok < 18:
    print("FAIL: validator acceptance dropped below 18/20"); ok = False
if py_v_out is None or cpp_v_out is None or abs(py_v_out - cpp_v_out) > 1e-6:
    print("FAIL: v2c numeric mismatch"); ok = False
if py_c_out is None or cpp_c_out is None or abs(py_c_out - cpp_c_out) > 1e-6:
    print("FAIL: c2v numeric mismatch"); ok = False
print("OMS SANITY: PASS" if ok else "OMS SANITY: FAIL")
sys.exit(0 if ok else 1)
