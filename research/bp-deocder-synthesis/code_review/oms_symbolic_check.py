"""Check OMS v2c & c2v against the *symbolic* validator (not probe).
Also test: if we removed the odd-parity check, would OMS still pass?
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cpp_seeder"))
import pushgp_cpp_seeder as M
from pushgp_ldpc.adapter import oms_seed_genome
from pushgp.genome import program_to_list

oms = oms_seed_genome()
hv = M.build_program(program_to_list(oms.prog_v2c))
hc = M.build_program(program_to_list(oms.prog_c2v))

print("=== OMS under symbolic validator (deg=8, iter=0) ===")
for iter_idx in (0, 1, 5):
    okv, rv = M.symbolic_validate_v2c(hv, 8, iter_idx)
    okc, rc = M.symbolic_validate_c2v(hc, 8, iter_idx)
    print(f"iter={iter_idx}  v2c: ok={okv}  reason={rv!r}")
    print(f"iter={iter_idx}  c2v: ok={okc}  reason={rc!r}")

# Trace symbolic outputs for inspection
print("\n=== symbolic_trace_v2c top expr ===")
tr = M.symbolic_trace_v2c(hv, 8, 0)
print(f"opaque={tr['opaque']} reason={tr.get('opaque_reason')!r} branches={tr['branches_seen']}")
if tr['steps']:
    last = tr['steps'][-1]
    print(f"final float stack: {last.get('float', [])}")
