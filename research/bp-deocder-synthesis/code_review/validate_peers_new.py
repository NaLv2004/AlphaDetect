"""Validate dumped pre-DCE peers under the new validator."""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pushgp.validators import validate_v2c, validate_c2v
from pushgp.genome import list_to_program

with open('code_review/_dce_dump.gen0.pkl', 'rb') as f:
    data = pickle.load(f)

prev = [list_to_program(p) for p in data['pre_pop_v']]
prec = [list_to_program(p) for p in data['pre_pop_c']]
print('pre_pop_v sizes:', [len(p) for p in prev])
print('pre_pop_c sizes:', [len(p) for p in prec])
print()

print('Validating pre-DCE V2C programs:')
n_v_pass = 0
for i, p in enumerate(prev):
    ok, why = validate_v2c(p, rng=np.random.default_rng(i))
    tag = 'PASS' if ok else 'REJECT'
    if ok:
        n_v_pass += 1
    print(f'  v2c #{i:2d} sz={len(p):3d}: {tag} ({why})')
print(f'  -> {n_v_pass}/{len(prev)} V2C pass')

print()
print('Validating pre-DCE C2V programs:')
n_c_pass = 0
for i, p in enumerate(prec):
    ok, why = validate_c2v(p, rng=np.random.default_rng(i + 100))
    tag = 'PASS' if ok else 'REJECT'
    if ok:
        n_c_pass += 1
    print(f'  c2v #{i:2d} sz={len(p):3d}: {tag} ({why})')
print(f'  -> {n_c_pass}/{len(prec)} C2V pass')
