"""Iteration-by-iteration diff: cdce.decode_bp (with OMS adapter genome)
vs ldpc_5g.decode_oms_fast (textbook OMS).  Same H, same input LLR.

If both implement the same OMS, llr_post at every iteration should be
bit-identical to ~1e-12.  Any divergence pinpoints the bug.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ldpc_5g import build_parity, build_oms_context, decode_oms_fast, bpsk_modulate, bpsk_llr
from pushgp_ldpc.eval_cpp import _get_cdce, _get_parity_handle
from pushgp.serialize import program_to_dict
from pushgp_ldpc.adapter import oms_seed_genome

par = build_parity(bgn=2, set_idx=1, zc=8)
N, M = par.cols, par.rows
print(f"BG2 set1 Zc=8: N={N} M={M}")

rng = np.random.default_rng(42)
# Use the trivial all-zero codeword for the diff (well-defined, no encoder issues).
cw = np.zeros(N, dtype=np.int8)
tx = bpsk_modulate(cw)  # +1
# Moderate noise, sigma2 corresponding to Eb/N0 = 2 dB at R=0.5
sigma2 = 1.0 / (2.0 * 0.5 * 10 ** (2.0 / 10.0))
sigma = np.sqrt(sigma2)
# Mimic _channel_inputs: first 2*Zc bits punctured (LLR=0); transmit rest
skip = 2 * par.zc
tx_chan = tx[skip:]
rx = tx_chan + sigma * rng.standard_normal(tx_chan.size)
llr_chan = bpsk_llr(rx, sigma2)
llr_in = np.zeros(N, dtype=np.float64)
llr_in[skip:] = llr_chan

# Reference: textbook OMS at max_iter T
ctx_ref = build_oms_context(par)

# pushgp path
cdce = _get_cdce()
parH = _get_parity_handle(par)
oms = oms_seed_genome()
v_dict = program_to_dict(oms.prog_v2c)
c_dict = program_to_dict(oms.prog_c2v)
evo = oms.evo_const_values().astype(np.float64)

print("\niter |   max|Δ post|   |  ref errs | pushgp errs")
print("-" * 55)
for T in range(1, 12):
    ref_post, _ = decode_oms_fast(llr_in.copy(), ctx_ref, max_iter=T, offset=0.25)
    pgp_post, _ = cdce.decode_bp(llr_in.copy(), parH, v_dict, c_dict, evo, T, 0.25, 0.5)
    diff = np.max(np.abs(ref_post - pgp_post))
    ref_err = int(((ref_post < 0.0).astype(np.int8) != cw).sum())
    pgp_err = int(((pgp_post < 0.0).astype(np.int8) != cw).sum())
    print(f"{T:>4} | {diff:>13.3e} | {ref_err:>9d} | {pgp_err:>10d}")

# Also print V2C and C2V program dicts so we can eyeball them.
print("\nv_dict =", v_dict)
print("\nc_dict =", c_dict)
print("\nevo    =", evo)
