"""Run only Part 4 (node budget comparison) of the Gen72 analysis.
Uses the fixed vm.py where Float.Inv = 1/x (was silently a no-op before).
This shows Gen72 behavior with BOTH old (no-op) and new (1/x) Float.Inv.
"""
import sys, os
CODE_DIR = r"D:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\code"
sys.path.insert(0, CODE_DIR)

import time
import numpy as np
from vm import MIMOPushVM
from bp_decoder import BPStackDecoder

# Reuse helpers from analyze_truebp1
from analyze_truebp1 import (
    qam16_constellation, GEN72_PROG, GEN8_PROG, MMSE_LB_PROG, DIST_PROG,
    node_budget_test,
)

print("="*70)
print("PART 4 (RERUN): NODE BUDGET COMPARISON")
print("vm.py: Float.Inv=1/x (FIXED; old behavior was no-op → score=cum_dist)")
print("="*70)

t0 = time.time()
node_budget_test(n_trials=400, snr_list=[10, 12, 14, 16])
print(f"\n[Part 4 done in {time.time()-t0:.1f}s]")
