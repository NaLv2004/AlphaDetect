"""Run cheap DCE tests (T1, T2, T3, T5) — leaves T4 (BER) for separate run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pushgp.tests.test_dce import (
    test_t1_hand_crafted_dead_tails,
    test_t2_invariant_self,
    test_t3_idempotence,
    test_t5_reduction_yield,
)

test_t1_hand_crafted_dead_tails()
test_t2_invariant_self(30)
test_t3_idempotence(20)
test_t5_reduction_yield(25, 60, 20, 0.10)
print("\n=== T1+T2+T3+T5 ALL PASSED ===")
