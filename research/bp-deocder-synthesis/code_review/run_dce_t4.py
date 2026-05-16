"""Run DCE T4 (BER equivalence) only — this one is heavy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pushgp.tests.test_dce import test_t4_ber_equivalence

# Use the production-equivalent settings: 20 pairs, 3 SNRs, 200 frames.
test_t4_ber_equivalence(20, (-2.0, -1.0, 0.0), 200, 8)
print("\n=== T4 PASSED ===")
