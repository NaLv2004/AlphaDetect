"""Diagnose WHY V2C with 4-op filter has 0.003% pass rate.

For each rejected program, classify the failure mode:
  A. fault (which reason?) -- step_max / recur_max / flop_max / handler
  B. final float stack empty
  C. final float top is not finite
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent             # bp-deocder-synthesis
sys.path.insert(0, str(ROOT))

from pushgp.random_program import RandomProgramGenerator
from pushgp.vm import VM
import numpy as np


V2C_FILTER = ["FVec.Len", "FVec.At", "Float.Add", "Exec.DoTimes"]


def main(n_progs: int = 500) -> int:
    rpg = RandomProgramGenerator(rng=np.random.default_rng(42), max_recur_depth=2)
    rng_size = np.random.default_rng(123)

    counts = {
        "pass": 0,
        "fault_step_max": 0,
        "fault_recur_max": 0,
        "fault_flop_max": 0,
        "fault_other": 0,
        "stack_empty_at_end": 0,
        "top_non_finite": 0,
    }
    fault_reasons: dict[str, int] = {}

    for _ in range(n_progs):
        size = int(rng_size.integers(4, 21))
        prog = rpg.random_program(V2C_FILTER, size, size)

        vm = VM()
        # V2C seeding
        vm.state.ctx_channel_llr = 0.5
        vm.state.ctx_has_channel_llr = True
        vm.state.ctx_incoming = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], dtype=np.float64)
        vm.state.ctx_deg = 8
        vm.state.ctx_iter = 0
        vm.state.ctx_max_iter = 25
        vm.state.ctx_noise_var = 1.0
        vm.state.ctx_edge_index = 0
        vm.state.ctx_code_rate = 0.5
        vm.state.ctx_evo_constants = np.array([0.1] * 8, dtype=np.float64)
        # Seed stacks (mirrors validators._seed_v2c_stacks)
        vm.state.floats.push(vm.state.ctx_channel_llr)
        vm.state.ints.push(vm.state.ctx_edge_index)
        vm.state.ints.push(vm.state.ctx_deg)
        vm.state.ints.push(vm.state.ctx_iter)
        vm.state.ints.push(vm.state.ctx_max_iter)
        vm.state.fvecs.push(vm.state.ctx_incoming.copy())

        vm.execute_block(prog)

        if vm.state.fault:
            reason = vm.state.fault_reason or "unknown"
            fault_reasons[reason] = fault_reasons.get(reason, 0) + 1
            if "step_max" in reason:
                counts["fault_step_max"] += 1
            elif "recur_max" in reason:
                counts["fault_recur_max"] += 1
            elif "flop_max" in reason:
                counts["fault_flop_max"] += 1
            else:
                counts["fault_other"] += 1
        else:
            top = vm.state.floats.peek()
            if top is None:
                counts["stack_empty_at_end"] += 1
            elif not np.isfinite(top):
                counts["top_non_finite"] += 1
            else:
                counts["pass"] += 1

    print(f"=== diagnose v2c-4op: n={n_progs} ===")
    for k, v in counts.items():
        pct = 100.0 * v / n_progs
        print(f"  {k:24s}: {v:5d} ({pct:5.1f}%)")
    print(f"--- fault reasons (raw) ---")
    for k, v in sorted(fault_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s}: {v}")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    sys.exit(main(n))
