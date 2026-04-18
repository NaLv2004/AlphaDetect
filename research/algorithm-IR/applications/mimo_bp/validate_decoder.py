"""Quick validation: test LMMSE baseline and hand-crafted BP programs."""
from __future__ import annotations

import sys
import pathlib
import time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.regeneration.codegen import emit_cpp_ops
from applications.mimo_bp.mimo_simulation import generate_dataset, dataset_to_dict
from applications.mimo_bp.cpp_evaluator import CppBPIREvaluator


import math as _math

def _safe_div(a, b): return a / b if abs(b) > 1e-30 else 0.0
def _safe_log(a): return _math.log(max(a, 1e-30))
def _safe_sqrt(a): return _math.sqrt(max(a, 0.0))

_HELPERS = {
    "__builtins__": __builtins__,
    "_safe_div": _safe_div,
    "_safe_log": _safe_log,
    "_safe_sqrt": _safe_sqrt,
    "abs": abs,
    "math": _math,
}


def make_program(source: str, name: str) -> list[int]:
    """Compile a Python source string to C++ opcodes."""
    ir = compile_source_to_ir(source, name, globals_dict=_HELPERS)
    return emit_cpp_ops(ir)


def main():
    Nt, Nr, M = 16, 16, 16
    snr_db = 24.0
    n_samples = 200
    max_nodes = 500
    max_bp_iters = 5

    rng = np.random.default_rng(42)
    ds = generate_dataset(n_samples, Nt, Nr, M, snr_db, rng)
    data = dataset_to_dict(ds)

    evaluator = CppBPIREvaluator(
        Nt=Nt, Nr=Nr, mod_order=M,
        max_nodes=max_nodes, max_bp_iters=max_bp_iters,
    )

    # 1. LMMSE baseline
    t0 = time.time()
    lmmse_ber = evaluator.evaluate_lmmse(data)
    t_lmmse = time.time() - t0
    print(f"LMMSE BER:      {lmmse_ber:.6f}  ({t_lmmse:.2f}s)")

    # 2. Hand-crafted programs (mimicking classic stack decoder)
    # f_down: pass parent context + local distance
    prog_down = make_program(
        "def f_down(parent_m_down, local_dist):\n    return parent_m_down + local_dist\n",
        "f_down"
    )
    # f_up: average of children's local dist + m_up
    prog_up = make_program(
        "def f_up(sum_child_ld, sum_child_m_up, n_children):\n    return sum_child_ld + sum_child_m_up\n",
        "f_up"
    )
    # f_belief: cumulative distance (classic stack decoder scoring)
    prog_belief = make_program(
        "def f_belief(cum_dist, m_down, m_up):\n    return cum_dist\n",
        "f_belief"
    )
    # h_halt: always run all iterations (never halt early)
    prog_halt = make_program(
        "def h_halt(old_root_m_up, new_root_m_up):\n    return 0.0\n",
        "h_halt"
    )

    t0 = time.time()
    ber_classic, flops_classic = evaluator.evaluate_genome(
        prog_down, prog_up, prog_belief, prog_halt, data,
    )
    t_classic = time.time() - t0
    print(f"Classic BER:    {ber_classic:.6f}  flops={flops_classic:.0f}  ({t_classic:.2f}s)")

    # 3. BP-enhanced: include messages in belief
    prog_belief2 = make_program(
        "def f_belief(cum_dist, m_down, m_up):\n    return cum_dist + m_down + m_up\n",
        "f_belief"
    )

    t0 = time.time()
    ber_bp, flops_bp = evaluator.evaluate_genome(
        prog_down, prog_up, prog_belief2, prog_halt, data,
    )
    t_bp = time.time() - t0
    print(f"BP-enhanced BER: {ber_bp:.6f}  flops={flops_bp:.0f}  ({t_bp:.2f}s)")

    # 4. Subtract-based belief (cum_dist - m_up tends to prefer shorter paths)
    prog_belief3 = make_program(
        "def f_belief(cum_dist, m_down, m_up):\n    return cum_dist - m_up\n",
        "f_belief"
    )

    t0 = time.time()
    ber_sub, flops_sub = evaluator.evaluate_genome(
        prog_down, prog_up, prog_belief3, prog_halt, data,
    )
    t_sub = time.time() - t0
    print(f"Sub-based BER:  {ber_sub:.6f}  flops={flops_sub:.0f}  ({t_sub:.2f}s)")

    print()
    print(f"Target BER: ~0.002 at {snr_db}dB")


if __name__ == "__main__":
    main()
