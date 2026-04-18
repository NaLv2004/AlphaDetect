"""Large-scale validation of the best evolved BP detector.

Tests with 2000+ samples per SNR point, multi-SNR sweep,
and compares against LMMSE baseline.
"""
from __future__ import annotations

import sys
import pathlib
import time
import json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.regeneration.codegen import emit_cpp_ops
from applications.mimo_bp.mimo_simulation import generate_dataset, dataset_to_dict
from applications.mimo_bp.cpp_evaluator import CppBPIREvaluator


import math

def _safe_div(a, b): return a / b if abs(a) > 1e-30 else 0.0
def _safe_log(a): return math.log(max(a, 1e-30))
def _safe_sqrt(a): return math.sqrt(max(a, 0.0))

_HELPERS = {
    "__builtins__": __builtins__,
    "_safe_div": _safe_div,
    "_safe_log": _safe_log,
    "_safe_sqrt": _safe_sqrt,
    "abs": abs,
    "math": math,
}


def make_program(source: str, name: str) -> list[int]:
    ir = compile_source_to_ir(source, name, globals_dict=_HELPERS)
    return emit_cpp_ops(ir)


def validate_programs(
    prog_down_src: str,
    prog_up_src: str,
    prog_belief_src: str,
    prog_halt_src: str,
    label: str = "Evolved",
    Nt: int = 16,
    Nr: int = 16,
    mod_order: int = 16,
    snr_dbs: list[float] | None = None,
    n_samples: int = 2000,
    max_nodes: int = 500,
    max_bp_iters: int = 5,
):
    """Validate programs across multiple SNR points."""
    if snr_dbs is None:
        snr_dbs = [16.0, 18.0, 20.0, 22.0, 24.0, 26.0]

    prog_down = make_program(prog_down_src, "f_down")
    prog_up = make_program(prog_up_src, "f_up")
    prog_belief = make_program(prog_belief_src, "f_belief")
    prog_halt = make_program(prog_halt_src, "h_halt")

    evaluator = CppBPIREvaluator(
        Nt=Nt, Nr=Nr, mod_order=mod_order,
        max_nodes=max_nodes, max_bp_iters=max_bp_iters,
    )

    print(f"\n{'='*70}")
    print(f"  Validation: {label}")
    print(f"  System: {Nt}x{Nr} {'16QAM' if mod_order == 16 else 'QPSK'}")
    print(f"  Samples per SNR: {n_samples}  ({n_samples * Nt} symbols)")
    print(f"  Max nodes: {max_nodes}, Max BP iters: {max_bp_iters}")
    print(f"{'='*70}")
    print(f"  {'SNR(dB)':>8}  {'BER(evolved)':>14}  {'BER(LMMSE)':>14}  "
          f"{'Gain(dB)':>10}  {'Flops':>8}  {'Time':>6}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*8}  {'-'*6}")

    results = []
    rng = np.random.default_rng(1234)

    for snr_db in snr_dbs:
        ds = generate_dataset(n_samples, Nt, Nr, mod_order, snr_db, rng)
        data = dataset_to_dict(ds)

        t0 = time.time()
        ber_evolved, avg_flops = evaluator.evaluate_genome(
            prog_down, prog_up, prog_belief, prog_halt, data,
        )
        t_evolved = time.time() - t0

        ber_lmmse = evaluator.evaluate_lmmse(data)

        # Gain in dB: 10*log10(BER_lmmse / BER_evolved)
        if ber_evolved > 0 and ber_lmmse > 0:
            gain_db = 10 * np.log10(ber_lmmse / ber_evolved)
        elif ber_evolved == 0:
            gain_db = float('inf')
        else:
            gain_db = 0.0

        total_symbols = n_samples * Nt
        sym_errors_evolved = int(round(ber_evolved * total_symbols))
        sym_errors_lmmse = int(round(ber_lmmse * total_symbols))

        print(f"  {snr_db:8.1f}  {ber_evolved:14.6f}  {ber_lmmse:14.6f}  "
              f"{gain_db:10.2f}  {avg_flops:8.0f}  {t_evolved:5.1f}s"
              f"  ({sym_errors_evolved}/{total_symbols} vs {sym_errors_lmmse}/{total_symbols})")

        results.append({
            "snr_db": snr_db,
            "ber_evolved": ber_evolved,
            "ber_lmmse": ber_lmmse,
            "gain_db": gain_db,
            "avg_flops": avg_flops,
            "sym_errors_evolved": sym_errors_evolved,
            "sym_errors_lmmse": sym_errors_lmmse,
        })

    # Check 24dB target
    r24 = [r for r in results if r["snr_db"] == 24.0]
    if r24:
        ber24 = r24[0]["ber_evolved"]
        target = 0.002
        status = "PASS" if ber24 <= target else "FAIL"
        print(f"\n  Target BER at 24dB: {target:.4f}")
        print(f"  Achieved BER at 24dB: {ber24:.6f}  [{status}]")

    return results


def main():
    # ---- Programs from the evolution run ----

    # Best evolved (from evolution output)
    evolved_results = validate_programs(
        prog_down_src="def f_down(parent_m_down, local_dist):\n    return -abs(local_dist + local_dist)\n",
        prog_up_src="def f_up(sum_child_ld, sum_child_m_up, n_children):\n    return _safe_log(sum_child_ld)\n",
        prog_belief_src="def f_belief(cum_dist, m_down, m_up):\n    return -_safe_log(cum_dist) - -m_up\n",
        prog_halt_src="def h_halt(old_root_m_up, new_root_m_up):\n    return new_root_m_up\n",
        label="Best Evolved",
    )

    # Hand-crafted baseline: classic cum_dist scoring
    classic_results = validate_programs(
        prog_down_src="def f_down(parent_m_down, local_dist):\n    return parent_m_down + local_dist\n",
        prog_up_src="def f_up(sum_child_ld, sum_child_m_up, n_children):\n    return sum_child_ld + sum_child_m_up\n",
        prog_belief_src="def f_belief(cum_dist, m_down, m_up):\n    return cum_dist\n",
        prog_halt_src="def h_halt(old_root_m_up, new_root_m_up):\n    return 0.0\n",
        label="Classic (cum_dist)",
    )

    # Hand-crafted BP-enhanced
    bp_results = validate_programs(
        prog_down_src="def f_down(parent_m_down, local_dist):\n    return parent_m_down + local_dist\n",
        prog_up_src="def f_up(sum_child_ld, sum_child_m_up, n_children):\n    return sum_child_ld + sum_child_m_up\n",
        prog_belief_src="def f_belief(cum_dist, m_down, m_up):\n    return cum_dist + m_down + m_up\n",
        prog_halt_src="def h_halt(old_root_m_up, new_root_m_up):\n    return 0.0\n",
        label="BP-enhanced (cum+m_down+m_up)",
    )


if __name__ == "__main__":
    main()
