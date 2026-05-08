"""Compare two snapshot npz files for detector output equivalence."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <snapshot_v0> <snapshot_v1>")
        sys.exit(2)
    p0, p1 = Path(sys.argv[1]), Path(sys.argv[2])
    d0 = np.load(p0, allow_pickle=True)
    d1 = np.load(p1, allow_pickle=True)

    algo_ids = list(d0["_algo_ids"])
    print(f"Pool: {len(algo_ids)} detectors")
    print(f"SNR : {d0['_meta_snr']} dB / trials {d0['_meta_trials']}")
    print()

    # Check inputs match
    for k in ("_inputs_H", "_inputs_y", "_inputs_x_true", "_inputs_sigma2"):
        if not np.array_equal(d0[k], d1[k]):
            print(f"!! input {k} differs between snapshots !!")
            sys.exit(2)
    print("OK: inputs identical between v0 and v1")
    print()

    # Compare per-detector
    atol = 1e-9
    rtol = 1e-7
    fully_equal = []          # bit-exact equal
    numerically_close = []    # close within (atol, rtol) but not bit-exact
    quant_equal = []          # x_quant identical (hard decisions match)
    differs = []              # quant differs

    for algo in algo_ids:
        algo = str(algo)
        k_x = f"xhat__{algo}"
        k_q = f"xquant__{algo}"
        k_e = f"errs__{algo}"
        if k_x not in d0.files or k_x not in d1.files:
            print(f"  {algo:30s} MISSING in one snapshot")
            continue
        x0, x1 = d0[k_x], d1[k_x]
        q0, q1 = d0[k_q], d1[k_q]
        e0, e1 = d0[k_e], d1[k_e]

        # bit-exact?
        bit_exact = np.array_equal(x0, x1)
        close = np.allclose(x0, x1, atol=atol, rtol=rtol, equal_nan=True)
        quant_eq = np.array_equal(q0, q1)
        err_eq = np.array_equal(e0, e1)

        max_abs_diff = float(np.max(np.abs(x0 - x1))) if x0.shape == x1.shape else float("inf")

        if bit_exact:
            fully_equal.append(algo)
            tag = "BIT-EXACT"
        elif close:
            numerically_close.append(algo)
            tag = f"CLOSE        max|d|={max_abs_diff:.2e}"
        elif quant_eq:
            quant_equal.append(algo)
            tag = f"QUANT-EQ     max|d|={max_abs_diff:.2e}"
        else:
            differs.append(algo)
            n_diff_trials = int(np.sum(np.any(q0 != q1, axis=1)))
            tag = f"DIFFERS      max|d|={max_abs_diff:.2e}  diff_trials={n_diff_trials}/{q0.shape[0]}"
        err_tag = "" if err_eq else f"  ERR_DIFF e0={int(e0.sum())} e1={int(e1.sum())}"
        print(f"  {algo:30s} {tag}{err_tag}")

    n = len(algo_ids)
    print()
    print(f"=== Summary over {n} detectors ===")
    print(f"  BIT-EXACT     : {len(fully_equal)}")
    print(f"  numeric close : {len(numerically_close)}  (atol={atol}, rtol={rtol})")
    print(f"  quant-equal   : {len(quant_equal)}        (xhat differs but hard decisions match)")
    print(f"  DIFFERS       : {len(differs)}")
    if differs:
        print(f"  Differing detectors: {differs}")

    # IR stats summary
    print()
    print(f"=== IR statistics: phi reduction ===")
    print(f"{'detector':30s} {'v0 phi':>8} {'v1 phi':>8} {'Δphi':>8} {'v0 ops':>8} {'v1 ops':>8} {'Δops':>8}")
    total_phi_v0 = total_phi_v1 = total_ops_v0 = total_ops_v1 = 0
    for algo in algo_ids:
        algo = str(algo)
        try:
            p0_phi = int(d0[f"stat__{algo}__n_phi"])
            p1_phi = int(d1[f"stat__{algo}__n_phi"])
            o0 = int(d0[f"stat__{algo}__n_ops"])
            o1 = int(d1[f"stat__{algo}__n_ops"])
        except KeyError:
            continue
        total_phi_v0 += p0_phi; total_phi_v1 += p1_phi
        total_ops_v0 += o0; total_ops_v1 += o1
    print(f"{'TOTAL':30s} {total_phi_v0:>8} {total_phi_v1:>8} {total_phi_v1-total_phi_v0:>8} "
          f"{total_ops_v0:>8} {total_ops_v1:>8} {total_ops_v1-total_ops_v0:>8}")
    print(f"  phi reduction : {(1-total_phi_v1/total_phi_v0)*100:.1f}%")
    print(f"  ops reduction : {(1-total_ops_v1/total_ops_v0)*100:.1f}%")

    # Exit code: 0 if all detectors at least quant-equal, 1 otherwise
    sys.exit(0 if not differs else 1)


if __name__ == "__main__":
    main()
