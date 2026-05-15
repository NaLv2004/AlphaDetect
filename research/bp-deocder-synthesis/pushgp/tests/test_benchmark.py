"""PR8 tests: champion benchmark utility."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_5g import build_parity
from pushgp_ldpc.adapter import oms_seed_genome
from pushgp_ldpc.benchmark import plot_results, run_benchmark, write_csv


@pytest.fixture(scope="module")
def small_par():
    return build_parity(bgn=2, set_idx=1, zc=2)


def test_benchmark_oms_seed_matches_oms_decoder(small_par, tmp_path):
    """When champion = OMS seed genome, BER must match OMS reference exactly."""
    g = oms_seed_genome()
    res = run_benchmark(g, small_par, snr_list=[2.0, 3.0],
                        n_frames=4, max_iter=8, code_rate=0.5, seed=7)
    assert res.ber_oms == res.ber_champ, \
        f"OMS seed should reproduce OMS exactly: {res.ber_oms} vs {res.ber_champ}"
    assert res.fer_oms == res.fer_champ


def test_benchmark_csv_written(small_par, tmp_path):
    g = oms_seed_genome()
    res = run_benchmark(g, small_par, snr_list=[2.0], n_frames=2, max_iter=4)
    p = tmp_path / "out.csv"
    write_csv(res, p)
    assert p.exists()
    with open(p) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["snr_db", "ber_oms", "ber_champ", "fer_oms", "fer_champ"]
    assert len(rows) == 2  # header + 1 row


def test_benchmark_plot_optional(small_par, tmp_path):
    g = oms_seed_genome()
    res = run_benchmark(g, small_par, snr_list=[2.0, 3.0], n_frames=2, max_iter=4)
    p = tmp_path / "out.png"
    ok = plot_results(res, p)
    if ok:
        assert p.exists()
    # If matplotlib isn't installed, the function returns False and that's fine.


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
