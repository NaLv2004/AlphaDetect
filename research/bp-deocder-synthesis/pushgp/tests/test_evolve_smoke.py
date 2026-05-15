"""PR7 tests: fitness eval + main_evolve smoke run."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_5g import build_parity
from pushgp.genome import Genome
from pushgp_ldpc.adapter import oms_seed_genome
from pushgp_ldpc.eval import EPS, FitnessConfig, evaluate_genome
from pushgp_ldpc.main_evolve import main as evolve_main


@pytest.fixture(scope="module")
def small_par():
    return build_parity(bgn=2, set_idx=1, zc=2)


def test_fitness_oms_better_than_random_constant(small_par):
    """The OMS seed should decisively beat a "do nothing" decoder.

    A genome whose V2C just returns L_v (no extrinsic info) and whose C2V
    returns 0 reduces to hard-decision on channel LLR.  At low SNR, this
    should have markedly worse BER than full BP.
    """
    cfg = FitnessConfig(
        par=small_par,
        snr_list=(1.0,),
        n_frames_per_snr=2,
        max_iter=4,
    )
    f_oms = evaluate_genome(oms_seed_genome(), cfg)

    # No-op genome: drop seeded inputs, push 0 → c2v always returns 0;
    # V2C just returns L_v (already on stack).  Both yield channel-only
    # decoding.
    from pushgp.program import Instruction
    no_op = Genome(
        prog_v2c=[Instruction("Float.Const0"), Instruction("Float.Add")],
        prog_c2v=[Instruction("Float.Const0")],
        log_constants=np.zeros(8),
    )
    f_noop = evaluate_genome(no_op, cfg)
    assert f_oms <= f_noop + 1e-6, f"OMS {f_oms} should be <= no-op {f_noop}"


def test_fitness_returns_finite(small_par):
    cfg = FitnessConfig(par=small_par, snr_list=(2.0, 3.0), n_frames_per_snr=2, max_iter=4)
    f = evaluate_genome(oms_seed_genome(), cfg)
    assert np.isfinite(f)
    assert f >= np.log10(EPS) - 1e-9
    assert f <= 1.0 + 1e-9


def test_fitness_handles_broken_genome(small_par):
    """Genome with empty programs returns from VM as None → adapter
    returns 0.0 for every edge → decoder reduces to channel only."""
    cfg = FitnessConfig(par=small_par, snr_list=(1.0,), n_frames_per_snr=2, max_iter=4)
    g = Genome(prog_v2c=[], prog_c2v=[], log_constants=np.zeros(8))
    f = evaluate_genome(g, cfg)
    assert np.isfinite(f)


# =============================================== Main CLI smoke


def test_main_evolve_smoke(tmp_path: Path):
    out_dir = tmp_path / "smoke"
    argv = [
        "--snr-list", "2.0",
        "--pop-size", "4",
        "--generations", "2",
        "--elitism", "1",
        "--tournament-k", "2",
        "--frames", "2",
        "--max-iter", "4",
        "--bgn", "2", "--set", "1", "--zc", "2",
        "--n-mutations", "1",
        "--max-retries", "5",
        "--seed", "0",
        "--out-dir", str(out_dir),
    ]
    res = evolve_main(argv)
    assert (out_dir / "champion.json").exists()
    assert (out_dir / "history.json").exists()
    with open(out_dir / "history.json") as f:
        meta = json.load(f)
    assert len(meta["history"]) == 2
    # Elitism guarantees monotonic non-increase of best fitness.
    bs = [g["best_fit"] for g in meta["history"]]
    for i in range(1, len(bs)):
        assert bs[i] <= bs[i - 1] + 1e-12, f"elitism broken: {bs}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
