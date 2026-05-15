"""PR6 regression: OMS seed genome + adapter ≡ default OMS callables."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_5g import (
    build_oms_context,
    build_parity,
    decode_bp,
    decode_oms_fast,
    default_c2v_oms,
    default_v2c_oms,
)
from pushgp.genome import Genome
from pushgp.validators import validate_genome
from pushgp_ldpc.adapter import (
    load_oms_seed,
    make_callables,
    oms_seed_genome,
    save_oms_seed,
)


@pytest.fixture(scope="module")
def small_par():
    return build_parity(bgn=2, set_idx=1, zc=2)  # 84x104


@pytest.fixture(scope="module")
def seed_genome():
    return oms_seed_genome()


# =============================================== Per-call equivalence


def test_oms_seed_v2c_matches_default():
    g = oms_seed_genome()
    v2c_fn, _ = make_callables(g)
    rng = np.random.default_rng(0)
    ctx = {"max_iter": 25, "offset": 0.25, "code_rate": 0.5}
    for trial in range(40):
        deg = int(rng.integers(2, 9))
        L_v = float(rng.uniform(-3.0, 3.0))
        incoming = rng.uniform(-3.0, 3.0, size=deg - 1)
        a = default_v2c_oms(L_v, incoming, deg, 0, ctx)
        b = v2c_fn(L_v, incoming, deg, 0, ctx)
        assert np.isclose(a, b, atol=1e-9, rtol=0), \
            f"trial {trial}: default={a} adapter={b} (L_v={L_v} incoming={incoming})"


def test_oms_seed_c2v_matches_default():
    g = oms_seed_genome()
    _, c2v_fn = make_callables(g)
    rng = np.random.default_rng(1)
    ctx = {"max_iter": 25, "offset": 0.25, "code_rate": 0.5}
    for trial in range(40):
        deg = int(rng.integers(3, 9))
        incoming = rng.uniform(-3.0, 3.0, size=deg - 1)
        a = default_c2v_oms(incoming, deg, 0, ctx)
        b = c2v_fn(incoming, deg, 0, ctx)
        assert np.isclose(a, b, atol=1e-9, rtol=0), \
            f"trial {trial}: default={a} adapter={b} (incoming={incoming})"


# =============================================== End-to-end equivalence


def test_decode_bp_with_oms_seed_matches_decode_oms_fast(small_par, seed_genome):
    par = small_par
    ctx = build_oms_context(par)
    v2c_fn, c2v_fn = make_callables(seed_genome)
    rng = np.random.default_rng(42)
    for trial in range(3):
        rx = rng.normal(0.0, 2.0, size=par.cols)
        ref, _ = decode_oms_fast(rx, ctx, max_iter=8, offset=0.25)
        seed, _ = decode_bp(rx, par, v2c_fn=v2c_fn, c2v_fn=c2v_fn,
                            max_iter=8, offset=0.25, return_iters=True)
        assert np.allclose(ref, seed, atol=1e-9, rtol=0), \
            f"trial {trial}: max diff {np.max(np.abs(ref - seed))}"


# =============================================== Validator passes the seed


def test_oms_seed_passes_validator(seed_genome):
    ok, why = validate_genome(seed_genome, rng=np.random.default_rng(0))
    assert ok, f"OMS seed should validate: {why}"


# =============================================== Disk roundtrip


def test_oms_seed_save_load_roundtrip(tmp_path):
    p = tmp_path / "oms.json"
    save_oms_seed(p)
    g = Genome.load(p)
    v2c_fn, c2v_fn = make_callables(g)
    ctx = {"max_iter": 25, "offset": 0.25}
    a = default_v2c_oms(0.7, np.array([0.5, -0.3, 1.1]), 4, 0, ctx)
    b = v2c_fn(0.7, np.array([0.5, -0.3, 1.1]), 4, 0, ctx)
    assert np.isclose(a, b)


def test_default_seed_file_can_be_written_and_loaded():
    """Persist `seeds/oms.json` to its canonical location and reload."""
    path = save_oms_seed()
    assert path.exists()
    g = load_oms_seed()
    assert isinstance(g, Genome)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
