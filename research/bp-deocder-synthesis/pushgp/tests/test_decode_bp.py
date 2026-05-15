"""PR5 regression: decode_bp(default OMS callables) == decode_oms_fast."""

from __future__ import annotations

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_5g import (
    build_oms_context,
    build_parity,
    decode_bp,
    decode_oms,
    decode_oms_fast,
    default_c2v_oms,
    default_v2c_oms,
)


@pytest.fixture(scope="module")
def small_par():
    # BG2 set 1, Zc=2 → 84 x 104 parity matrix, fast enough for many trials.
    return build_parity(bgn=2, set_idx=1, zc=2)


def test_default_callables_basic_shape():
    incoming = np.array([0.5, -0.3, 1.1])
    out = default_v2c_oms(0.7, incoming, 4, 0, {"max_iter": 25, "offset": 0.25, "code_rate": 0.5})
    assert np.isclose(out, 0.7 + 0.5 - 0.3 + 1.1)
    out2 = default_c2v_oms(np.array([0.5, -0.3, 1.1]), 4, 0,
                           {"max_iter": 25, "offset": 0.25})
    # min(|.|) - 0.25 = 0.05; sign product = -1
    assert np.isclose(out2, -0.05)


def test_default_callables_empty_incoming():
    ctx = {"offset": 0.25}
    assert default_v2c_oms(0.7, np.array([]), 1, 0, ctx) == 0.7
    assert default_c2v_oms(np.array([]), 1, 0, ctx) == 0.0


def test_decode_bp_matches_decode_oms_fast(small_par):
    par = small_par
    ctx = build_oms_context(par)
    rng = np.random.default_rng(2024)
    for trial in range(5):
        rx = rng.normal(0.0, 2.0, size=par.cols)
        llr_a, _ = decode_oms_fast(rx, ctx, max_iter=8, offset=0.25)
        llr_b = decode_bp(rx, par, max_iter=8, offset=0.25)
        # Both follow the same flooded schedule with identical OMS update;
        # outputs must be bit-for-bit equal up to tiny float-order effects.
        assert np.allclose(llr_a, llr_b, atol=1e-9, rtol=0), \
            f"trial {trial}: decode_bp diverged from decode_oms_fast " \
            f"(max diff {np.max(np.abs(llr_a - llr_b))})"


def test_decode_bp_matches_decode_oms_slow(small_par):
    """Also check against the canonical (slower) decode_oms reference."""
    par = small_par
    rng = np.random.default_rng(7)
    rx = rng.normal(0.0, 2.0, size=par.cols)
    llr_ref, iters_ref = decode_oms(rx, par, max_iter=8, offset=0.25)
    llr_new, iters_new = decode_bp(rx, par, max_iter=8, offset=0.25, return_iters=True)
    assert np.allclose(llr_ref, llr_new, atol=1e-9, rtol=0)
    assert iters_ref == iters_new


def test_decode_bp_handles_faulty_callables(small_par):
    """Bad callables (raise / return non-finite) must not crash the loop."""
    par = small_par
    rng = np.random.default_rng(0)
    rx = rng.normal(0.0, 2.0, size=par.cols)

    def bad_v2c(L_v, incoming, deg, it, ctx):
        raise RuntimeError("boom")

    def bad_c2v(incoming, deg, it, ctx):
        return float("nan")

    out = decode_bp(rx, par, v2c_fn=bad_v2c, c2v_fn=bad_c2v, max_iter=3)
    # All extrinsic info zero → posterior == channel LLR.
    assert np.allclose(out, rx)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
