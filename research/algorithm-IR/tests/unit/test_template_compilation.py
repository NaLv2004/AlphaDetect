"""M2 acceptance — every L3 detector template (annotation-only) compiles
to a FunctionIR whose ``slot_meta`` matches the slot definition table,
passes the validator, and runs end-to-end with BER == 0 on a benign
4×4 QPSK channel.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from algorithm_ir.frontend.slot_dsl import slot
from algorithm_ir.ir.validator import validate_function_ir
from evolution.ir_pool import _DETECTOR_SPECS, _template_globals, compile_detector_template


@pytest.mark.parametrize("spec", _DETECTOR_SPECS, ids=lambda s: s.algo_id)
def test_template_compiles_with_slot_meta(spec):
    ir = compile_detector_template(spec)
    declared = set(spec.slot_defs_fn().keys())
    annotated = set(ir.slot_meta.keys())
    assert annotated, f"{spec.algo_id}: no slot_meta entries"
    # M2 only annotates 2nd-tier slots; tertiary leaves in slot_defs are
    # allowed to be absent from slot_meta. The reverse must hold strictly:
    # every annotated pop_key MUST be a declared slot.
    extra = annotated - declared
    assert not extra, (
        f"{spec.algo_id}: slot_meta has undeclared keys {sorted(extra)}"
    )
    assert validate_function_ir(ir) == [], (
        f"{spec.algo_id}: validator reported errors"
    )
    for pop_key, meta in ir.slot_meta.items():
        assert meta.op_ids, f"{spec.algo_id}/{pop_key}: empty op_ids"
        assert meta.inputs, f"{spec.algo_id}/{pop_key}: no declared inputs"
        assert meta.outputs, f"{spec.algo_id}/{pop_key}: no declared outputs"


@pytest.mark.parametrize("spec", _DETECTOR_SPECS, ids=lambda s: s.algo_id)
def test_template_runtime_recovers_qpsk_at_high_snr(spec):
    """The DSL stub turns ``with slot(...)`` into a no-op CM, so the
    annotated template must produce BER == 0 at 20 dB on 4×4 QPSK."""
    rng = np.random.default_rng(0)
    Nt, Nr = 4, 4
    H = (rng.standard_normal((Nr, Nt)) + 1j * rng.standard_normal((Nr, Nt))) / np.sqrt(2)
    bits = rng.integers(0, 2, size=(Nt, 2))
    sym = (1 - 2 * bits[:, 0] + 1j * (1 - 2 * bits[:, 1])) / np.sqrt(2)
    snr_db = 20.0
    sigma2 = (10 ** (-snr_db / 10)) * 0.5
    n = (rng.standard_normal(Nr) + 1j * rng.standard_normal(Nr)) * np.sqrt(sigma2)
    y = H @ sym + n

    g = dict(_template_globals())
    if spec.extra_globals:
        g.update(spec.extra_globals)
    g.update({"np": np, "slot": slot})
    if spec.algo_id == "stack":
        g.setdefault("STACK_BEAM", 4)
    constellation = np.array(
        [(1 + 1j), (1 - 1j), (-1 + 1j), (-1 - 1j)], dtype=complex
    ) / np.sqrt(2)
    exec(compile(spec.source, f"<{spec.algo_id}>", "exec"), g)
    fn = g[spec.func_name]

    out = fn(H, y, sigma2, constellation)
    assert out is not None
    out = np.asarray(out, dtype=complex).reshape(-1)
    assert out.shape == (Nt,), f"{spec.algo_id}: output shape {out.shape}"
    # Hard-decision BER vs sym
    rec_bits = np.column_stack([(out.real < 0).astype(int), (out.imag < 0).astype(int)])
    ber = float(np.mean(rec_bits != bits))
    assert ber == 0.0, f"{spec.algo_id}: BER={ber} at 20 dB (expected 0)"
