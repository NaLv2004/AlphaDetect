"""S1 hard test: every core algorithm in the IR pool must expose at
least one resolvable, slot-evolvable region.

This is the regression that catches the regression where
``compile_slot_default`` silently swallowed exceptions, leaving
slot_populations empty and the corresponding descriptors phantom.

Per S1 plan: kbest, bp, soft_sic, turbo_linear, particle_filter,
importance_sampling are the previously-broken algorithms; all six MUST
now have at least one slot pop with a resolvable region. The legacy
healthy detectors (lmmse, zf, osic, ep, amp, sa, mh) are also asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from evolution.ir_pool import build_ir_pool
from evolution.gp.region_resolver import resolve_slot_region


# Algorithms that S1 explicitly re-enabled. If any of these regress to
# an empty slot_populations dict, this test will fail loudly.
S1_REPAIRED_ALGOS = (
    "kbest",
    "bp",
    "soft_sic",
    "turbo_linear",
    "particle_filter",
    "importance_sampling",
)

# Algorithms that have always been healthy — sanity-only.
ALWAYS_HEALTHY_ALGOS = ("lmmse", "zf", "osic", "ep", "amp", "sa", "mh")


@pytest.fixture(scope="module")
def pool():
    return build_ir_pool(np.random.default_rng(42))


@pytest.mark.parametrize("algo_id", S1_REPAIRED_ALGOS + ALWAYS_HEALTHY_ALGOS)
def test_algorithm_in_pool(pool, algo_id):
    g = next((x for x in pool if x.algo_id == algo_id), None)
    assert g is not None, f"{algo_id} missing from IR pool"


@pytest.mark.parametrize("algo_id", S1_REPAIRED_ALGOS)
def test_s1_repaired_algo_has_resolvable_slot(pool, algo_id):
    g = next((x for x in pool if x.algo_id == algo_id), None)
    assert g is not None, f"{algo_id} missing"
    assert len(g.slot_populations) > 0, (
        f"{algo_id} has zero slot_populations after S1 — regression in "
        f"compile_slot_default / EXTENDED_SLOT_DEFAULTS / FII inliner"
    )
    n_resolvable = 0
    for slot_key in g.slot_populations:
        info = resolve_slot_region(g, slot_key)
        if info is not None and len(info.op_ids) > 0:
            n_resolvable += 1
    assert n_resolvable > 0, (
        f"{algo_id}: no slot in {list(g.slot_populations.keys())} resolves "
        f"to a non-empty region — micro-evolution would be a no-op"
    )
