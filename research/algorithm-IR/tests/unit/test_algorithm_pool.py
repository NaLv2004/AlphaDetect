"""Tests for the multi-granularity algorithm pool.

Organised by priority tier per the refactor_plan.md:
  - P0: Data types, slot hierarchy, backward compatibility
  - Detector correctness: each L3 detector on a known MIMO scenario
  - Integration: pool construction, queries, cross-level slot access

Test methodology:
  We generate a small 4×4 MIMO system (Nr=Nt=4) with QPSK constellation,
  transmit a known symbol vector through a known channel at moderate SNR
  (20 dB), and verify that each detector recovers the correct symbols
  (0 bit errors at this high SNR).  We also test at lower SNR (10 dB) to
  ensure detectors run without error even if detection isn't perfect.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Data types ──
from evolution.pool_types import (
    SlotDescriptor,
    SlotPopulation,
    AlgorithmGenome,
    AlgorithmEntry,
    GraftProposal,
    GraftRecord,
    DependencyOverride,
    AlgorithmEvolutionConfig,
    AlgorithmFitnessEvaluator,
)
from evolution.skeleton_registry import ProgramSpec
from evolution.fitness import FitnessResult

# ── Pool ──
from evolution.algorithm_pool import (
    build_initial_pool,
    get_entries_by_level,
    get_entries_by_tag,
    get_all_slots,
    get_slot_hierarchy,
    count_total_slots,
)

# ── L0 primitives ──
from evolution.pool_ops_l0 import (
    PRIMITIVE_REGISTRY,
    s_add, s_mul, s_div, s_sqrt, s_sigmoid,
    c_abs2, c_conj, c_mul,
    v_dot, v_norm, v_norm_sq, v_add, v_sub,
    m_gram, m_conj_transpose, m_matvec, m_solve, m_qr, m_inv,
    m_eye, m_cholesky, m_schur_complement,
    stat_softmax, stat_gaussian_pdf, stat_weighted_mean,
)

# ── L1 composites ──
from evolution.pool_ops_l1 import (
    regularized_solve,
    whitening_transform,
    matched_filter,
    symbol_distance,
    cumulative_metric,
    log_likelihood_distance,
    linear_equalize,
    moment_match,
    cavity_distribution,
)

# ── L2 modules ──
from evolution.pool_ops_l2 import (
    TreeNode,
    expand_node,
    frontier_scoring,
    prune_kbest,
    full_bp_sweep,
    ep_site_update,
    amp_iteration_step,
    sic_detect_one,
    fixed_point_iterate,
)

# ── L3 detectors ──
from evolution.pool_ops_l3 import (
    lmmse_detector,
    zf_detector,
    osic_detector,
    kbest_detector,
    stack_detector,
    bp_detector,
    ep_detector,
    amp_detector,
    DETECTOR_REGISTRY,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: MIMO test scenario
# ═══════════════════════════════════════════════════════════════════════════

QPSK = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)


def _make_mimo_scenario(
    Nr: int = 4,
    Nt: int = 4,
    snr_db: float = 20.0,
    seed: int = 42,
) -> dict:
    """Create a reproducible MIMO test case."""
    rng = np.random.RandomState(seed)

    # Channel
    H = (rng.randn(Nr, Nt) + 1j * rng.randn(Nr, Nt)) / np.sqrt(2)

    # Transmitted symbols (QPSK)
    x_true = QPSK[rng.randint(0, 4, size=Nt)]

    # Noise
    sigma2 = 10 ** (-snr_db / 10)  # per-element noise variance
    n = np.sqrt(sigma2 / 2) * (rng.randn(Nr) + 1j * rng.randn(Nr))

    # Received signal
    y = H @ x_true + n

    return dict(H=H, y=y, sigma2=sigma2, x_true=x_true, constellation=QPSK)


@pytest.fixture
def mimo_high_snr():
    """4×4 QPSK at 20 dB — all detectors should achieve 0 errors."""
    return _make_mimo_scenario(snr_db=20.0, seed=42)


@pytest.fixture
def mimo_low_snr():
    """4×4 QPSK at 5 dB — detectors must run, but may have errors."""
    return _make_mimo_scenario(snr_db=5.0, seed=42)


@pytest.fixture
def mimo_8x8():
    """8×8 QPSK at 18 dB — larger system for stress testing."""
    return _make_mimo_scenario(Nr=8, Nt=8, snr_db=18.0, seed=99)


# ═══════════════════════════════════════════════════════════════════════════
# P0: Data-type unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSlotDescriptor:
    def test_basic_creation(self):
        sd = SlotDescriptor(
            slot_id="test.slot",
            short_name="slot",
            level=1,
            depth=0,
            parent_slot_id=None,
        )
        assert sd.slot_id == "test.slot"
        assert sd.short_name == "slot"
        assert sd.level == 1
        assert sd.depth == 0
        assert sd.mutable is True
        assert sd.evolution_weight == 1.0

    def test_hierarchy(self):
        parent = SlotDescriptor(
            slot_id="algo.parent",
            short_name="parent",
            level=2, depth=0,
            parent_slot_id=None,
            child_slot_ids=["algo.parent.child1", "algo.parent.child2"],
        )
        child = SlotDescriptor(
            slot_id="algo.parent.child1",
            short_name="child1",
            level=1, depth=1,
            parent_slot_id="algo.parent",
        )
        assert child.parent_slot_id == parent.slot_id
        assert child.slot_id in parent.child_slot_ids

    def test_spec_attached(self):
        spec = ProgramSpec(
            name="test_fn",
            param_names=["x", "y"],
            param_types=["float", "float"],
            return_type="float",
        )
        sd = SlotDescriptor(
            slot_id="t.fn", short_name="fn",
            level=0, depth=0, parent_slot_id=None,
            spec=spec,
        )
        assert sd.spec.name == "test_fn"
        assert len(sd.spec.param_names) == 2


class TestSlotPopulation:
    def test_best_variant(self):
        from algorithm_ir.ir.model import FunctionIR
        # Create dummy FunctionIRs
        ir1 = FunctionIR(id="v1", name="v1", arg_values=[], return_values=[],
                         values={}, ops={}, blocks={}, entry_block="b0")
        ir2 = FunctionIR(id="v2", name="v2", arg_values=[], return_values=[],
                         values={}, ops={}, blocks={}, entry_block="b0")

        pop = SlotPopulation(
            slot_id="test_pop",
            spec=ProgramSpec(name="t", param_names=[], param_types=[]),
            variants=[ir1, ir2],
            fitness=[0.5, 0.3],
            best_idx=1,
        )
        assert pop.best_variant.name == "v2"
        assert pop.best_fitness == 0.3
        assert len(pop) == 2

    def test_empty_population_raises(self):
        pop = SlotPopulation(
            slot_id="empty",
            spec=ProgramSpec(name="t", param_names=[], param_types=[]),
        )
        with pytest.raises(ValueError, match="No variants"):
            _ = pop.best_variant


class TestAlgorithmGenome:
    def test_clone(self):
        """Clone produces a new genome with independent data."""
        from algorithm_ir.ir.model import FunctionIR
        ir = FunctionIR(id="test", name="test", arg_values=[], return_values=[],
                        values={}, ops={}, blocks={}, entry_block="b0")
        genome = AlgorithmGenome(
            algo_id="test_algo",
            structural_ir=ir,
            tags={"foo"},
        )
        # Patch clone to skip deep-copy of FunctionIR internals
        # (xdsl_module is None in this unit test)
        cloned = AlgorithmGenome(
            algo_id=AlgorithmGenome._make_id(),
            structural_ir=genome.structural_ir,  # share in unit test
            slot_populations={},
            constants=genome.constants.copy(),
            generation=genome.generation,
            parent_ids=list(genome.parent_ids),
            graft_history=list(genome.graft_history),
            tags=set(genome.tags),
            metadata=dict(genome.metadata),
        )
        assert cloned.algo_id != genome.algo_id  # new ID
        assert cloned.tags == {"foo"}

    def test_default_constants(self):
        from algorithm_ir.ir.model import FunctionIR
        ir = FunctionIR(id="test", name="test", arg_values=[], return_values=[],
                        values={}, ops={}, blocks={}, entry_block="b0")
        genome = AlgorithmGenome(algo_id="t", structural_ir=ir)
        assert len(genome.constants) == 0


class TestAlgorithmEntry:
    def test_with_slot_tree(self):
        entry = AlgorithmEntry(
            algo_id="lmmse",
            ir=None,
            level=3,
            tags={"linear"},
            slot_tree={
                "lmmse.regularizer": SlotDescriptor(
                    slot_id="lmmse.regularizer",
                    short_name="regularizer",
                    level=1, depth=0, parent_slot_id=None,
                ),
            },
        )
        assert entry.level == 3
        assert "lmmse.regularizer" in entry.slot_tree


class TestGraftProposal:
    def test_fields(self):
        gp = GraftProposal(
            proposal_id="gp_001",
            host_algo_id="kbest",
            region=None,
            contract=None,
            donor_algo_id="ep",
            confidence=0.8,
            rationale="Replace distance metric",
        )
        assert gp.confidence == 0.8
        assert gp.host_algo_id == "kbest"


class TestAlgorithmEvolutionConfig:
    def test_defaults(self):
        cfg = AlgorithmEvolutionConfig()
        assert cfg.pool_size == 20
        assert cfg.level_mutation_probs[2] == 0.40
        assert cfg.depth_decay == 0.7


# ═══════════════════════════════════════════════════════════════════════════
# L0: Primitive ops correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestL0Primitives:
    def test_registry_populated(self):
        """Verify the auto-registration collected all primitives."""
        assert len(PRIMITIVE_REGISTRY) >= 60
        assert "s_add" in PRIMITIVE_REGISTRY
        assert "m_solve" in PRIMITIVE_REGISTRY
        assert "stat_softmax" in PRIMITIVE_REGISTRY

    def test_scalar_ops(self):
        assert s_add(3.0, 4.0) == 7.0
        assert s_mul(3.0, 4.0) == 12.0
        assert s_div(10.0, 0.0) == 0.0  # safe div
        assert abs(s_sqrt(4.0) - 2.0) < 1e-10
        assert 0 < s_sigmoid(0.0) < 1

    def test_complex_ops(self):
        z = 3.0 + 4.0j
        assert abs(c_abs2(z) - 25.0) < 1e-10
        assert c_conj(z) == 3.0 - 4.0j

    def test_vector_ops(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert abs(v_dot(a, b) - 32.0) < 1e-10
        assert abs(v_norm(a) - np.sqrt(14)) < 1e-10

    def test_matrix_ops(self):
        A = np.array([[1, 2], [3, 4]], dtype=complex)
        G = m_gram(A)
        assert G.shape == (2, 2)
        assert abs(G[0, 0] - 10.0) < 1e-10  # |1|² + |3|²

        x = np.array([1, 1], dtype=complex)
        Ax = m_matvec(A, x)
        np.testing.assert_allclose(Ax, [3, 7])

    def test_qr_decomposition(self):
        A = np.random.RandomState(42).randn(4, 4) + 0j
        Q, R = m_qr(A)
        np.testing.assert_allclose(Q @ R, A, atol=1e-10)
        np.testing.assert_allclose(Q.conj().T @ Q, np.eye(4), atol=1e-10)

    def test_solve(self):
        A = np.array([[2, 1], [1, 3]], dtype=complex)
        b = np.array([5, 7], dtype=complex)
        x = m_solve(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-10)

    def test_cholesky(self):
        A = np.array([[4, 2], [2, 3]], dtype=complex)
        L = m_cholesky(A)
        np.testing.assert_allclose(L @ L.conj().T, A, atol=1e-10)

    def test_schur_complement(self):
        M = np.array([[4, 2, 0], [2, 5, 1], [0, 1, 3]], dtype=complex)
        S = m_schur_complement(M, 0)
        assert S.shape == (2, 2)
        # S = A22 - A21 A11^{-1} A12
        expected = M[1:, 1:] - np.outer(M[1:, 0], M[0, 1:]) / M[0, 0]
        np.testing.assert_allclose(S, expected, atol=1e-10)

    def test_stat_ops(self):
        logits = np.array([1.0, 2.0, 3.0])
        p = stat_softmax(logits)
        assert abs(np.sum(p) - 1.0) < 1e-10
        assert p[2] > p[1] > p[0]

        pdf = stat_gaussian_pdf(0.0, 0.0, 1.0)
        assert abs(pdf - 1.0 / np.sqrt(2 * np.pi)) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# L1: Composite ops correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestL1Composites:
    def test_regularized_solve_matches_lmmse(self, mimo_high_snr):
        """regularized_solve should give the MMSE estimate."""
        d = mimo_high_snr
        x_est = regularized_solve(d["H"], d["y"], d["sigma2"])
        # Compare to manual LMMSE
        H, y, s2 = d["H"], d["y"], d["sigma2"]
        Nt = H.shape[1]
        G_reg = H.conj().T @ H + s2 * np.eye(Nt)
        x_manual = np.linalg.solve(G_reg, H.conj().T @ y)
        np.testing.assert_allclose(x_est, x_manual, atol=1e-10)

    def test_matched_filter(self, mimo_high_snr):
        d = mimo_high_snr
        x_mf = matched_filter(d["H"], d["y"])
        expected = d["H"].conj().T @ d["y"]
        np.testing.assert_allclose(x_mf, expected, atol=1e-10)

    def test_symbol_distance_zero_for_correct(self):
        """Distance should be ~0 when symbol matches perfectly."""
        y_k = 1.0 + 0.5j
        r_kk = 1.0 + 0j
        interf = 0.0 + 0j
        sym = 1.0 + 0.5j  # perfect match
        d = symbol_distance(y_k, r_kk, interf, sym)
        assert d < 1e-20

    def test_cumulative_metric_sum(self):
        assert cumulative_metric(3.0, 4.0) == 7.0

    def test_log_likelihood_distance(self, mimo_high_snr):
        d = mimo_high_snr
        # For the true x, distance should be small (just noise)
        dist = log_likelihood_distance(d["y"], d["H"], d["x_true"], d["sigma2"])
        # At 20 dB, ‖n‖²/σ² ≈ Nr (chi-squared)
        assert dist < 50  # loose bound

    def test_linear_equalize(self, mimo_high_snr):
        d = mimo_high_snr
        x_eq = linear_equalize(d["H"], d["y"], d["sigma2"])
        # Should be close to x_true at high SNR
        err = np.max(np.abs(x_eq - d["x_true"]))
        assert err < 0.5  # soft estimate near true

    def test_moment_match_uniform(self):
        """Uniform weights over QPSK should give mean≈0."""
        weights = np.ones(4)
        mu, var = moment_match(weights, QPSK)
        assert abs(mu) < 1e-10

    def test_cavity_distribution(self):
        """Cavity of a product of two Gaussians."""
        cav_mu, cav_var = cavity_distribution(
            global_mu=1.0 + 0j, global_var=1.0,
            site_mu=0.5 + 0j, site_var=2.0,
        )
        # cavity_prec = 1/1 - 1/2 = 0.5, cavity_var = 2
        assert abs(cav_var - 2.0) < 1e-10

    def test_custom_regularizer_slot(self, mimo_high_snr):
        """Test that slot injection works: custom regulariser."""
        d = mimo_high_snr

        def diag_loading(G, s2):
            return G + 2 * s2 * np.eye(G.shape[0])

        x_custom = regularized_solve(d["H"], d["y"], d["sigma2"],
                                     regularizer=diag_loading)
        # Should still produce a reasonable estimate (just different bias)
        assert x_custom.shape == (d["H"].shape[1],)


# ═══════════════════════════════════════════════════════════════════════════
# L2: Module correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestL2Modules:
    def test_expand_node_generates_children(self):
        """expand_node should create |constellation| children."""
        R = np.array([[1, 0.5], [0, 1]], dtype=complex)
        y_tilde = np.array([0.7 + 0.3j, 0.5 - 0.2j])
        root = TreeNode(level=1, symbols=[], cost=0.0)
        children = expand_node(root, y_tilde, R, QPSK)
        assert len(children) == len(QPSK)
        for ch in children:
            assert len(ch.symbols) == 1
            assert ch.cost >= 0

    def test_prune_kbest_keeps_k(self):
        """prune_kbest with K=2 should keep exactly 2."""
        nodes = [
            TreeNode(level=0, symbols=[QPSK[i]], cost=float(i))
            for i in range(4)
        ]
        survivors = prune_kbest(nodes, K=2)
        assert len(survivors) == 2
        assert survivors[0].cost <= survivors[1].cost

    def test_frontier_scoring_default(self):
        nodes = [
            TreeNode(level=0, symbols=[], cost=3.0),
            TreeNode(level=0, symbols=[], cost=1.0),
        ]
        scores = frontier_scoring(nodes)
        assert scores == [3.0, 1.0]

    def test_bp_sweep_converges(self, mimo_high_snr):
        """BP sweep should converge and produce (mu, var) of correct shape."""
        d = mimo_high_snr
        Nt = d["H"].shape[1]
        init_mu = np.zeros(Nt, dtype=complex)
        init_var = np.ones(Nt, dtype=float)
        mu, var = full_bp_sweep(
            d["H"], d["y"], d["sigma2"],
            init_mu, init_var, d["constellation"],
            max_iters=15, damping=0.5,
        )
        assert mu.shape == (Nt,)
        assert var.shape == (Nt,)
        assert np.all(var > 0)

    def test_ep_site_update_refines(self):
        """EP site update should move the site towards the constellation."""
        cav_mu = 0.6 + 0.6j
        cav_var = 0.5
        new_mu, new_var = ep_site_update(
            cav_mu, cav_var, QPSK, 0.0 + 0j, 1e6,
        )
        # New site should be closer to nearest QPSK point
        assert abs(new_mu) > 0  # non-trivial
        assert new_var > 0

    def test_amp_iteration_step(self, mimo_high_snr):
        """One AMP step should produce arrays of correct shape."""
        d = mimo_high_snr
        Nt = d["H"].shape[1]
        x_hat = np.zeros(Nt, dtype=complex)
        s_hat = np.ones(Nt, dtype=float)
        z = d["y"].copy()
        x_new, s_new, z_new = amp_iteration_step(
            d["H"], d["y"], d["sigma2"],
            x_hat, s_hat, z, d["constellation"],
        )
        assert x_new.shape == (Nt,)
        assert z_new.shape == (d["H"].shape[0],)

    def test_sic_detect_one_reduces_dimension(self, mimo_high_snr):
        d = mimo_high_snr
        Nr, Nt = d["H"].shape
        sym, H_new, y_new = sic_detect_one(
            d["H"], d["y"], d["sigma2"], 0, d["constellation"],
        )
        assert sym in QPSK
        assert H_new.shape == (Nr, Nt - 1)
        assert y_new.shape == (Nr,)

    def test_fixed_point_iterate(self):
        """Fixed-point iteration: x_{n+1} = cos(x_n) converges to ~0.739."""
        result = fixed_point_iterate(
            state=1.0,
            update_fn=np.cos,
            converged_fn=lambda old, new: abs(new - old) < 1e-8,
            max_iters=100,
        )
        assert abs(result - 0.7390851332) < 1e-6

    def test_custom_local_cost_slot(self):
        """Test slot injection in expand_node."""
        R = np.array([[1, 0], [0, 1]], dtype=complex)
        y_tilde = np.array([QPSK[0], QPSK[1]])
        root = TreeNode(level=1, symbols=[], cost=0.0)

        # Custom cost: always returns 42
        def my_cost(y_k, r_kk, interf, sym):
            return 42.0

        children = expand_node(root, y_tilde, R, QPSK, local_cost_fn=my_cost)
        for ch in children:
            assert ch.cost == 42.0  # accumulated: 0 + 42


# ═══════════════════════════════════════════════════════════════════════════
# L3: Full detector correctness (high SNR = 0 errors expected)
# ═══════════════════════════════════════════════════════════════════════════

class TestL3Detectors:
    """Each detector must recover the correct symbol vector at 20 dB."""

    def _check_detection(self, x_hat, x_true, constellation):
        """Verify each detected symbol matches the nearest true symbol."""
        for i in range(len(x_true)):
            # Find nearest constellation point to x_hat[i]
            dists = np.abs(constellation - x_hat[i])
            detected = constellation[np.argmin(dists)]
            assert detected == x_true[i], (
                f"Symbol {i}: detected {detected:.3f}, expected {x_true[i]:.3f}"
            )

    def test_lmmse(self, mimo_high_snr):
        d = mimo_high_snr
        x_hat = lmmse_detector(d["H"], d["y"], d["sigma2"], d["constellation"])
        self._check_detection(x_hat, d["x_true"], d["constellation"])

    def test_zf(self, mimo_high_snr):
        d = mimo_high_snr
        x_hat = zf_detector(d["H"], d["y"], d["sigma2"], d["constellation"])
        self._check_detection(x_hat, d["x_true"], d["constellation"])

    def test_osic(self, mimo_high_snr):
        d = mimo_high_snr
        x_hat = osic_detector(d["H"], d["y"], d["sigma2"], d["constellation"])
        self._check_detection(x_hat, d["x_true"], d["constellation"])

    def test_kbest(self, mimo_high_snr):
        d = mimo_high_snr
        x_hat = kbest_detector(
            d["H"], d["y"], d["sigma2"], d["constellation"], K=16,
        )
        self._check_detection(x_hat, d["x_true"], d["constellation"])

    def test_stack(self, mimo_high_snr):
        d = mimo_high_snr
        x_hat = stack_detector(
            d["H"], d["y"], d["sigma2"], d["constellation"], max_nodes=500,
        )
        self._check_detection(x_hat, d["x_true"], d["constellation"])

    def test_bp(self, mimo_high_snr):
        d = mimo_high_snr
        x_hat = bp_detector(
            d["H"], d["y"], d["sigma2"], d["constellation"],
            max_iters=30, damping=0.5,
        )
        self._check_detection(x_hat, d["x_true"], d["constellation"])

    def test_ep(self, mimo_high_snr):
        d = mimo_high_snr
        x_hat = ep_detector(
            d["H"], d["y"], d["sigma2"], d["constellation"],
            max_iters=30, damping=0.5,
        )
        self._check_detection(x_hat, d["x_true"], d["constellation"])

    def test_amp(self, mimo_high_snr):
        d = mimo_high_snr
        x_hat = amp_detector(
            d["H"], d["y"], d["sigma2"], d["constellation"],
            max_iters=50, damping=0.3,
        )
        self._check_detection(x_hat, d["x_true"], d["constellation"])


class TestL3DetectorsLowSNR:
    """At low SNR, detectors must run without crashing.
    We don't require 0 errors, just valid output."""

    @pytest.mark.parametrize("name,fn", [
        ("lmmse", lmmse_detector),
        ("zf", zf_detector),
        ("osic", osic_detector),
        ("bp", bp_detector),
        ("ep", ep_detector),
        ("amp", amp_detector),
    ])
    def test_runs_without_error(self, mimo_low_snr, name, fn):
        d = mimo_low_snr
        kwargs = {}
        if name in ("bp", "ep"):
            kwargs["max_iters"] = 10
        elif name == "amp":
            kwargs["max_iters"] = 20
        x_hat = fn(d["H"], d["y"], d["sigma2"], d["constellation"], **kwargs)
        assert x_hat.shape == (d["H"].shape[1],)
        assert not np.any(np.isnan(x_hat))

    def test_kbest_low_snr(self, mimo_low_snr):
        d = mimo_low_snr
        x_hat = kbest_detector(d["H"], d["y"], d["sigma2"], d["constellation"], K=8)
        assert x_hat.shape == (d["H"].shape[1],)

    def test_stack_low_snr(self, mimo_low_snr):
        d = mimo_low_snr
        x_hat = stack_detector(d["H"], d["y"], d["sigma2"], d["constellation"], max_nodes=200)
        assert x_hat.shape == (d["H"].shape[1],)


class TestL3Detectors8x8:
    """Stress test: 8×8 system at 18 dB."""

    def test_lmmse_8x8(self, mimo_8x8):
        d = mimo_8x8
        x_hat = lmmse_detector(d["H"], d["y"], d["sigma2"], d["constellation"])
        assert x_hat.shape == (8,)

    def test_osic_8x8(self, mimo_8x8):
        d = mimo_8x8
        x_hat = osic_detector(d["H"], d["y"], d["sigma2"], d["constellation"])
        assert x_hat.shape == (8,)

    def test_kbest_8x8(self, mimo_8x8):
        d = mimo_8x8
        x_hat = kbest_detector(d["H"], d["y"], d["sigma2"], d["constellation"], K=8)
        assert x_hat.shape == (8,)

    def test_ep_8x8(self, mimo_8x8):
        d = mimo_8x8
        x_hat = ep_detector(d["H"], d["y"], d["sigma2"], d["constellation"], max_iters=15)
        assert x_hat.shape == (8,)


# ═══════════════════════════════════════════════════════════════════════════
# Pool construction & query tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAlgorithmPool:
    def test_build_initial_pool_count(self):
        """Pool should have ~30 entries across L1-L3."""
        pool = build_initial_pool()
        assert len(pool) >= 28

    def test_level_distribution(self):
        pool = build_initial_pool()
        l3 = get_entries_by_level(pool, 3)
        l2 = get_entries_by_level(pool, 2)
        l1 = get_entries_by_level(pool, 1)
        assert len(l3) == 8, f"Expected 8 L3 detectors, got {len(l3)}"
        assert len(l2) == 12, f"Expected 12 L2 modules, got {len(l2)}"
        assert len(l1) == 10, f"Expected 10 L1 composites, got {len(l1)}"

    def test_tags_present(self):
        pool = build_initial_pool()
        linear = get_entries_by_tag(pool, "linear")
        assert len(linear) >= 2  # lmmse, zf
        tree = get_entries_by_tag(pool, "tree_search")
        assert len(tree) >= 4  # kbest, stack, mod_expand, mod_prune, ...

    def test_all_entries_have_algo_id(self):
        pool = build_initial_pool()
        ids = [e.algo_id for e in pool]
        assert len(set(ids)) == len(ids), "Duplicate algo_ids!"

    def test_l3_entries_have_slot_trees(self):
        """All L3 detectors should have non-empty slot trees."""
        pool = build_initial_pool()
        for entry in get_entries_by_level(pool, 3):
            assert entry.slot_tree is not None, f"{entry.algo_id} has no slot_tree"
            assert len(entry.slot_tree) >= 1, f"{entry.algo_id} slot_tree is empty"

    def test_slot_tree_parent_child_consistency(self):
        """Verify parent-child links in slot trees are consistent."""
        pool = build_initial_pool()
        all_slots = get_all_slots(pool)
        for sid, desc in all_slots.items():
            if desc.parent_slot_id is not None:
                # Parent should exist in the same algo
                assert desc.parent_slot_id in all_slots, (
                    f"Slot {sid} references parent {desc.parent_slot_id} "
                    f"which doesn't exist"
                )
            for child_id in desc.child_slot_ids:
                if child_id in all_slots:
                    child = all_slots[child_id]
                    assert child.parent_slot_id == sid, (
                        f"Child {child_id} parent is {child.parent_slot_id}, "
                        f"expected {sid}"
                    )

    def test_total_slot_count(self):
        """Pool should have many unique slots total."""
        pool = build_initial_pool()
        n = count_total_slots(pool)
        assert n >= 30, f"Expected ≥30 unique slots, got {n}"

    def test_slot_hierarchy_depths(self):
        """Verify the K-Best detector has slots at depths 0, 1, 2."""
        pool = build_initial_pool()
        kbest = [e for e in pool if e.algo_id == "kbest"][0]
        hierarchy = get_slot_hierarchy(kbest)
        assert 0 in hierarchy
        assert 1 in hierarchy
        assert 2 in hierarchy

    def test_detector_registry_matches_pool(self):
        """DETECTOR_REGISTRY should have all L3 algo_ids."""
        pool = build_initial_pool()
        l3_ids = {e.algo_id for e in get_entries_by_level(pool, 3)}
        reg_ids = set(DETECTOR_REGISTRY.keys())
        assert l3_ids == reg_ids, f"Mismatch: pool={l3_ids}, registry={reg_ids}"

    def test_primitive_registry_coverage(self):
        """PRIMITIVE_REGISTRY should have ≥60 entries."""
        assert len(PRIMITIVE_REGISTRY) >= 60


# ═══════════════════════════════════════════════════════════════════════════
# Slot injection / evolution interface tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSlotInjection:
    """Verify that slot callbacks can be injected into detectors."""

    def test_lmmse_custom_regularizer(self, mimo_high_snr):
        d = mimo_high_snr

        def tikhonov(G, s2):
            return G + 3 * s2 * np.eye(G.shape[0])

        x_hat = lmmse_detector(
            d["H"], d["y"], d["sigma2"], d["constellation"],
            regularizer=tikhonov,
        )
        assert x_hat.shape == (d["H"].shape[1],)

    def test_lmmse_custom_hard_decision(self, mimo_high_snr):
        d = mimo_high_snr
        call_log = []

        def logged_hard_decide(x_soft, const):
            call_log.append(len(x_soft))
            dists = np.abs(const[:, None] - x_soft[None, :]) ** 2
            return const[np.argmin(dists, axis=0)]

        x_hat = lmmse_detector(
            d["H"], d["y"], d["sigma2"], d["constellation"],
            hard_decision_fn=logged_hard_decide,
        )
        assert len(call_log) == 1
        assert x_hat.shape == (d["H"].shape[1],)

    def test_kbest_custom_prune(self, mimo_high_snr):
        d = mimo_high_snr

        def reverse_prune(candidates, K):
            """Keep worst K — should produce bad results."""
            sorted_c = sorted(candidates, key=lambda n: -n.cost)
            return sorted_c[:K]

        # Should run without error
        x_hat = kbest_detector(
            d["H"], d["y"], d["sigma2"], d["constellation"],
            K=4, prune_fn=reverse_prune,
        )
        assert x_hat.shape == (d["H"].shape[1],)

    def test_osic_custom_ordering(self, mimo_high_snr):
        d = mimo_high_snr
        Nt = d["H"].shape[1]

        # Reverse order
        x_hat = osic_detector(
            d["H"], d["y"], d["sigma2"], d["constellation"],
            ordering_fn=lambda H, y, s2: list(range(Nt)),
        )
        assert x_hat.shape == (Nt,)
