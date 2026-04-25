"""MicroPopulation — μ+λ typed-GP loop for a single slot population.

Phase H+4 §7.

Drop-in replacement for the body of
``evolution.slot_evolution.step_slot_population``. The legacy function
remains as a thin shim that constructs a ``MicroPopulation`` and runs
one generation; existing callers (algorithm_engine._micro_evolve and
train_gnn) need no API changes.

Differences from the legacy step:

  * Children are produced by typed operators from
    ``OPERATOR_REGISTRY`` (weighted random selection), not by a single
    constant-perturbation function.
  * Each proposal flows through ``run_operator_with_gates`` (gates 1/2/3/7)
    BEFORE being evaluated. No-op IRs and over-complex children never
    reach the subprocess.
  * Per-operator stats are tracked in ``OperatorStats`` and aggregated
    into ``SlotMicroStats``.

Single-representation principle: every step here works on FunctionIR.
Python source only appears inside ``evaluate_slot_variant``, which is
the *evaluation boundary*.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from algorithm_ir.ir.model import FunctionIR

from evolution.gp.canonical_hash import canonical_ir_hash
from evolution.gp.contract import SlotContract, TypedPort, build_slot_contract
from evolution.gp.operators.base import (
    GPContext,
    OPERATOR_REGISTRY,
    OperatorStats,
    run_operator_with_gates,
)
from evolution.gp.region_resolver import resolve_slot_region

if TYPE_CHECKING:
    from evolution.pool_types import AlgorithmGenome, SlotPopulation
    from evolution.slot_evolution import SlotMicroStats

logger = logging.getLogger(__name__)


@dataclass
class _OperatorPick:
    name: str
    weight: float
    is_crossover: bool = False


def _build_operator_pool() -> list[_OperatorPick]:
    pool: list[_OperatorPick] = []
    for name, (_factory, weight, is_xover) in OPERATOR_REGISTRY.items():
        if weight > 0:
            pool.append(_OperatorPick(name=name, weight=weight, is_crossover=bool(is_xover)))
    return pool


def _weighted_pick(rng: np.random.Generator, pool: list[_OperatorPick]) -> _OperatorPick:
    weights = np.array([p.weight for p in pool], dtype=float)
    total = weights.sum()
    if total <= 0:
        return pool[int(rng.integers(0, len(pool)))]
    p = weights / total
    idx = int(rng.choice(len(pool), p=p))
    return pool[idx]


def _make_contract_from_region(slot_key: str,
                               complexity_cap: int) -> SlotContract:
    """Deprecated stub kept only for back-compat imports.

    R4: real contracts are derived from ``SlotPopulation.spec`` via
    :func:`evolution.gp.contract.build_slot_contract`. This wrapper
    survives only because a handful of tests imported it directly; it
    now emits the same minimal contract shape but using the
    ``type_lattice`` ``TYPE_TOP`` token so the meaning is explicit.
    """
    from algorithm_ir.ir.type_lattice import TYPE_TOP
    short = slot_key.split(".")[-1]
    return SlotContract(
        slot_key=slot_key,
        short_name=short,
        input_ports=(TypedPort("in", TYPE_TOP),),
        output_ports=(TypedPort("out", TYPE_TOP),),
        complexity_cap=complexity_cap,
    )


def micro_population_step(
    genome: "AlgorithmGenome",
    pop_key: str,
    pop: "SlotPopulation",
    *,
    evaluator: Any,
    rng: np.random.Generator,
    n_children: int = 4,
    n_trials: int = 8,
    timeout_sec: float = 1.0,
    snr_db: float = 16.0,
    max_pop_size: int = 16,
    complexity_cap: int = 4096,
) -> "SlotMicroStats":
    """Run one (μ+λ) micro-generation using the typed GP operators.

    Returns a ``SlotMicroStats`` aggregated across all operator picks.
    Per-operator counters are stashed in ``stats.per_operator`` (a
    dict, attached dynamically — the dataclass tolerates extra attrs).
    """
    # Local import to avoid circular import at module load.
    from evolution.slot_evolution import (
        SlotMicroStats,
        apply_slot_variant,
        evaluate_slot_variant,
    )

    stats = SlotMicroStats(slot_pop_key=pop_key)
    per_operator: dict[str, OperatorStats] = {}
    stats.per_operator = per_operator     # type: ignore[attr-defined]

    # Skip if no variants at all.
    if not pop.variants:
        stats.skipped_no_variants = 1
        return stats

    # Resolve region (and consequently sids).
    region_info = resolve_slot_region(genome, pop_key)
    if region_info is None:
        stats.skipped_no_sids = 1
        return stats

    contract = build_slot_contract(
        pop, slot_key=pop_key, complexity_cap=complexity_cap,
    )
    # NB: region_info.op_ids are op_ids in genome.ir, but the parent IRs we
    # mutate here are the slot-body variants (pop.variants[i]) which have
    # totally different op_ids. The variant IR IS the slot body, so the
    # entire variant is in scope — pass an empty frozenset to disable the
    # region filter inside operators.
    region_op_ids: frozenset[str] = frozenset()

    # Bootstrap baseline fitness for the current best.
    bi = pop.best_idx
    if bi >= len(pop.fitness):
        pop.fitness = list(pop.fitness) + [float("inf")] * (
            len(pop.variants) - len(pop.fitness)
        )
    if not np.isfinite(pop.fitness[bi]):
        baseline_ser, _ = evaluate_slot_variant(
            genome, pop_key, pop.variants[bi],
            evaluator=evaluator,
            n_trials=n_trials, timeout_sec=timeout_sec, snr_db=snr_db,
        )
        pop.fitness[bi] = baseline_ser
    stats.best_before = pop.fitness[bi]

    # Build operator pool.
    op_pool = _build_operator_pool()
    if not op_pool:
        # No operators registered — fall through to skip.
        stats.skipped_no_variants = 1
        logger.warning("micro_population_step: no operators registered in OPERATOR_REGISTRY")
        return stats

    n_parents = len(pop.variants)
    for _ in range(n_children):
        stats.n_attempted += 1

        # Tournament-of-2 parent selection biased toward better fitness.
        finite = [(i, f) for i, f in enumerate(pop.fitness) if np.isfinite(f)]
        if finite:
            i1, i2 = rng.integers(0, len(finite), size=2)
            cand = finite[i1] if finite[i1][1] <= finite[i2][1] else finite[i2]
            parent_idx = cand[0]
        else:
            parent_idx = int(rng.integers(0, n_parents))
        parent = pop.variants[parent_idx]
        if parent is None:
            # Defensive: legacy pop entries may be None placeholders.
            continue

        pick = _weighted_pick(rng, op_pool)
        op_stats = per_operator.setdefault(pick.name, OperatorStats(name=pick.name))
        factory, _, _ = OPERATOR_REGISTRY[pick.name]
        op_instance = factory()

        ctx = GPContext(
            contract=contract,
            region_op_ids=region_op_ids,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )
        parent_hash = canonical_ir_hash(parent)

        # S3: For crossover operators, supply a second parent drawn from
        # the current population (excluding the chosen parent). If the
        # population has only one variant, fall back to mutation-style
        # call (parent2_ir=None) so the operator can no-op gracefully.
        parent2_ir = None
        if pick.is_crossover and n_parents > 1:
            other_indices = [i for i in range(n_parents)
                             if i != parent_idx and pop.variants[i] is not None]
            if other_indices:
                # Tournament-of-2 among the rest, biased by fitness if finite.
                ja, jb = rng.integers(0, len(other_indices), size=2)
                ia = other_indices[int(ja)]
                ib = other_indices[int(jb)]
                fa, fb = pop.fitness[ia], pop.fitness[ib]
                if np.isfinite(fa) and np.isfinite(fb):
                    pick2 = ia if fa <= fb else ib
                else:
                    pick2 = ia if np.isfinite(fa) else ib
                parent2_ir = pop.variants[pick2]

        result = run_operator_with_gates(
            op_instance, ctx, parent, parent_hash,
            parent2_ir=parent2_ir, stats=op_stats,
        )
        if not result.accepted_structurally or result.child_ir is None:
            # Logged to op_stats already; account in coarse stats so the
            # train_gnn telemetry doesn't show silent failures.
            stats.n_validate_failed += 1
            continue

        child = result.child_ir

        # Sanity: child arg signature must match parent.
        if list(child.arg_values) != list(parent.arg_values):
            op_stats.n_validate_rejected += 1
            continue

        # Splice + validate (gate inside apply_slot_variant).
        flat_ir = apply_slot_variant(genome, pop_key, child, stats=stats)
        if flat_ir is None:
            stats.n_apply_failed += 1
            continue
        stats.n_validated += 1

        # Evaluate via evaluator boundary (this is where source emerges).
        ser, _src = evaluate_slot_variant(
            genome, pop_key, child,
            evaluator=evaluator,
            n_trials=n_trials, timeout_sec=timeout_sec, snr_db=snr_db,
            stats=stats,
        )
        if not np.isfinite(ser) or ser >= 1.0:
            stats.n_eval_failed += 1
            op_stats.n_probe_rejected += 1
            # Record the failed variant so we don't keep regenerating it.
            pop.variants.append(child)
            pop.fitness.append(float("inf"))
            continue

        # R6: behavior signature gate. Two structurally-distinct IRs
        # can still produce identical decoded symbols (synonyms). The
        # SER-equality fallback (abs(ser - parent_ser) < 1e-9) is fine
        # when the evaluator does not expose decoded x_hat. When the
        # evaluator DOES support `evaluate_source_returning_xhat`, we
        # use sha1(x_hat.tobytes()) as a much stronger behavior key:
        # genuine behavior change flips at least one symbol -> hash
        # mismatch, even if the SER tie-rounds to the same float.
        parent_ser = pop.fitness[parent_idx]
        is_noop_behavior = False
        eval_xhat = getattr(evaluator, "evaluate_source_returning_xhat", None)
        emit_src = getattr(evaluator, "_emit_source_for_ir", None)  # optional helper
        if eval_xhat is not None:
            try:
                # Materialise child + parent through the same probe,
                # compare hashed x_hat. Source emission is owned by the
                # evaluator (S5 contract) — we only pass IR via emit
                # helper if available; otherwise we degrade silently.
                from algorithm_ir.regeneration.codegen import emit_python_source
                # Splice parent through apply_slot_variant for a fair
                # comparison (both children evaluated against the same
                # surrounding genome state).
                parent_flat = apply_slot_variant(genome, pop_key, parent)
                if parent_flat is not None:
                    src_parent = emit_python_source(parent_flat)
                    src_child = emit_python_source(flat_ir)
                    fname_p = (parent_flat.name or "detector")
                    fname_c = (flat_ir.name or "detector")
                    xhat_p = eval_xhat(src_parent, fname_p, snr_db=14.0)
                    xhat_c = eval_xhat(src_child, fname_c, snr_db=14.0)
                    if xhat_p is not None and xhat_c is not None:
                        import hashlib
                        h_p = hashlib.sha1(np.asarray(xhat_p, dtype=complex).tobytes()).hexdigest()
                        h_c = hashlib.sha1(np.asarray(xhat_c, dtype=complex).tobytes()).hexdigest()
                        is_noop_behavior = (h_p == h_c)
            except Exception as _exc:
                logger.debug("R6 behavior_hash probe failed: %r — falling back to SER", _exc)
                is_noop_behavior = (
                    np.isfinite(parent_ser) and abs(ser - parent_ser) < 1e-9
                )
        else:
            # SER-equality fallback (legacy path).
            is_noop_behavior = (
                np.isfinite(parent_ser) and abs(ser - parent_ser) < 1e-9
            )
        if is_noop_behavior:
            stats.n_noop_behavior += 1
            op_stats.n_noop_behavior += 1
            # Do NOT add to pop — behavior-identical variants would just
            # consume slots in the (mu+lambda) truncation step.
            continue

        stats.n_evaluated += 1
        op_stats.n_evaluated += 1
        pop.variants.append(child)
        pop.fitness.append(ser)
        if ser < stats.best_before - 1e-9:
            stats.n_improved += 1
            op_stats.n_improved += 1

    # Recompute best and truncate (same logic as legacy step).
    if pop.fitness:
        best_i, best_f = 0, pop.fitness[0]
        for i, f in enumerate(pop.fitness):
            if f < best_f:
                best_f = f
                best_i = i
        pop.best_idx = best_i
    stats.best_after = pop.fitness[pop.best_idx] if pop.fitness else float("inf")

    if len(pop.variants) > max_pop_size:
        keep = {0}
        sorted_idx = sorted(range(len(pop.variants)),
                            key=lambda i: pop.fitness[i])
        for i in sorted_idx:
            if len(keep) >= max_pop_size:
                break
            keep.add(i)
        keep_sorted = sorted(keep)
        pop.variants = [pop.variants[i] for i in keep_sorted]
        pop.fitness = [pop.fitness[i] for i in keep_sorted]
        best_i = min(range(len(pop.fitness)), key=lambda i: pop.fitness[i])
        pop.best_idx = best_i
        stats.best_after = pop.fitness[best_i]

    return stats
