"""Algorithm pool builder.

Constructs a multi-granularity initial pool of AlgorithmEntry objects,
covering L0 (primitives), L1 (composites), L2 (modules), and L3
(complete detectors).

The pool is the seed for the two-level evolution engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evolution.pool_types import (
    AlgorithmEntry,
    AlgorithmGenome,
    SlotDescriptor,
    SlotPopulation,
)
from evolution.skeleton_registry import ProgramSpec
from evolution.fitness import FitnessResult

# Import all level implementations
from evolution.pool_ops_l0 import PRIMITIVE_REGISTRY
from evolution.pool_ops_l1 import (
    regularized_solve, whitening_transform, matched_filter,
    symbol_distance, cumulative_metric, log_likelihood_distance,
    linear_equalize,
    moment_match, cavity_distribution, kl_projection,
)
from evolution.pool_ops_l2 import (
    expand_node, frontier_scoring, prune_kbest, best_first_step,
    full_bp_sweep, message_up, message_down, gaussian_bp_message,
    ep_site_update, amp_iteration_step, sic_detect_one,
    fixed_point_iterate,
)
from evolution.pool_ops_l3 import (
    lmmse_detector, zf_detector, osic_detector, kbest_detector,
    stack_detector, bp_detector, ep_detector, amp_detector,
    DETECTOR_REGISTRY,
)


# ═══════════════════════════════════════════════════════════════════════════
# Slot definitions per algorithm
# ═══════════════════════════════════════════════════════════════════════════

def _lmmse_slots() -> dict[str, SlotDescriptor]:
    return {
        "lmmse.regularizer": SlotDescriptor(
            slot_id="lmmse.regularizer",
            short_name="regularizer",
            level=1, depth=0,
            parent_slot_id=None,
            spec=ProgramSpec(
                name="regularizer",
                param_names=["G", "sigma2"],
                param_types=["mat", "float"],
                return_type="mat",
            ),
            description="Regularisation of Gram matrix G + λI",
            domain_tags={"linear_algebra"},
        ),
        "lmmse.hard_decision": SlotDescriptor(
            slot_id="lmmse.hard_decision",
            short_name="hard_decision",
            level=0, depth=0,
            parent_slot_id=None,
            spec=ProgramSpec(
                name="hard_decision",
                param_names=["x_soft", "constellation"],
                param_types=["vec_cx", "vec_cx"],
                return_type="vec_cx",
            ),
            description="Map soft estimates to constellation points",
            domain_tags={"distance"},
        ),
    }


def _zf_slots() -> dict[str, SlotDescriptor]:
    return {
        "zf.hard_decision": SlotDescriptor(
            slot_id="zf.hard_decision",
            short_name="hard_decision",
            level=0, depth=0,
            parent_slot_id=None,
            spec=ProgramSpec(
                name="hard_decision",
                param_names=["x_soft", "constellation"],
                param_types=["vec_cx", "vec_cx"],
                return_type="vec_cx",
            ),
            description="Map soft estimates to constellation points",
            domain_tags={"distance"},
        ),
    }


def _osic_slots() -> dict[str, SlotDescriptor]:
    return {
        "osic.ordering": SlotDescriptor(
            slot_id="osic.ordering",
            short_name="ordering",
            level=2, depth=0,
            parent_slot_id=None,
            spec=ProgramSpec(
                name="ordering",
                param_names=["H", "y", "sigma2"],
                param_types=["mat", "vec_cx", "float"],
                return_type="list_int",
            ),
            description="Layer detection ordering strategy",
            domain_tags={"sic"},
        ),
        "osic.sic_step": SlotDescriptor(
            slot_id="osic.sic_step",
            short_name="sic_step",
            level=2, depth=0,
            parent_slot_id=None,
            child_slot_ids=[
                "osic.sic_step.detect",
                "osic.sic_step.hard_decision",
                "osic.sic_step.cancel",
            ],
            spec=ProgramSpec(
                name="sic_step",
                param_names=["H", "y", "sigma2", "idx", "constellation"],
                param_types=["mat", "vec_cx", "float", "int", "vec_cx"],
                return_type="tuple",
            ),
            description="One SIC detection + cancellation step",
            domain_tags={"sic"},
        ),
        "osic.sic_step.detect": SlotDescriptor(
            slot_id="osic.sic_step.detect",
            short_name="detect",
            level=1, depth=1,
            parent_slot_id="osic.sic_step",
            spec=ProgramSpec(
                name="detect",
                param_names=["H", "y", "sigma2", "idx"],
                param_types=["mat", "vec_cx", "float", "int"],
                return_type="cx",
            ),
            description="Single-layer detection (default: MMSE equalize)",
            domain_tags={"filtering"},
        ),
        "osic.sic_step.hard_decision": SlotDescriptor(
            slot_id="osic.sic_step.hard_decision",
            short_name="hard_decision",
            level=0, depth=1,
            parent_slot_id="osic.sic_step",
            spec=ProgramSpec(
                name="hard_decision",
                param_names=["x_layer", "constellation"],
                param_types=["cx", "vec_cx"],
                return_type="cx",
            ),
            description="Hard decision on one layer",
            domain_tags={"distance"},
        ),
        "osic.sic_step.cancel": SlotDescriptor(
            slot_id="osic.sic_step.cancel",
            short_name="cancel",
            level=0, depth=1,
            parent_slot_id="osic.sic_step",
            spec=ProgramSpec(
                name="cancel",
                param_names=["y", "H", "idx", "x_hard"],
                param_types=["vec_cx", "mat", "int", "cx"],
                return_type="tuple",
            ),
            description="Subtract detected layer from received signal",
            domain_tags={"sic"},
        ),
    }


def _kbest_slots() -> dict[str, SlotDescriptor]:
    return {
        "kbest.expand": SlotDescriptor(
            slot_id="kbest.expand",
            short_name="expand",
            level=2, depth=0,
            parent_slot_id=None,
            child_slot_ids=[
                "kbest.expand.local_cost",
                "kbest.expand.cumulative_cost",
            ],
            spec=ProgramSpec(
                name="expand",
                param_names=["node", "y_tilde", "R", "constellation"],
                param_types=["TreeNode", "vec_cx", "mat", "vec_cx"],
                return_type="list_TreeNode",
            ),
            description="Expand a tree node at one level",
            domain_tags={"tree_search"},
        ),
        "kbest.expand.local_cost": SlotDescriptor(
            slot_id="kbest.expand.local_cost",
            short_name="local_cost",
            level=1, depth=1,
            parent_slot_id="kbest.expand",
            child_slot_ids=[
                "kbest.expand.local_cost.residual",
                "kbest.expand.local_cost.distance",
            ],
            spec=ProgramSpec(
                name="local_cost",
                param_names=["y_k", "r_kk", "interf", "symbol"],
                param_types=["cx", "cx", "cx", "cx"],
                return_type="float",
            ),
            description="Local cost at one tree level",
            domain_tags={"distance"},
        ),
        "kbest.expand.local_cost.residual": SlotDescriptor(
            slot_id="kbest.expand.local_cost.residual",
            short_name="residual",
            level=0, depth=2,
            parent_slot_id="kbest.expand.local_cost",
            spec=ProgramSpec(
                name="residual",
                param_names=["y_k", "r_kk", "interf", "sym"],
                param_types=["cx", "cx", "cx", "cx"],
                return_type="cx",
            ),
            description="Residual computation: y - r*sym - interf",
            domain_tags={"distance"},
        ),
        "kbest.expand.local_cost.distance": SlotDescriptor(
            slot_id="kbest.expand.local_cost.distance",
            short_name="distance",
            level=0, depth=2,
            parent_slot_id="kbest.expand.local_cost",
            spec=ProgramSpec(
                name="distance",
                param_names=["residual"],
                param_types=["cx"],
                return_type="float",
            ),
            description="Distance metric: |r|²",
            domain_tags={"distance"},
        ),
        "kbest.expand.cumulative_cost": SlotDescriptor(
            slot_id="kbest.expand.cumulative_cost",
            short_name="cumulative_cost",
            level=1, depth=1,
            parent_slot_id="kbest.expand",
            spec=ProgramSpec(
                name="cumulative_cost",
                param_names=["parent_cost", "local_cost"],
                param_types=["float", "float"],
                return_type="float",
            ),
            description="Cumulative path cost",
            domain_tags={"distance"},
        ),
        "kbest.prune": SlotDescriptor(
            slot_id="kbest.prune",
            short_name="prune",
            level=2, depth=0,
            parent_slot_id=None,
            child_slot_ids=["kbest.prune.ranking", "kbest.prune.selection"],
            spec=ProgramSpec(
                name="prune",
                param_names=["candidates", "K"],
                param_types=["list_TreeNode", "int"],
                return_type="list_TreeNode",
            ),
            description="Prune candidates to K-best",
            domain_tags={"tree_search"},
        ),
        "kbest.prune.ranking": SlotDescriptor(
            slot_id="kbest.prune.ranking",
            short_name="ranking",
            level=0, depth=1,
            parent_slot_id="kbest.prune",
            spec=ProgramSpec(
                name="ranking",
                param_names=["candidates"],
                param_types=["list_TreeNode"],
                return_type="list_float",
            ),
            description="Score candidates for ranking",
            domain_tags={"tree_search"},
        ),
        "kbest.prune.selection": SlotDescriptor(
            slot_id="kbest.prune.selection",
            short_name="selection",
            level=0, depth=1,
            parent_slot_id="kbest.prune",
            spec=ProgramSpec(
                name="selection",
                param_names=["sorted_candidates", "K"],
                param_types=["list_TreeNode", "int"],
                return_type="list_TreeNode",
            ),
            description="Select survivors from ranked candidates",
            domain_tags={"tree_search"},
        ),
    }


def _bp_slots() -> dict[str, SlotDescriptor]:
    return {
        "bp.bp_sweep": SlotDescriptor(
            slot_id="bp.bp_sweep",
            short_name="bp_sweep",
            level=2, depth=0,
            parent_slot_id=None,
            spec=ProgramSpec(
                name="bp_sweep",
                param_names=["H", "y", "sigma2", "Px", "constellation", "max_iters"],
                param_types=["mat", "vec_cx", "float", "mat", "vec_cx", "int"],
                return_type="tuple",
            ),
            description="Full BP message-passing sweep (probability-domain)",
            domain_tags={"message_passing"},
        ),
        "bp.final_decision": SlotDescriptor(
            slot_id="bp.final_decision",
            short_name="final_decision",
            level=0, depth=0,
            parent_slot_id=None,
            spec=ProgramSpec(
                name="final_decision",
                param_names=["mu", "constellation"],
                param_types=["vec_cx", "vec_cx"],
                return_type="vec_cx",
            ),
            description="Final hard decision from BP beliefs",
            domain_tags={"distance"},
        ),
    }


def _ep_slots() -> dict[str, SlotDescriptor]:
    return {
        "ep.cavity": SlotDescriptor(
            slot_id="ep.cavity",
            short_name="cavity",
            level=1, depth=0,
            parent_slot_id=None,
            child_slot_ids=[],
            spec=ProgramSpec(
                name="cavity",
                param_names=["t", "h2", "gamma_i", "alpha_i"],
                param_types=["cx", "float", "cx", "float"],
                return_type="tuple",
            ),
            description="Cavity distribution (pass-through in precision parameterization)",
            domain_tags={"distribution"},
        ),
        "ep.cavity.var": SlotDescriptor(
            slot_id="ep.cavity.var",
            short_name="cavity_var",
            level=0, depth=1,
            parent_slot_id="ep.cavity",
            spec=ProgramSpec(
                name="cavity_var",
                param_names=["h2", "alpha_i"],
                param_types=["float", "float"],
                return_type="float",
            ),
            description="Cavity variance computation",
            domain_tags={"distribution"},
        ),
        "ep.site_update": SlotDescriptor(
            slot_id="ep.site_update",
            short_name="site_update",
            level=2, depth=0,
            parent_slot_id=None,
            child_slot_ids=[],
            spec=ProgramSpec(
                name="site_update",
                param_names=["t", "h2", "const", "gamma_i", "alpha_i"],
                param_types=["cx", "float", "vec_cx", "cx", "float"],
                return_type="tuple",
            ),
            description="EP site approximation update (precision domain)",
            domain_tags={"inference"},
        ),
        "ep.final_decision": SlotDescriptor(
            slot_id="ep.final_decision",
            short_name="final_decision",
            level=0, depth=0,
            parent_slot_id=None,
            spec=ProgramSpec(
                name="final_decision",
                param_names=["mu", "constellation"],
                param_types=["vec_cx", "vec_cx"],
                return_type="vec_cx",
            ),
            description="Final hard decision",
            domain_tags={"distance"},
        ),
    }


def _amp_slots() -> dict[str, SlotDescriptor]:
    return {
        "amp.amp_iterate": SlotDescriptor(
            slot_id="amp.amp_iterate",
            short_name="amp_iterate",
            level=2, depth=0,
            parent_slot_id=None,
            child_slot_ids=[
                "amp.iterate.residual",
                "amp.iterate.onsager",
                "amp.iterate.effective_obs",
                "amp.iterate.denoiser",
                "amp.iterate.divergence",
            ],
            spec=ProgramSpec(
                name="amp_iterate",
                param_names=["G", "Gtilde", "g_scale", "gtilde", "yMFtilde", "sigma2", "x", "tau_s", "z", "const"],
                param_types=["mat", "mat", "vec_f", "vec_f", "vec_cx", "float", "vec_cx", "vec_f", "vec_cx", "vec_cx"],
                return_type="tuple",
            ),
            description="One AMP iteration step (LAMA-style)",
            domain_tags={"inference"},
        ),
        "amp.iterate.residual": SlotDescriptor(
            slot_id="amp.iterate.residual",
            short_name="residual",
            level=0, depth=1,
            parent_slot_id="amp.amp_iterate",
            spec=ProgramSpec(
                name="residual",
                param_names=["y", "H", "x_hat"],
                param_types=["vec_cx", "mat", "vec_cx"],
                return_type="vec_cx",
            ),
            description="Residual y - Hx",
            domain_tags={"inference"},
        ),
        "amp.iterate.onsager": SlotDescriptor(
            slot_id="amp.iterate.onsager",
            short_name="onsager",
            level=1, depth=1,
            parent_slot_id="amp.amp_iterate",
            spec=ProgramSpec(
                name="onsager",
                param_names=["z_new", "z_old", "s_hat", "Nr"],
                param_types=["vec_cx", "vec_cx", "vec_f", "int"],
                return_type="vec_cx",
            ),
            description="Onsager correction term",
            domain_tags={"inference"},
        ),
        "amp.iterate.effective_obs": SlotDescriptor(
            slot_id="amp.iterate.effective_obs",
            short_name="effective_obs",
            level=0, depth=1,
            parent_slot_id="amp.amp_iterate",
            spec=ProgramSpec(
                name="effective_obs",
                param_names=["x_hat", "H", "z"],
                param_types=["vec_cx", "mat", "vec_cx"],
                return_type="vec_cx",
            ),
            description="Effective observation x + H^H z",
            domain_tags={"inference"},
        ),
        "amp.iterate.denoiser": SlotDescriptor(
            slot_id="amp.iterate.denoiser",
            short_name="denoiser",
            level=1, depth=1,
            parent_slot_id="amp.amp_iterate",
            spec=ProgramSpec(
                name="denoiser",
                param_names=["r", "tau", "constellation"],
                param_types=["vec_cx", "float", "vec_cx"],
                return_type="tuple",
            ),
            description="MMSE denoiser η(r)",
            domain_tags={"inference", "distribution"},
        ),
        "amp.iterate.divergence": SlotDescriptor(
            slot_id="amp.iterate.divergence",
            short_name="divergence",
            level=0, depth=1,
            parent_slot_id="amp.amp_iterate",
            spec=ProgramSpec(
                name="divergence",
                param_names=["x_new", "r", "tau"],
                param_types=["vec_cx", "vec_cx", "float"],
                return_type="vec_f",
            ),
            description="Denoiser divergence for Onsager correction",
            domain_tags={"inference"},
        ),
        "amp.final_decision": SlotDescriptor(
            slot_id="amp.final_decision",
            short_name="final_decision",
            level=0, depth=0,
            parent_slot_id=None,
            spec=ProgramSpec(
                name="final_decision",
                param_names=["x_hat", "constellation"],
                param_types=["vec_cx", "vec_cx"],
                return_type="vec_cx",
            ),
            description="Final hard decision",
            domain_tags={"distance"},
        ),
    }


def _stack_slots() -> dict[str, SlotDescriptor]:
    return {
        "stack.node_select": SlotDescriptor(
            slot_id="stack.node_select",
            short_name="node_select",
            level=1, depth=0,
            parent_slot_id=None,
            spec=ProgramSpec(
                name="node_select",
                param_names=["open_set"],
                param_types=["list_TreeNode"],
                return_type="int",
            ),
            description="Priority selection from open set",
            domain_tags={"tree_search"},
        ),
        "stack.expand": SlotDescriptor(
            slot_id="stack.expand",
            short_name="expand",
            level=2, depth=0,
            parent_slot_id=None,
            spec=ProgramSpec(
                name="expand",
                param_names=["node", "y_tilde", "R", "constellation"],
                param_types=["TreeNode", "vec_cx", "mat", "vec_cx"],
                return_type="list_TreeNode",
            ),
            description="Expand one tree node",
            domain_tags={"tree_search"},
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# L2 module slot definitions
# ═══════════════════════════════════════════════════════════════════════════

def _l2_module_slots() -> dict[str, dict[str, SlotDescriptor]]:
    """Slot definitions for standalone L2 modules."""
    return {
        "mod_expand_node": {
            "mod_expand.local_cost": SlotDescriptor(
                slot_id="mod_expand.local_cost",
                short_name="local_cost",
                level=1, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="local_cost",
                    param_names=["y_k", "r_kk", "interf", "sym"],
                    param_types=["cx", "cx", "cx", "cx"],
                    return_type="float",
                ),
                domain_tags={"distance"},
            ),
            "mod_expand.cumulative_cost": SlotDescriptor(
                slot_id="mod_expand.cumulative_cost",
                short_name="cumulative_cost",
                level=1, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="cumulative_cost",
                    param_names=["parent_cost", "local_cost"],
                    param_types=["float", "float"],
                    return_type="float",
                ),
                domain_tags={"distance"},
            ),
        },
        "mod_bp_sweep": {
            "mod_bp.message_fn": SlotDescriptor(
                slot_id="mod_bp.message_fn",
                short_name="message_fn",
                level=1, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="message_fn",
                    param_names=["mu_in", "prec_in", "H_col", "sigma2"],
                    param_types=["vec_cx", "vec_f", "vec_cx", "float"],
                    return_type="tuple",
                ),
                domain_tags={"message_passing"},
            ),
            "mod_bp.belief_update": SlotDescriptor(
                slot_id="mod_bp.belief_update",
                short_name="belief_update",
                level=1, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="belief_update",
                    param_names=["mu", "var", "constellation"],
                    param_types=["cx", "float", "vec_cx"],
                    return_type="tuple",
                ),
                domain_tags={"message_passing", "distribution"},
            ),
            "mod_bp.halt": SlotDescriptor(
                slot_id="mod_bp.halt",
                short_name="halt",
                level=0, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="halt",
                    param_names=["old_mu", "new_mu"],
                    param_types=["vec_cx", "vec_cx"],
                    return_type="bool",
                ),
                domain_tags={"iterative"},
            ),
        },
        "mod_ep_site_update": {
            "mod_ep.tilted": SlotDescriptor(
                slot_id="mod_ep.tilted",
                short_name="tilted",
                level=1, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="tilted",
                    param_names=["cav_mu", "cav_var", "constellation"],
                    param_types=["cx", "float", "vec_cx"],
                    return_type="vec_f",
                ),
                domain_tags={"distribution"},
            ),
            "mod_ep.moment": SlotDescriptor(
                slot_id="mod_ep.moment",
                short_name="moment",
                level=1, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="moment",
                    param_names=["tilted", "constellation"],
                    param_types=["vec_f", "vec_cx"],
                    return_type="tuple",
                ),
                domain_tags={"distribution"},
            ),
        },
        "mod_amp_step": {
            "mod_amp.residual": SlotDescriptor(
                slot_id="mod_amp.residual",
                short_name="residual",
                level=0, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="residual",
                    param_names=["y", "H", "x_hat"],
                    param_types=["vec_cx", "mat", "vec_cx"],
                    return_type="vec_cx",
                ),
                domain_tags={"inference"},
            ),
            "mod_amp.denoiser": SlotDescriptor(
                slot_id="mod_amp.denoiser",
                short_name="denoiser",
                level=1, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="denoiser",
                    param_names=["r", "tau", "constellation"],
                    param_types=["vec_cx", "float", "vec_cx"],
                    return_type="tuple",
                ),
                domain_tags={"inference", "distribution"},
            ),
        },
        "mod_sic_step": {
            "mod_sic.detect": SlotDescriptor(
                slot_id="mod_sic.detect",
                short_name="detect",
                level=1, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="detect",
                    param_names=["H", "y", "sigma2", "idx"],
                    param_types=["mat", "vec_cx", "float", "int"],
                    return_type="cx",
                ),
                domain_tags={"filtering"},
            ),
            "mod_sic.hard_decision": SlotDescriptor(
                slot_id="mod_sic.hard_decision",
                short_name="hard_decision",
                level=0, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="hard_decision",
                    param_names=["x_layer", "constellation"],
                    param_types=["cx", "vec_cx"],
                    return_type="cx",
                ),
                domain_tags={"distance"},
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# L1 composite slot definitions
# ═══════════════════════════════════════════════════════════════════════════

def _l1_composite_slots() -> dict[str, dict[str, SlotDescriptor]]:
    """Slot definitions for L1 composites."""
    return {
        "comp_regularized_solve": {
            "comp_regsol.regularizer": SlotDescriptor(
                slot_id="comp_regsol.regularizer",
                short_name="regularizer",
                level=0, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="regularizer",
                    param_names=["G", "sigma2"],
                    param_types=["mat", "float"],
                    return_type="mat",
                ),
                domain_tags={"linear_algebra"},
            ),
        },
        "comp_symbol_distance": {
            "comp_symdist.residual": SlotDescriptor(
                slot_id="comp_symdist.residual",
                short_name="residual",
                level=0, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="residual",
                    param_names=["y_k", "r_kk", "interf", "sym"],
                    param_types=["cx", "cx", "cx", "cx"],
                    return_type="cx",
                ),
                domain_tags={"distance"},
            ),
            "comp_symdist.distance": SlotDescriptor(
                slot_id="comp_symdist.distance",
                short_name="distance",
                level=0, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="distance",
                    param_names=["residual"],
                    param_types=["cx"],
                    return_type="float",
                ),
                domain_tags={"distance"},
            ),
        },
        "comp_moment_match": {
            "comp_mm.mean": SlotDescriptor(
                slot_id="comp_mm.mean",
                short_name="mean",
                level=0, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="mean",
                    param_names=["weights", "points"],
                    param_types=["vec_f", "vec_cx"],
                    return_type="cx",
                ),
                domain_tags={"distribution"},
            ),
            "comp_mm.var": SlotDescriptor(
                slot_id="comp_mm.var",
                short_name="var",
                level=0, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="var",
                    param_names=["weights", "points", "mean"],
                    param_types=["vec_f", "vec_cx", "cx"],
                    return_type="float",
                ),
                domain_tags={"distribution"},
            ),
        },
        "comp_cavity_dist": {
            "comp_cav.var": SlotDescriptor(
                slot_id="comp_cav.var",
                short_name="cavity_var",
                level=0, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="cavity_var",
                    param_names=["g_var", "s_var"],
                    param_types=["float", "float"],
                    return_type="float",
                ),
                domain_tags={"distribution"},
            ),
            "comp_cav.mu": SlotDescriptor(
                slot_id="comp_cav.mu",
                short_name="cavity_mu",
                level=0, depth=0,
                parent_slot_id=None,
                spec=ProgramSpec(
                    name="cavity_mu",
                    param_names=["g_mu", "g_var", "s_mu", "s_var"],
                    param_types=["cx", "float", "cx", "float"],
                    return_type="cx",
                ),
                domain_tags={"distribution"},
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pool builder
# ═══════════════════════════════════════════════════════════════════════════

def build_initial_pool() -> list[AlgorithmEntry]:
    """Build the multi-granularity initial algorithm pool.

    Returns ~30 AlgorithmEntry objects covering:
      - 8 Level-3 complete detectors
      - 12 Level-2 algorithm modules
      - 10 Level-1 composite operations

    Level-0 primitives are registered in PRIMITIVE_REGISTRY and accessed
    via random_program_for_slot, not as AlgorithmEntry objects.
    """
    pool: list[AlgorithmEntry] = []

    # ── L3: Complete detectors ────────────────────────────────────────────
    l3_entries = [
        ("lmmse", 3, {"original", "linear"}, _lmmse_slots()),
        ("zf", 3, {"original", "linear"}, _zf_slots()),
        ("osic", 3, {"original", "sic"}, _osic_slots()),
        ("kbest", 3, {"original", "tree_search"}, _kbest_slots()),
        ("stack", 3, {"original", "tree_search"}, _stack_slots()),
        ("bp", 3, {"original", "message_passing"}, _bp_slots()),
        ("ep", 3, {"original", "approximate_inference"}, _ep_slots()),
        ("amp", 3, {"original", "approximate_inference"}, _amp_slots()),
    ]

    for algo_id, level, tags, slot_tree in l3_entries:
        pool.append(AlgorithmEntry(
            algo_id=algo_id,
            ir=None,       # IR will be compiled on demand
            source=None,
            level=level,
            tags=tags,
            slot_tree=slot_tree,
        ))

    # ── L2: Algorithm modules ────────────────────────────────────────────
    l2_module_slot_defs = _l2_module_slots()
    l2_entries = [
        ("mod_expand_node", 2, {"original", "tree_search"}),
        ("mod_frontier_score", 2, {"original", "tree_search"}),
        ("mod_prune_kbest", 2, {"original", "tree_search"}),
        ("mod_best_first_step", 2, {"original", "tree_search"}),
        ("mod_bp_sweep", 2, {"original", "message_passing"}),
        ("mod_message_up", 2, {"original", "message_passing"}),
        ("mod_message_down", 2, {"original", "message_passing"}),
        ("mod_gaussian_bp_msg", 2, {"original", "message_passing"}),
        ("mod_ep_site_update", 2, {"original", "inference"}),
        ("mod_amp_step", 2, {"original", "inference"}),
        ("mod_sic_step", 2, {"original", "sic"}),
        ("mod_fixed_point", 2, {"original", "iterative"}),
    ]

    for algo_id, level, tags in l2_entries:
        slot_tree = l2_module_slot_defs.get(algo_id, {})
        pool.append(AlgorithmEntry(
            algo_id=algo_id,
            ir=None,
            source=None,
            level=level,
            tags=tags,
            slot_tree=slot_tree if slot_tree else None,
        ))

    # ── L1: Composite operations ──────────────────────────────────────────
    l1_composite_slot_defs = _l1_composite_slots()
    l1_entries = [
        ("comp_regularized_solve", 1, {"original", "linear_algebra"}),
        ("comp_symbol_distance", 1, {"original", "distance"}),
        ("comp_cumulative_metric", 1, {"original", "distance"}),
        ("comp_linear_equalize", 1, {"original", "filtering"}),
        ("comp_matched_filter", 1, {"original", "filtering"}),
        ("comp_moment_match", 1, {"original", "distribution"}),
        ("comp_cavity_dist", 1, {"original", "distribution"}),
        ("comp_kl_projection", 1, {"original", "distribution"}),
        ("comp_log_likelihood", 1, {"original", "distance"}),
        ("comp_whitening", 1, {"original", "linear_algebra"}),
    ]

    for algo_id, level, tags in l1_entries:
        slot_tree = l1_composite_slot_defs.get(algo_id, {})
        pool.append(AlgorithmEntry(
            algo_id=algo_id,
            ir=None,
            source=None,
            level=level,
            tags=tags,
            slot_tree=slot_tree if slot_tree else None,
        ))

    return pool


# ═══════════════════════════════════════════════════════════════════════════
# Query helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_entries_by_level(pool: list[AlgorithmEntry], level: int) -> list[AlgorithmEntry]:
    """Filter pool entries by level."""
    return [e for e in pool if e.level == level]


def get_entries_by_tag(pool: list[AlgorithmEntry], tag: str) -> list[AlgorithmEntry]:
    """Filter pool entries that have a specific tag."""
    return [e for e in pool if tag in e.tags]


def get_all_slots(pool: list[AlgorithmEntry]) -> dict[str, SlotDescriptor]:
    """Collect all SlotDescriptors from all pool entries."""
    result = {}
    for entry in pool:
        if entry.slot_tree:
            result.update(entry.slot_tree)
    return result


def get_slot_hierarchy(entry: AlgorithmEntry) -> dict[int, list[str]]:
    """Group an entry's slots by depth."""
    if not entry.slot_tree:
        return {}
    by_depth: dict[int, list[str]] = {}
    for sid, desc in entry.slot_tree.items():
        by_depth.setdefault(desc.depth, []).append(sid)
    return by_depth


def count_total_slots(pool: list[AlgorithmEntry]) -> int:
    """Count total number of unique slots across the entire pool."""
    return len(get_all_slots(pool))
