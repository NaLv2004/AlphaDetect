"""BP skeleton definitions for the MIMO BP detector.

Defines the 4 evolved programs (f_down, f_up, f_belief, h_halt) as
ProgramSpecs + SkeletonSpec for use with the evolution framework.
"""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolution.skeleton_registry import ProgramSpec, SkeletonSpec, SkeletonRegistry


def bp_skeleton() -> SkeletonRegistry:
    """Create a SkeletonRegistry with the MIMO BP detector skeleton.

    Programs:
        f_down:  (parent_m_down, local_dist) → m_down
        f_up:  (sum_child_ld, sum_child_m_up, n_children) → m_up
        f_belief:  (cum_dist, m_down, m_up) → score
        h_halt:  (old_root_m_up, new_root_m_up) → halt_flag
    """
    spec = SkeletonSpec(
        skeleton_id="mimo_bp_detector",
        program_specs=[
            ProgramSpec(
                name="f_down",
                param_names=["parent_m_down", "local_dist"],
                param_types=["float", "float"],
                return_type="float",
                constraints={"min_depth": 1, "max_depth": 6},
            ),
            ProgramSpec(
                name="f_up",
                param_names=["sum_child_ld", "sum_child_m_up", "n_children"],
                param_types=["float", "float", "float"],
                return_type="float",
                constraints={"min_depth": 1, "max_depth": 6},
            ),
            ProgramSpec(
                name="f_belief",
                param_names=["cum_dist", "m_down", "m_up"],
                param_types=["float", "float", "float"],
                return_type="float",
                constraints={"min_depth": 1, "max_depth": 6},
            ),
            ProgramSpec(
                name="h_halt",
                param_names=["old_root_m_up", "new_root_m_up"],
                param_types=["float", "float"],
                return_type="float",
                constraints={"min_depth": 1, "max_depth": 4},
            ),
        ],
        mode="explicit_slots",
    )

    registry = SkeletonRegistry()
    registry.register(spec)
    return registry
