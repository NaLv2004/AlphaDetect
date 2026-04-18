"""Skeleton specification and automatic validation via IR analysis.

No hand-coded validation functions — validation is derived from
the skeleton's dependency structure using algorithm_ir.region analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from algorithm_ir.ir.model import FunctionIR


@dataclass
class ProgramSpec:
    """Specification for one evolvable program slot.

    Defines the expected signature, types, and constraints.
    Validation is automatic — no ``validation_fn`` field.
    """
    name: str
    param_names: list[str]
    param_types: list[str]
    return_type: str = "float"
    slot_regions: list[str] | None = None   # explicit region IDs, or None for auto
    constraints: dict[str, Any] = field(default_factory=dict)
    # e.g. {"min_depth": 1, "max_depth": 10, "has_loop": False}


@dataclass
class SkeletonSpec:
    """Full skeleton specification — explicit slots or auto-discover."""
    skeleton_id: str
    program_specs: list[ProgramSpec] = field(default_factory=list)
    host_ir: FunctionIR | None = None   # concrete skeleton for auto discovery
    mode: str = "explicit_slots"        # "explicit_slots" | "auto_discover"


class SkeletonRegistry:
    """Registry of skeleton specs with automatic program validation."""

    def __init__(self) -> None:
        self._specs: dict[str, SkeletonSpec] = {}
        self._program_specs: dict[str, ProgramSpec] = {}

    def register(self, spec: SkeletonSpec) -> None:
        """Register a skeleton spec."""
        self._specs[spec.skeleton_id] = spec
        for ps in spec.program_specs:
            self._program_specs[ps.name] = ps

    def get_program_spec(self, role: str) -> ProgramSpec | None:
        return self._program_specs.get(role)

    def get_skeleton_spec(self, skeleton_id: str) -> SkeletonSpec | None:
        return self._specs.get(skeleton_id)

    @property
    def roles(self) -> list[str]:
        """All registered program role names."""
        return list(self._program_specs.keys())

    def validate_program(self, role: str, func_ir: FunctionIR) -> list[str]:
        """Validate a FunctionIR against the spec for the given role.

        Returns list of violation messages (empty = valid).
        Validation is automatic — derived from IR structure analysis:
          1. Argument count matches
          2. Argument names match (from IR arg value var_names)
          3. Return type is compatible
          4. All declared params are used (backward slice from return)
          5. Depth constraints satisfied
        """
        violations: list[str] = []
        spec = self._program_specs.get(role)
        if spec is None:
            violations.append(f"No spec registered for role '{role}'")
            return violations

        # 1. Argument count
        n_args = len(func_ir.arg_values)
        n_expected = len(spec.param_names)
        if n_args != n_expected:
            violations.append(
                f"Arg count mismatch: expected {n_expected}, got {n_args}"
            )

        # 2. Argument names
        for i, vid in enumerate(func_ir.arg_values):
            v = func_ir.values[vid]
            ir_name = v.attrs.get("var_name") or v.name_hint
            if i < len(spec.param_names) and ir_name and ir_name != spec.param_names[i]:
                violations.append(
                    f"Arg {i} name mismatch: expected '{spec.param_names[i]}', "
                    f"got '{ir_name}'"
                )

        # 3. Return type check (from return op output type)
        for op in func_ir.ops.values():
            if op.opcode == "return" and op.inputs:
                ret_val = func_ir.values.get(op.inputs[0])
                if ret_val and ret_val.type_hint:
                    if spec.return_type == "bool" and ret_val.type_hint not in ("bool", "i1"):
                        violations.append(
                            f"Return type mismatch: expected bool, "
                            f"got '{ret_val.type_hint}'"
                        )
                break

        # 4. All params used (backward slice from return value)
        used_args = self._find_used_args(func_ir)
        for i, vid in enumerate(func_ir.arg_values):
            if vid not in used_args:
                if i < len(spec.param_names):
                    violations.append(
                        f"Unused parameter: '{spec.param_names[i]}'"
                    )

        # 5. Depth constraints
        max_depth = spec.constraints.get("max_depth")
        if max_depth is not None:
            depth = self._compute_ir_depth(func_ir)
            if depth > max_depth:
                violations.append(
                    f"IR depth {depth} exceeds max_depth {max_depth}"
                )

        return violations

    def validate_genome_programs(
        self, programs: dict[str, FunctionIR]
    ) -> dict[str, list[str]]:
        """Validate all programs in a genome. Returns {role: [violations]}."""
        result: dict[str, list[str]] = {}
        for role, func_ir in programs.items():
            violations = self.validate_program(role, func_ir)
            if violations:
                result[role] = violations
        return result

    @staticmethod
    def _find_used_args(func_ir: FunctionIR) -> set[str]:
        """Find which arg value IDs are transitively used by return ops."""
        # Backward slice from return values
        return_values: list[str] = []
        for op in func_ir.ops.values():
            if op.opcode == "return":
                return_values.extend(op.inputs)

        visited: set[str] = set()
        queue = list(return_values)
        while queue:
            vid = queue.pop()
            if vid in visited:
                continue
            visited.add(vid)
            v = func_ir.values.get(vid)
            if v and v.def_op:
                def_op = func_ir.ops.get(v.def_op)
                if def_op:
                    queue.extend(def_op.inputs)

        arg_set = set(func_ir.arg_values)
        return visited & arg_set

    @staticmethod
    def _compute_ir_depth(func_ir: FunctionIR) -> int:
        """Approximate IR depth as max nesting level of blocks."""
        # Simple: count block nesting via block succs
        depths: dict[str, int] = {}
        queue = [func_ir.entry_block]
        depths[func_ir.entry_block] = 0
        max_d = 0
        visited = set()
        while queue:
            bid = queue.pop(0)
            if bid in visited:
                continue
            visited.add(bid)
            d = depths[bid]
            max_d = max(max_d, d)
            block = func_ir.blocks.get(bid)
            if block:
                for succ in block.succs:
                    if succ not in depths:
                        depths[succ] = d + 1
                    queue.append(succ)
        return max_d
