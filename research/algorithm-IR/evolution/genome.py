"""IR-based genome for evolutionary algorithms."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from typing import Any

import numpy as np

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.regeneration.codegen import emit_python_source


def _compile_source_to_callable(source: str, func_name: str) -> callable:
    """compile() + exec() a Python source string, return the named function."""
    code = compile(source, f"<evolved_{func_name}>", "exec")
    namespace: dict[str, Any] = {}
    # Provide safe math helpers in the namespace
    namespace["__builtins__"] = __builtins__
    namespace["_safe_div"] = _safe_div
    namespace["_safe_log"] = _safe_log
    namespace["_safe_sqrt"] = _safe_sqrt
    namespace["abs"] = abs
    exec(code, namespace)  # noqa: S102
    fn = namespace.get(func_name)
    if fn is None:
        raise RuntimeError(
            f"Compiled source does not define function '{func_name}'.\n"
            f"Source:\n{source}"
        )
    return fn


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-30 else 0.0


def _safe_log(a: float) -> float:
    import math
    return math.log(max(a, 1e-30))


def _safe_sqrt(a: float) -> float:
    import math
    return math.sqrt(max(a, 0.0))


class IRGenome:
    """An evolved individual: N named FunctionIRs + evolvable constants.

    Each program role (e.g. "f_down", "f_up") maps to one FunctionIR.
    Constants are log-domain evolvable floats.
    """

    def __init__(
        self,
        programs: dict[str, FunctionIR],
        constants: np.ndarray | None = None,
        generation: int = 0,
        parent_ids: list[str] | None = None,
        genome_id: str | None = None,
    ) -> None:
        self.programs = programs
        self.constants = (
            constants if constants is not None
            else np.zeros(0, dtype=np.float64)
        )
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.genome_id = genome_id or self._make_id()
        self._callable_cache: dict[str, callable] = {}

    def _make_id(self) -> str:
        import uuid
        return f"g_{uuid.uuid4().hex[:8]}"

    # ----- Code generation -----

    def to_source(self, role: str) -> str:
        """Generate Python source for the given program role."""
        return emit_python_source(self.programs[role])

    def to_callable(self, role: str) -> callable:
        """Compile the program to a Python callable (cached)."""
        if role in self._callable_cache:
            return self._callable_cache[role]

        func_ir = self.programs[role]
        source = emit_python_source(func_ir)
        try:
            fn = _compile_source_to_callable(source, func_ir.name)
        except Exception:
            # Fallback: return a function that always returns 0.0
            fn = lambda *args, **kwargs: 0.0  # noqa: E731
        self._callable_cache[role] = fn
        return fn

    def to_cpp_ops(self, role: str) -> list[int]:
        """Serialize the program to C++ opcode array."""
        from algorithm_ir.regeneration.codegen import emit_cpp_ops
        return emit_cpp_ops(self.programs[role])

    def invalidate_cache(self) -> None:
        """Clear compiled callable cache (call after mutation)."""
        self._callable_cache.clear()

    # ----- Cloning -----

    def clone(self) -> IRGenome:
        """Deep copy: programs, constants, metadata."""
        new_programs = {}
        for role, func_ir in self.programs.items():
            new_programs[role] = copy.deepcopy(func_ir)
        return IRGenome(
            programs=new_programs,
            constants=self.constants.copy(),
            generation=self.generation,
            parent_ids=list(self.parent_ids),
        )

    # ----- Hashing / diversity -----

    def structural_hash(self) -> int:
        """Hash of opcode sequences for niche diversity."""
        h = hashlib.md5(usedforsecurity=False)
        for role in sorted(self.programs.keys()):
            func_ir = self.programs[role]
            opcodes = []
            for block_id in sorted(func_ir.blocks.keys()):
                block = func_ir.blocks[block_id]
                for op_id in block.op_ids:
                    if op_id in func_ir.ops:
                        opcodes.append(func_ir.ops[op_id].opcode)
            h.update("|".join(opcodes).encode())
        return int(h.hexdigest()[:16], 16)

    # ----- Serialization -----

    def serialize(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict (programs as Python source)."""
        programs_src = {}
        for role, func_ir in self.programs.items():
            programs_src[role] = emit_python_source(func_ir)
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "constants": self.constants.tolist(),
            "programs": programs_src,
        }

    @classmethod
    def deserialize(
        cls, data: dict[str, Any], compile_fn=None
    ) -> IRGenome:
        """Deserialize from dict.

        Programs stored as source are recompiled via compile_source_to_ir.
        """
        if compile_fn is None:
            from algorithm_ir.frontend.ir_builder import compile_source_to_ir
            compile_fn = compile_source_to_ir

        programs: dict[str, FunctionIR] = {}
        for role, source in data["programs"].items():
            func_name = source.split("(")[0].replace("def ", "").strip()
            programs[role] = compile_fn(source, func_name)

        return cls(
            programs=programs,
            constants=np.array(data.get("constants", []), dtype=np.float64),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            genome_id=data.get("genome_id"),
        )

    def __repr__(self) -> str:
        roles = list(self.programs.keys())
        return (
            f"IRGenome(id={self.genome_id}, gen={self.generation}, "
            f"roles={roles}, n_const={len(self.constants)})"
        )
