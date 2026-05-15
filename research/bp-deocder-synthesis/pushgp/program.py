"""Program / Instruction representation.

An evolved program is a Python list of `Instruction` objects.  Control
instructions (Exec.If / Exec.When / Exec.DoTimes / Exec.DoRange /
Exec.While) carry one or two nested code blocks as plain lists.

We deliberately keep `Instruction` minimal — JSON serialisation lives
in `pushgp.genome` (PR3), random generation in `pushgp.random_program`
(PR3), evolution operators in `pushgp.mutation` / `crossover` (PR4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Instruction:
    name: str
    code_block: Optional[List["Instruction"]] = None
    code_block2: Optional[List["Instruction"]] = None

    def is_control(self) -> bool:
        return self.code_block is not None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if not self.is_control():
            return self.name
        b1 = "{...}" if self.code_block else "{}"
        if self.code_block2 is not None:
            b2 = "{...}" if self.code_block2 else "{}"
            return f"{self.name}{b1}{b2}"
        return f"{self.name}{b1}"


def deep_copy_program(prog: List[Instruction]) -> List[Instruction]:
    out: List[Instruction] = []
    for ins in prog:
        cb = deep_copy_program(ins.code_block) if ins.code_block is not None else None
        cb2 = deep_copy_program(ins.code_block2) if ins.code_block2 is not None else None
        out.append(Instruction(name=ins.name, code_block=cb, code_block2=cb2))
    return out


def program_length(prog: List[Instruction]) -> int:
    """Total instruction count, including nested blocks."""
    n = 0
    for ins in prog:
        n += 1
        if ins.code_block is not None:
            n += program_length(ins.code_block)
        if ins.code_block2 is not None:
            n += program_length(ins.code_block2)
    return n


__all__ = ["Instruction", "deep_copy_program", "program_length"]
