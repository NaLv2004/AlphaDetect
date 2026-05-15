"""Genome dataclass + JSON I/O.

A Genome holds two evolved programs (V2C, C2V) plus a vector of 8
log-domain evolved constants (actual value = 10^x, x ∈ [-3, 3]).

Phase-1 design (locked by user): only V2C and C2V are evolved; the
posterior LLR sum and the H·c=0 halt criterion are fixed in the LDPC
simulator and not part of the genome.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .program import Instruction, deep_copy_program

N_EVO_CONSTS = 8
LOG_CONST_MIN = -3.0
LOG_CONST_MAX = 3.0
MAX_PROG_LEN = 80  # total instruction count (incl. nested) per V2C / C2V program


# ============================================================ Instruction JSON

def instruction_to_dict(ins: Instruction) -> Dict[str, Any]:
    d: Dict[str, Any] = {"name": ins.name}
    if ins.code_block is not None:
        d["code_block"] = [instruction_to_dict(c) for c in ins.code_block]
    if ins.code_block2 is not None:
        d["code_block2"] = [instruction_to_dict(c) for c in ins.code_block2]
    return d


def dict_to_instruction(d: Dict[str, Any]) -> Instruction:
    cb = d.get("code_block")
    cb2 = d.get("code_block2")
    return Instruction(
        name=d["name"],
        code_block=[dict_to_instruction(x) for x in cb] if cb is not None else None,
        code_block2=[dict_to_instruction(x) for x in cb2] if cb2 is not None else None,
    )


def program_to_list(prog: List[Instruction]) -> List[Dict[str, Any]]:
    return [instruction_to_dict(i) for i in prog]


def list_to_program(lst: List[Dict[str, Any]]) -> List[Instruction]:
    return [dict_to_instruction(d) for d in lst]


# ====================================================================== Genome


@dataclass
class Genome:
    prog_v2c: List[Instruction] = field(default_factory=list)
    prog_c2v: List[Instruction] = field(default_factory=list)
    log_constants: np.ndarray = field(
        default_factory=lambda: np.zeros(N_EVO_CONSTS, dtype=np.float64)
    )

    # ---- accessors -------------------------------------------------------
    def evo_const_values(self) -> np.ndarray:
        """Actual constants seen by the VM = 10^log_constants, clamped."""
        x = np.clip(self.log_constants, LOG_CONST_MIN, LOG_CONST_MAX)
        return np.power(10.0, x)

    def copy(self) -> "Genome":
        return Genome(
            prog_v2c=deep_copy_program(self.prog_v2c),
            prog_c2v=deep_copy_program(self.prog_c2v),
            log_constants=self.log_constants.copy(),
        )

    # ---- JSON I/O --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prog_v2c": program_to_list(self.prog_v2c),
            "prog_c2v": program_to_list(self.prog_c2v),
            "log_constants": [float(x) for x in self.log_constants.tolist()],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Genome":
        lc = np.asarray(d.get("log_constants", []), dtype=np.float64)
        if lc.size != N_EVO_CONSTS:
            lc2 = np.zeros(N_EVO_CONSTS, dtype=np.float64)
            n = min(lc.size, N_EVO_CONSTS)
            lc2[:n] = lc[:n]
            lc = lc2
        return cls(
            prog_v2c=list_to_program(d.get("prog_v2c", [])),
            prog_c2v=list_to_program(d.get("prog_c2v", [])),
            log_constants=lc,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Genome":
        return cls.from_dict(json.loads(Path(path).read_text()))


__all__ = [
    "Genome",
    "Instruction",
    "N_EVO_CONSTS",
    "LOG_CONST_MIN",
    "LOG_CONST_MAX",
    "MAX_PROG_LEN",
    "instruction_to_dict",
    "dict_to_instruction",
    "program_to_list",
    "list_to_program",
]
