"""Lossless JSON-friendly serialization of Push-GP programs / genomes.

Format is a plain dict tree of {"name": str,
"code_block": [child, ...] | absent,
"code_block2": [child, ...] | absent}.  Atomic instructions reduce to
{"name": <atomic_name>}.  Round-trips via `dict_to_instr` / `program_to_dict`.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .program import Instruction


def instr_to_dict(ins: Instruction) -> Dict[str, Any]:
    d: Dict[str, Any] = {"name": ins.name}
    if ins.code_block is not None:
        d["code_block"] = [instr_to_dict(c) for c in ins.code_block]
    if ins.code_block2 is not None:
        d["code_block2"] = [instr_to_dict(c) for c in ins.code_block2]
    return d


def program_to_dict(prog: List[Instruction]) -> List[Dict[str, Any]]:
    return [instr_to_dict(i) for i in prog]


def genome_to_dict(g) -> Dict[str, Any]:
    return {
        "prog_v2c": program_to_dict(g.prog_v2c),
        "prog_c2v": program_to_dict(g.prog_c2v),
        "log_constants": [float(c) for c in g.log_constants],
    }


def dict_to_instr(d: Dict[str, Any]) -> Instruction:
    cb = [dict_to_instr(c) for c in d["code_block"]] if "code_block" in d else None
    cb2 = [dict_to_instr(c) for c in d["code_block2"]] if "code_block2" in d else None
    return Instruction(name=d["name"], code_block=cb, code_block2=cb2)


def dict_to_program(lst: List[Dict[str, Any]]) -> List[Instruction]:
    return [dict_to_instr(d) for d in lst]


def tree_size(prog: List[Instruction]) -> int:
    """Total number of Instruction nodes in a program (incl. nested)."""
    n = 0
    stack = list(prog)
    while stack:
        ins = stack.pop()
        n += 1
        if ins.code_block is not None:
            stack.extend(ins.code_block)
        if ins.code_block2 is not None:
            stack.extend(ins.code_block2)
    return n


def tree_max_depth(prog: List[Instruction]) -> int:
    """Maximum nesting depth of any node in the program (top level = 0)."""
    best = 0

    def rec(lst, d):
        nonlocal best
        if d > best:
            best = d
        for ins in lst:
            if ins.code_block is not None:
                rec(ins.code_block, d + 1)
            if ins.code_block2 is not None:
                rec(ins.code_block2, d + 1)

    rec(prog, 0)
    return best


__all__ = [
    "instr_to_dict", "program_to_dict", "genome_to_dict",
    "dict_to_instr", "dict_to_program",
    "tree_size", "tree_max_depth",
    "genome_fingerprint",
]


def genome_fingerprint(g, *, const_quant: int = 2) -> str:
    """Stable structural hash for dedup.

    Includes both V2C/C2V program trees and log_constants rounded to
    `const_quant` decimal places (so near-duplicates collide).
    """
    payload = {
        "v": program_to_dict(g.prog_v2c),
        "c": program_to_dict(g.prog_c2v),
        "k": [round(float(x), const_quant) for x in g.log_constants],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
