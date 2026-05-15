"""Type tags for the typed-stack Push VM.

Each `Type` enum value identifies one typed stack inside the VM.  We use a
plain `str`-valued Enum so error messages, logs and JSON serialisation are
human-readable.

The set of types is deliberately wide (compared to the MIMO Push-GP
reference) to maximise the expressive power of the search space:
scalars (Float / Int / Bool) plus the matching variable-length vector
types (FloatVec / BoolVec / IntVec) plus a placeholder FloatMat type.

We do NOT introduce a Code stack: control-flow code blocks live as
syntactic children of control instructions (see `genome.Instruction`),
matching the MIMO reference and keeping VM semantics simple.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class Type(str, Enum):
    FLOAT = "Float"
    INT = "Int"
    BOOL = "Bool"
    FLOATVEC = "FloatVec"
    BOOLVEC = "BoolVec"
    INTVEC = "IntVec"
    FLOATMAT = "FloatMat"


# Default per-type stack depth limits.  These are intentionally generous
# so most evolved programs run without ever hitting the cap; the limits
# exist purely to keep memory bounded if a pathological program tries to
# `Dup` in a loop.
DEFAULT_MAX_DEPTHS: Final[dict[Type, int]] = {
    Type.FLOAT: 200,
    Type.INT: 200,
    Type.BOOL: 200,
    Type.FLOATVEC: 50,
    Type.BOOLVEC: 50,
    Type.INTVEC: 50,
    Type.FLOATMAT: 10,
}


# Hard numeric guards used everywhere in the VM.
LLR_CLAMP: Final[float] = 30.0  # any |x| > LLR_CLAMP is clamped to ±LLR_CLAMP
FLOAT_ABS_MAX: Final[float] = 1e12  # absolute hard cap before NaN/Inf handling
NAN_INF_REPLACEMENT: Final[float] = 0.0


__all__ = [
    "Type",
    "DEFAULT_MAX_DEPTHS",
    "LLR_CLAMP",
    "FLOAT_ABS_MAX",
    "NAN_INF_REPLACEMENT",
]
