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

# VM numeric guard policy
# -----------------------
# DOMAIN errors (e.g. log(<=0), sqrt(<0), atanh(|x|>=1), 1/0, x%0, NaN result)
# trigger an immediate `vm.fault = True` — the program is rejected, not
# silently coerced to 0.  This is the only way the seeder validator and
# DCE oracle can distinguish a legitimate constant-zero program from a
# program that degenerated into a constant via a hidden guard.
#
# OVERFLOW (e.g. exp(large), a*b > clamp, ±Inf result) is clamped in
# magnitude to ±_FLOAT_CLAMP (default ±30.0, matching LLR_CLAMP).  We
# align the clamp with the LLR scale so legitimate BP messages survive
# the VM unchanged, while values driven into hundreds/thousands by an
# evo-amplified `Float.Exp` are visibly truncated instead of silently
# coerced to 0.  The clamp can be tuned at runtime via
# `set_float_clamp(x)` or by setting the environment variable
# `PUSHGP_FLOAT_CLAMP` before importing this module.
import os as _os

def _read_clamp_env(default: float) -> float:
    raw = _os.environ.get("PUSHGP_FLOAT_CLAMP")
    if raw is None:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    if not (v > 0.0):
        return default
    return v

_FLOAT_CLAMP: float = _read_clamp_env(30.0)


def get_float_clamp() -> float:
    """Return the current overflow-clamp magnitude (>0)."""
    return _FLOAT_CLAMP


def set_float_clamp(x: float) -> None:
    """Set the overflow-clamp magnitude.  Must be > 0."""
    global _FLOAT_CLAMP
    if not (x > 0.0):
        raise ValueError(f"float clamp must be > 0, got {x!r}")
    _FLOAT_CLAMP = float(x)


# Backwards-compatible alias retained for any caller that still imports
# the name; points at the (mutable) clamp via a callable, not a constant.
FLOAT_ABS_MAX: Final[float] = 1e12  # legacy, no longer used by guards


# Sentinel raised by guarded ops to signal a DOMAIN error.  Handlers
# catch this and set `vm.state.fault`.
class DomainError(ArithmeticError):
    """Raised by VM op helpers when an operand is outside the op's domain."""


__all__ = [
    "Type",
    "DEFAULT_MAX_DEPTHS",
    "LLR_CLAMP",
    "FLOAT_ABS_MAX",
    "DomainError",
    "get_float_clamp",
    "set_float_clamp",
]
