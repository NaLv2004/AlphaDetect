"""Push-GP framework for evolving BP/NMS LDPC decoder update rules.

This package is intentionally decoupled from any specific LDPC simulator:
it only defines the language (types, stacks, instructions, VM), the genome
representation, and the evolutionary operators.

LDPC-specific glue lives in `pushgp_ldpc/` (sibling package).
"""

__version__ = "0.1.0"
