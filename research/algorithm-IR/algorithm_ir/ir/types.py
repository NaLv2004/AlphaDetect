"""
Custom type attribute for the alg dialect.

AlgType wraps a Python type-hint string (e.g. "int", "float", "list")
as a proper xDSL TypeAttribute so it can be used as operand/result types.
"""
from __future__ import annotations

from xdsl.ir import ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition


@irdl_attr_definition
class AlgType(ParametrizedAttribute, TypeAttribute):
    """A type in the Algorithm IR — carries a type-hint string."""
    name = "alg.type"
