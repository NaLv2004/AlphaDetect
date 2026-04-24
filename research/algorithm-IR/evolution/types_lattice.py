"""Backward-compatibility shim.

The type lattice was relocated to ``algorithm_ir.ir.type_lattice``
because types are a property of the IR, not of the evolutionary engine.
This module re-exports the public API for any caller still importing
from the old location.

New code should import directly from ``algorithm_ir.ir.type_lattice``.
"""
from __future__ import annotations

from algorithm_ir.ir.type_lattice import *  # noqa: F401, F403
from algorithm_ir.ir.type_lattice import (  # noqa: F401
    PRIMITIVE_TYPES,
    TENSOR_TYPES,
    OBJECT_TYPES,
    ALL_ATOMIC_TYPES,
    TYPE_TOP,
    TYPE_VOID,
    is_subtype,
    unify,
    available_ops_for_type,
    default_value,
    infer_value_type,
    parse_composite,
    is_tuple_type,
    is_list_type,
    is_dict_type,
    tuple_components,
    list_element_type,
    dict_value_type,
    combine_binary_type,
    combine_unary_type,
    infer_call_return_type,
    register_callable_return,
    callable_return_type,
    is_numeric,
    is_array_like,
    is_real,
    is_complex,
    promote_dtype,
    promote_rank,
)
