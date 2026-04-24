"""Pytest unit tests for evolution.types_lattice (S0)."""
from __future__ import annotations

import numpy as np
import pytest

from evolution.types_lattice import (
    PRIMITIVE_TYPES,
    TENSOR_TYPES,
    OBJECT_TYPES,
    TYPE_TOP,
    TYPE_VOID,
    available_ops_for_type,
    default_value,
    infer_value_type,
    is_subtype,
    parse_composite,
    unify,
)


def test_primitive_self_subtype():
    for t in PRIMITIVE_TYPES:
        assert is_subtype(t, t)


def test_tensor_self_subtype():
    for t in TENSOR_TYPES:
        assert is_subtype(t, t)


def test_top_is_supertype_of_all():
    for t in list(PRIMITIVE_TYPES) + list(TENSOR_TYPES):
        assert is_subtype(t, TYPE_TOP)


def test_int_is_subtype_of_float():
    assert is_subtype("int", "float")


def test_float_not_subtype_of_int():
    assert not is_subtype("float", "int")


def test_unify_returns_least_supertype():
    res = unify("int", "float")
    assert res in {"float", "complex", TYPE_TOP}


def test_unify_identity():
    assert unify("int", "int") == "int"


def test_default_value_returns_correct_type_for_primitives():
    assert isinstance(default_value("int"), int)
    assert isinstance(default_value("float"), float)


def test_default_value_returns_ndarray_for_vector():
    v = default_value("vec_f")
    assert isinstance(v, np.ndarray)
    assert v.ndim == 1


def test_default_value_returns_ndarray_for_matrix():
    m = default_value("mat_f")
    assert isinstance(m, np.ndarray)
    assert m.ndim == 2


def test_infer_value_type_int():
    assert infer_value_type(7) in {"int", "float"}


def test_infer_value_type_ndarray_vector():
    v = np.zeros(4, dtype=float)
    t = infer_value_type(v)
    assert t in TENSOR_TYPES or t == TYPE_TOP


def test_infer_value_type_ndarray_matrix():
    m = np.zeros((4, 4), dtype=float)
    t = infer_value_type(m)
    assert t in TENSOR_TYPES or t == TYPE_TOP


def test_available_ops_returns_iterable():
    ops = available_ops_for_type("float")
    assert hasattr(ops, "__iter__")
    assert len(list(ops)) >= 1


def test_parse_composite_tuple():
    parsed = parse_composite("tuple<int,float>")
    assert parsed is not None


def test_parse_composite_list():
    parsed = parse_composite("list<float>")
    assert parsed is not None


def test_parse_composite_invalid_returns_none():
    assert parse_composite("not_a_composite") is None


def test_void_type_constant():
    assert TYPE_VOID == "void"


def test_top_type_constant():
    assert TYPE_TOP == "any"
