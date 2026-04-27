"""Runtime stub for the ``with slot(name, inputs=..., outputs=...):`` DSL.

The IR builder parses ``with slot(...)`` blocks at compile time and tags ops
with ``slot_id`` plus populates ``FunctionIR.slot_meta``. At Python run time,
``slot(...)`` is an inert no-op context manager so the same source string can
also be exec'd directly (e.g. by ``materialize`` for evaluation).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any


@contextmanager
def slot(name: str, *, inputs: tuple = (), outputs: tuple = ()) -> Any:  # noqa: ARG001
    yield None


__all__ = ["slot"]
