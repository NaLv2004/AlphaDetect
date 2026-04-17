from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from types import FunctionType


@dataclass
class ParsedFunction:
    tree: ast.FunctionDef
    source: str
    filename: str
    globals_dict: dict


def parse_function(fn: FunctionType) -> ParsedFunction:
    source = textwrap.dedent(inspect.getsource(fn))
    module = ast.parse(source)
    func_node = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            func_node = node
            break
    if func_node is None:
        raise ValueError(f"Could not locate function AST for {fn.__name__}")
    return ParsedFunction(
        tree=func_node,
        source=source,
        filename=inspect.getsourcefile(fn) or "<unknown>",
        globals_dict=fn.__globals__,
    )


def source_span(node: ast.AST) -> tuple[int, int, int, int] | None:
    if not hasattr(node, "lineno"):
        return None
    return (
        getattr(node, "lineno", 0),
        getattr(node, "col_offset", 0),
        getattr(node, "end_lineno", getattr(node, "lineno", 0)),
        getattr(node, "end_col_offset", getattr(node, "col_offset", 0)),
    )

