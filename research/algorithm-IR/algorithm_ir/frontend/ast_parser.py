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

    # Build globals_dict including closure variables
    globals_dict = dict(fn.__globals__)
    if fn.__closure__ and fn.__code__.co_freevars:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                globals_dict[name] = cell.cell_contents
            except ValueError:
                pass

    return ParsedFunction(
        tree=func_node,
        source=source,
        filename=inspect.getsourcefile(fn) or "<unknown>",
        globals_dict=globals_dict,
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

