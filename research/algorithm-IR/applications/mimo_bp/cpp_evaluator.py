"""Python ↔ C++ bridge for the BP-IR evaluator DLL.

Loads bp_ir_eval.dll and provides a Pythonic interface.
"""

from __future__ import annotations

import ctypes
import os
import struct
import numpy as np
from pathlib import Path
from typing import Any


# Path to DLL (next to this file in cpp/)
_DLL_DIR = Path(__file__).parent / "cpp"
_DLL_NAME = "bp_ir_eval.dll"
_DLL_PATH = _DLL_DIR / _DLL_NAME

_lib = None


def _load_dll():
    """Load the DLL lazily."""
    global _lib
    if _lib is not None:
        return _lib
    if not _DLL_PATH.exists():
        raise FileNotFoundError(
            f"DLL not found at {_DLL_PATH}. "
            f"Build it first with: cd {_DLL_DIR} && build.bat"
        )
    _lib = ctypes.CDLL(str(_DLL_PATH))
    _setup_prototypes(_lib)
    return _lib


def _setup_prototypes(lib):
    """Declare C function signatures."""
    # bp_ir_create
    lib.bp_ir_create.restype = ctypes.c_void_p
    lib.bp_ir_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
    ]
    # bp_ir_destroy
    lib.bp_ir_destroy.restype = None
    lib.bp_ir_destroy.argtypes = [ctypes.c_void_p]
    # bp_ir_eval_one
    lib.bp_ir_eval_one.restype = ctypes.c_double
    lib.bp_ir_eval_one.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    # bp_ir_eval_dataset
    lib.bp_ir_eval_dataset.restype = ctypes.c_double
    lib.bp_ir_eval_dataset.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    # ir_eval_expr
    lib.ir_eval_expr.restype = ctypes.c_double
    lib.ir_eval_expr.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ]
    # bp_ir_lmmse
    lib.bp_ir_lmmse.restype = ctypes.c_double
    lib.bp_ir_lmmse.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]


def _to_int_array(data: list[int]) -> ctypes.Array:
    """Convert Python int list to ctypes int array."""
    arr = (ctypes.c_int * len(data))(*data)
    return arr


def _to_double_array(data: np.ndarray) -> ctypes.Array:
    """Convert numpy array to ctypes double array (contiguous)."""
    arr = np.ascontiguousarray(data, dtype=np.float64)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _interleave_complex(arr: np.ndarray) -> np.ndarray:
    """Convert complex array to interleaved real format."""
    flat = arr.ravel()
    interleaved = np.empty(2 * len(flat), dtype=np.float64)
    interleaved[0::2] = flat.real
    interleaved[1::2] = flat.imag
    return interleaved


def ir_eval_expr_cpp(program: list[int], args: list[float]) -> float:
    """Evaluate a single IR expression using the C++ evaluator."""
    lib = _load_dll()
    prog_arr = _to_int_array(program)
    args_arr = (ctypes.c_double * len(args))(*args)
    return lib.ir_eval_expr(prog_arr, len(program), args_arr, len(args))


class CppBPIREvaluator:
    """Python wrapper for the C++ BP-IR decoder DLL."""

    def __init__(
        self,
        Nt: int = 16,
        Nr: int = 16,
        mod_order: int = 16,
        constellation: np.ndarray | None = None,
        max_nodes: int = 500,
        max_bp_iters: int = 5,
    ):
        self.Nt = Nt
        self.Nr = Nr
        self.M = mod_order

        if constellation is None:
            constellation = self._default_qam16()
        self.constellation = constellation

        lib = _load_dll()

        cons_interleaved = _interleave_complex(constellation)
        cons_arr = np.ascontiguousarray(cons_interleaved)

        self._handle = lib.bp_ir_create(
            Nt, Nr, mod_order,
            cons_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            max_nodes, max_bp_iters,
        )
        self._lib = lib

    def __del__(self):
        if hasattr(self, '_handle') and self._handle is not None:
            self._lib.bp_ir_destroy(self._handle)
            self._handle = None

    def evaluate_genome(
        self,
        prog_down: list[int],
        prog_up: list[int],
        prog_belief: list[int],
        prog_halt: list[int],
        dataset: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        """Evaluate a 4-program genome on a dataset.

        Args:
            prog_down/up/belief/halt: Flat opcode arrays from emit_cpp_ops.
            dataset: Dict with 'H', 'y', 'x_true', 'noise_var' arrays.

        Returns:
            (avg_ber, avg_flops)
        """
        n_samples = len(dataset['noise_var'])
        H_all = _interleave_complex(dataset['H'].reshape(-1))
        y_all = _interleave_complex(dataset['y'].reshape(-1))
        x_all = _interleave_complex(dataset['x_true'].reshape(-1))
        noise = np.ascontiguousarray(dataset['noise_var'], dtype=np.float64)

        d_arr = _to_int_array(prog_down)
        u_arr = _to_int_array(prog_up)
        b_arr = _to_int_array(prog_belief)
        h_arr = _to_int_array(prog_halt)

        avg_flops = ctypes.c_double(0.0)
        total_faults = ctypes.c_int(0)

        avg_ber = self._lib.bp_ir_eval_dataset(
            self._handle,
            d_arr, len(prog_down),
            u_arr, len(prog_up),
            b_arr, len(prog_belief),
            h_arr, len(prog_halt),
            n_samples,
            H_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            noise.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(avg_flops),
            ctypes.byref(total_faults),
        )

        return avg_ber, avg_flops.value

    def evaluate_lmmse(self, dataset: dict[str, np.ndarray]) -> float:
        """LMMSE baseline BER."""
        n_samples = len(dataset['noise_var'])
        H_all = _interleave_complex(dataset['H'].reshape(-1))
        y_all = _interleave_complex(dataset['y'].reshape(-1))
        x_all = _interleave_complex(dataset['x_true'].reshape(-1))
        noise = np.ascontiguousarray(dataset['noise_var'], dtype=np.float64)

        return self._lib.bp_ir_lmmse(
            self._handle,
            n_samples,
            H_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            noise.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    @staticmethod
    def _default_qam16() -> np.ndarray:
        """Generate standard 16-QAM constellation (normalized)."""
        levels = np.array([-3, -1, 1, 3], dtype=np.float64)
        norm = np.sqrt(10.0)  # average power normalization
        points = []
        for re in levels:
            for im in levels:
                points.append(complex(re / norm, im / norm))
        return np.array(points, dtype=np.complex128)
