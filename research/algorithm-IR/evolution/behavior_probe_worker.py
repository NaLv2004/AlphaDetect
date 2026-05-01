"""Isolated behavior-probe worker for materialized detector functions.

This module is launched as a subprocess by ``train_gnn.py``.  It keeps
potentially non-terminating generated detector code out of the trainer
process, so timeout threads cannot accumulate and starve later training.
"""

from __future__ import annotations

import os
import pickle
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evolution.materialize import _default_exec_namespace


def _nearest_constellation_symbols(x_hat: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    x_hat = np.asarray(x_hat)
    out = np.empty(x_hat.shape[0], dtype=complex)
    for i in range(x_hat.shape[0]):
        dists = np.abs(constellation - x_hat[i]) ** 2
        out[i] = constellation[int(np.argmin(dists))]
    return out


def _compile_from_source(source: str, func_name: str, label: str):
    ns = _default_exec_namespace()
    exec(compile(source, f"<behavior_probe:{label}>", "exec"), ns)
    fn = ns.get(func_name)
    if fn is None:
        raise RuntimeError(f"Function {func_name!r} not found in {label}")
    return fn


def _call_with_thread_timeout(fn, args: tuple[Any, ...], timeout: float):
    out: dict[str, Any] = {}

    def _runner() -> None:
        try:
            out["result"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001
            out["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=max(float(timeout), 0.01))
    if t.is_alive():
        raise TimeoutError(f"materialized detector exceeded {timeout:.1f}s")
    if "error" in out:
        raise out["error"]
    return out.get("result")


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    host_fn = _compile_from_source(
        payload["host_source"],
        payload["host_func_name"],
        "host",
    )
    child_fn = _compile_from_source(
        payload["child_source"],
        payload["child_func_name"],
        "child",
    )
    timeout = float(payload["timeout"])
    probes = payload["probes"]

    total_changed = 0
    total_symbols = 0
    n_timeouts = 0
    for H, _x_true, y, sigma2, constellation in probes:
        try:
            host_out = _call_with_thread_timeout(
                host_fn, (H, y, sigma2, constellation), timeout,
            )
            child_out = _call_with_thread_timeout(
                child_fn, (H, y, sigma2, constellation), timeout,
            )
            host_sym = _nearest_constellation_symbols(np.asarray(host_out), constellation)
            child_sym = _nearest_constellation_symbols(np.asarray(child_out), constellation)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, TimeoutError):
                n_timeouts += 1
            return {
                "ok": False,
                "exception_repr": f"{type(exc).__name__}: {exc}",
                "n_timeouts": n_timeouts,
                "total_changed": 0,
                "total_symbols": 0,
            }

        if host_sym.shape != child_sym.shape:
            return {
                "ok": False,
                "exception_repr": f"shape_mismatch: host={host_sym.shape} child={child_sym.shape}",
                "n_timeouts": n_timeouts,
                "total_changed": 0,
                "total_symbols": 0,
            }
        total_changed += int(np.sum(np.abs(host_sym - child_sym) > 1e-6))
        total_symbols += len(host_sym)

    return {
        "ok": True,
        "exception_repr": None,
        "n_timeouts": n_timeouts,
        "total_changed": total_changed,
        "total_symbols": total_symbols,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: behavior_probe_worker.py <input.pkl> <output.pkl>", file=sys.stderr)
        return 2
    in_path, out_path = sys.argv[1], sys.argv[2]
    try:
        with open(in_path, "rb") as fh:
            payload = pickle.load(fh)
        result = _run(payload)
    except BaseException as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "exception_repr": f"{type(exc).__name__}: {exc}",
            "n_timeouts": 1 if isinstance(exc, TimeoutError) else 0,
            "total_changed": 0,
            "total_symbols": 0,
        }
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
