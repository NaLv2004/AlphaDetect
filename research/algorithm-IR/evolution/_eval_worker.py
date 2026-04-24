"""Worker script — read genome eval requests from stdin, write results to stdout.

Used by :class:`SubprocessMIMOEvaluator`.  Each request/response is a
length-prefixed pickle frame on the binary stdin/stdout streams.
"""

from __future__ import annotations

import os
import pickle
import struct
import sys
import time


def _read_frame(stream) -> object | None:
    header = stream.read(4)
    if len(header) < 4:
        return None
    (n,) = struct.unpack("!I", header)
    payload = stream.read(n)
    if len(payload) < n:
        return None
    return pickle.loads(payload)


def _write_frame(stream, obj: object) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!I", len(payload)))
    stream.write(payload)
    stream.flush()


def main() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)

    import numpy as np
    import concurrent.futures as _cf
    from evolution.mimo_evaluator import (
        generate_mimo_sample,
        qam16_constellation,
        qpsk_constellation,
        _nearest_symbols,
    )
    from evolution.materialize import _default_exec_namespace

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # First message is config.
    cfg = _read_frame(stdin)
    if cfg is None:
        return

    constellation = (
        qam16_constellation() if cfg.mod_order == 16 else qpsk_constellation()
    )

    while True:
        msg = _read_frame(stdin)
        if msg is None:
            return
        # Accept either the legacy 4-tuple or a 5-tuple with an
        # optional overrides dict (n_trials / timeout_sec / snr_db_list).
        overrides = None
        if len(msg) == 5:
            algo_id, source, func_name, complexity, overrides = msg
        else:
            algo_id, source, func_name, complexity = msg

        # Per-request effective config fields.
        eff_n_trials = cfg.n_trials
        eff_timeout_sec = cfg.timeout_sec
        eff_snr_list = cfg.snr_db_list
        if overrides:
            eff_n_trials = int(overrides.get("n_trials", eff_n_trials))
            eff_timeout_sec = float(overrides.get("timeout_sec", eff_timeout_sec))
            eff_snr_list = overrides.get("snr_db_list", eff_snr_list)

        try:
            ns = _default_exec_namespace()
            exec(compile(source, f"<materialized:{algo_id}>", "exec"), ns)
            fn = ns.get(func_name)
            if fn is None:
                raise RuntimeError(f"function {func_name!r} not found")
        except BaseException as exc:  # noqa: BLE001
            _write_frame(stdout, ("ok", {
                "metrics": {"ser": 1.0, "compile_error": 1.0,
                            "complexity": complexity, "eval_time": 0.0},
                "is_valid": False,
                "weights": {"ser": 1.0,
                            "complexity": cfg.complexity_weight,
                            "eval_time": 0.0},
            }))
            continue

        rng = np.random.default_rng(cfg.seed)
        ser_per_snr: dict = {}
        total_time = 0.0
        _trial_timeout = max(1.0, eff_timeout_sec / max(eff_n_trials, 1) * 3)

        for snr_db in eff_snr_list:
            errors = 0
            total = 0
            t0 = time.perf_counter()
            _ex = _cf.ThreadPoolExecutor(max_workers=1)
            _ex_dirty = False

            for trial_idx in range(eff_n_trials):
                H, x_true, y, sigma2 = generate_mimo_sample(
                    cfg.Nr, cfg.Nt, constellation, snr_db, rng,
                )
                if _ex_dirty:
                    _ex.shutdown(wait=False)
                    _ex = _cf.ThreadPoolExecutor(max_workers=1)
                    _ex_dirty = False
                try:
                    future = _ex.submit(fn, H, y, sigma2, constellation)
                    x_hat = future.result(timeout=_trial_timeout)
                    if x_hat is None or len(x_hat) != cfg.Nt:
                        errors += cfg.Nt
                    else:
                        x_hat = _nearest_symbols(x_hat, constellation)
                        errors += int(np.sum(np.abs(x_true - x_hat) > 1e-6))
                except _cf.TimeoutError:
                    future.cancel()
                    errors += cfg.Nt
                    _ex_dirty = True
                except BaseException:
                    errors += cfg.Nt

                total += cfg.Nt
                elapsed = time.perf_counter() - t0
                if elapsed > eff_timeout_sec:
                    remaining = eff_n_trials - (trial_idx + 1)
                    errors += remaining * cfg.Nt
                    total += remaining * cfg.Nt
                    break

            _ex.shutdown(wait=False)
            total_time += time.perf_counter() - t0
            ser_per_snr[snr_db] = errors / max(total, 1)

        avg_ser = float(np.mean(list(ser_per_snr.values())))
        metrics = {"ser": avg_ser, "complexity": complexity, "eval_time": total_time}
        for snr, ser in ser_per_snr.items():
            metrics[f"ser_{snr:.0f}dB"] = ser
        weights = {"ser": 1.0, "complexity": cfg.complexity_weight, "eval_time": 0.0}
        for snr in ser_per_snr:
            weights[f"ser_{snr:.0f}dB"] = 0.0

        _write_frame(stdout, ("ok", {
            "metrics": metrics,
            "is_valid": True,
            "weights": weights,
        }))


if __name__ == "__main__":
    main()
