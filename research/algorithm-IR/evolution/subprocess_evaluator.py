"""Subprocess-isolated wrapper around :class:`MIMOFitnessEvaluator`.

Uses ``subprocess.Popen`` (NOT ``multiprocessing``) to launch worker
processes via the dedicated ``_eval_worker.py`` script.  This avoids
Windows ``spawn``-mode re-importing the calling script as ``__main__``
— a problem the project's :file:`train_gnn.py` would otherwise hit
because all of its top-level setup code is executable on import.

Each worker process handles one genome at a time over a length-prefixed
pickle protocol on stdin/stdout.  The parent kills + respawns a worker
whenever a request exceeds its wall-clock deadline (typically because
an auto-repaired graft compiled to numpy code that hangs forever).
"""

from __future__ import annotations

import logging
import os
import pickle
import struct
import subprocess
import sys
import time
from typing import Any, TYPE_CHECKING

import numpy as np

from evolution.fitness import FitnessResult
from evolution.pool_types import AlgorithmFitnessEvaluator, AlgorithmGenome
from evolution.mimo_evaluator import MIMOEvalConfig
from evolution.materialize import materialize, _extract_func_name_from_full

if TYPE_CHECKING:
    from algorithm_ir.ir.model import FunctionIR

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER_SCRIPT = os.path.join(_HERE, "_eval_worker.py")


def _send_frame(stream, obj) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!I", len(payload)))
    stream.write(payload)
    stream.flush()


def _materialize_for_subprocess(
    genome: AlgorithmGenome,
) -> tuple[str, str, str, float]:
    source = materialize(genome)
    func_name = _extract_func_name_from_full(source, genome.algo_id)
    total_ops = len(genome.structural_ir.ops) if genome.structural_ir else 0
    for pop in genome.slot_populations.values():
        if pop.variants and pop.best_idx < len(pop.variants):
            best = pop.variants[pop.best_idx]
            if best is not None:
                total_ops += len(best.ops)
    complexity = min(total_ops / 500.0, 1.0)
    return genome.algo_id, source, func_name, complexity


# ─────────────────────────────────────────────────────────────────────


class _Worker:
    """Persistent subprocess worker handle with async stdout reader."""

    def __init__(self, cfg: MIMOEvalConfig) -> None:
        import queue
        import threading

        env = os.environ.copy()
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(var, "1")
        repo_root = os.path.dirname(_HERE)
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            [sys.executable, "-u", _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            bufsize=0,
            close_fds=(os.name != "nt"),
            creationflags=flags,
        )
        self.calls = 0

        try:
            _send_frame(self.proc.stdin, cfg)
        except (BrokenPipeError, OSError):
            self.kill()
            raise

        self._inbox: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _reader_loop(self) -> None:
        stdout = self.proc.stdout
        while True:
            try:
                header = stdout.read(4)
                if not header or len(header) < 4:
                    self._inbox.put(None)
                    return
                (n,) = struct.unpack("!I", header)
                payload = stdout.read(n)
                if len(payload) < n:
                    self._inbox.put(None)
                    return
                msg = pickle.loads(payload)
            except Exception:
                self._inbox.put(None)
                return
            self._inbox.put(msg)

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, payload) -> None:
        _send_frame(self.proc.stdin, payload)

    def try_recv(self) -> tuple[bool, object | None]:
        """Non-blocking receive.  Returns ``(have_msg, msg_or_none)``."""
        try:
            msg = self._inbox.get_nowait()
            return True, msg
        except Exception:
            return False, None

    def kill(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2.0)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=1.0)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────


class SubprocessMIMOEvaluator(AlgorithmFitnessEvaluator):
    """Evaluate genomes inside terminate-able worker subprocesses."""

    def __init__(
        self,
        config: MIMOEvalConfig | None = None,
        *,
        deadline_factor: float = 4.0,
        max_calls_per_worker: int = 100,
    ) -> None:
        self.config = config or MIMOEvalConfig()
        self.deadline_factor = max(1.5, float(deadline_factor))
        self.max_calls_per_worker = int(max_calls_per_worker)
        self.n_workers = max(1, int(getattr(self.config, "batch_workers", 1)))
        self._workers: list[_Worker] = []

    def _spawn_worker(self) -> _Worker:
        return _Worker(self.config)

    def _ensure_pool(self) -> None:
        self._workers = [w for w in self._workers if w.is_alive()]
        while len(self._workers) < self.n_workers:
            try:
                self._workers.append(self._spawn_worker())
            except Exception as exc:
                logger.error("Failed to spawn worker: %s", exc)
                break

    def shutdown(self) -> None:
        for w in self._workers:
            w.kill()
        self._workers.clear()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    def evaluate_single_result(self, result: Any) -> float:
        from evolution.mimo_evaluator import symbol_error_rate
        if result is None:
            return 1.0
        x_true, x_hat = result
        return symbol_error_rate(x_true, x_hat)

    def evaluate_source_quick(
        self,
        source: str,
        func_name: str,
        *,
        algo_id: str = "quick",
        n_trials: int = 5,
        timeout_sec: float = 0.5,
        snr_db: float | None = None,
    ) -> float:
        """Evaluate a raw source string in a worker and return scalar SER.

        Used by algorithm_engine's micro-evolve to isolate execution of
        mutated slot variants that may segfault the main process.
        """
        overrides: dict = {"n_trials": int(n_trials), "timeout_sec": float(timeout_sec)}
        if snr_db is not None:
            overrides["snr_db_list"] = [float(snr_db)]
        payload = (algo_id, source, func_name, 0.0, overrides)

        self._ensure_pool()
        if not self._workers:
            return 1.0
        worker = self._workers.pop(0)
        deadline = time.perf_counter() + float(timeout_sec) * 4.0 + 1.0
        try:
            worker.send(payload)
        except (BrokenPipeError, OSError):
            worker.kill()
            return 1.0

        msg = None
        while time.perf_counter() < deadline:
            have, msg = worker.try_recv()
            if have:
                break
            time.sleep(0.005)
        if msg is None:
            worker.kill()
            return 1.0

        worker.calls += 1
        if worker.calls >= self.max_calls_per_worker or not worker.is_alive():
            worker.kill()
        else:
            self._workers.append(worker)

        try:
            tag, back = msg
        except Exception:
            return 1.0
        if tag != "ok":
            return 1.0
        try:
            return float(back["metrics"].get("ser", 1.0))
        except Exception:
            return 1.0

    def evaluate_ir_quick(
        self,
        ir: "FunctionIR",
        *,
        algo_id: str = "quick",
        n_trials: int = 5,
        timeout_sec: float = 0.5,
        snr_db: float | None = None,
    ) -> float:
        """S5: IR-only evaluator boundary.

        The evaluator is the SOLE component permitted to materialise
        FunctionIR -> Python source -> exec. Callers (slot_evolution,
        gp.population, etc.) must hand a FunctionIR here, never source.
        """
        from algorithm_ir.regeneration.codegen import emit_python_source
        try:
            source = emit_python_source(ir)
        except Exception:
            return 1.0
        func_name = getattr(ir, "name", None) or "detector"
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(func_name))
        if not safe or safe[0].isdigit():
            safe = "_" + safe
        return self.evaluate_source_quick(
            source, safe,
            algo_id=algo_id,
            n_trials=n_trials,
            timeout_sec=timeout_sec,
            snr_db=snr_db,
        )

    def evaluate_source_returning_xhat(
        self,
        source: str,
        func_name: str,
        *,
        algo_id: str = "probe",
        n_trials: int = 8,
        timeout_sec: float = 0.5,
        snr_db: float = 14.0,
        n_tx: int = 4,
        n_rx: int = 4,
        seed: int = 0xB17,
    ) -> "np.ndarray | None":
        """S4: probe a detector and return the concatenated xhat array.

        Used by the behavior-hash gate. The probe is fully deterministic
        — fixed seed, fixed channel/noise sequence, fixed constellation —
        so two structurally-different IRs that produce identical xhats
        on this probe are behavioral synonyms.

        Returns None on failure.
        """
        overrides: dict = {
            "n_trials": int(n_trials),
            "timeout_sec": float(timeout_sec),
            "snr_db_list": [float(snr_db)],
            "return_xhat": True,
            "probe_seed": int(seed),
            "probe_n_tx": int(n_tx),
            "probe_n_rx": int(n_rx),
        }
        payload = (algo_id, source, func_name, 0.0, overrides)

        self._ensure_pool()
        if not self._workers:
            return None
        worker = self._workers.pop(0)
        deadline = time.perf_counter() + float(timeout_sec) * 4.0 + 1.0
        try:
            worker.send(payload)
        except (BrokenPipeError, OSError):
            worker.kill()
            return None

        msg = None
        while time.perf_counter() < deadline:
            have, msg = worker.try_recv()
            if have:
                break
            time.sleep(0.005)
        if msg is None:
            worker.kill()
            return None

        worker.calls += 1
        if worker.calls >= self.max_calls_per_worker or not worker.is_alive():
            worker.kill()
        else:
            self._workers.append(worker)

        try:
            tag, back = msg
        except Exception:
            return None
        if tag != "ok":
            return None
        xhat = back.get("xhat") if isinstance(back, dict) else None
        if xhat is None:
            return None
        try:
            return np.asarray(xhat, dtype=complex)
        except Exception:
            return None

    def evaluate(self, genome: AlgorithmGenome) -> FitnessResult:
        try:
            payload = _materialize_for_subprocess(genome)
        except Exception as exc:
            return _failed_result("materialize_failed", repr(exc))

        self._ensure_pool()
        if not self._workers:
            return _failed_result("no_worker")
        worker = self._workers.pop(0)
        deadline = time.perf_counter() + self.config.timeout_sec * self.deadline_factor

        try:
            worker.send(payload)
        except (BrokenPipeError, OSError):
            worker.kill()
            return _failed_result("worker_send_failed")

        msg = None
        while time.perf_counter() < deadline:
            have, msg = worker.try_recv()
            if have:
                break
            time.sleep(0.01)
        if msg is None:
            worker.kill()
            return _failed_result("subprocess_timeout")

        worker.calls += 1
        if worker.calls >= self.max_calls_per_worker or not worker.is_alive():
            worker.kill()
        else:
            self._workers.append(worker)

        try:
            tag, payload_back = msg
        except Exception:
            return _failed_result("worker_protocol")

        if tag == "ok":
            return FitnessResult(
                metrics=payload_back["metrics"],
                is_valid=payload_back["is_valid"],
                weights=payload_back["weights"],
            )
        return _failed_result("worker_exception", str(payload_back))

    def evaluate_batch(
        self, genomes: list[AlgorithmGenome]
    ) -> list[FitnessResult]:
        if not genomes:
            return []
        n = len(genomes)
        results: list[FitnessResult | None] = [None] * n
        payloads: list[tuple | None] = []
        for i, g in enumerate(genomes):
            try:
                payloads.append(_materialize_for_subprocess(g))
            except Exception as exc:
                results[i] = _failed_result("materialize_failed", repr(exc))
                payloads.append(None)

        next_idx = 0
        in_flight: dict[int, tuple[_Worker, float]] = {}

        def _send(i: int) -> None:
            self._ensure_pool()
            if not self._workers:
                results[i] = _failed_result("no_worker")
                return
            worker = self._workers.pop(0)
            try:
                worker.send(payloads[i])
            except (BrokenPipeError, OSError):
                worker.kill()
                results[i] = _failed_result("worker_send_failed")
                return
            in_flight[i] = (
                worker,
                time.perf_counter() + self.config.timeout_sec * self.deadline_factor,
            )

        while next_idx < n and len(in_flight) < self.n_workers:
            if payloads[next_idx] is not None:
                _send(next_idx)
            next_idx += 1

        while in_flight:
            now = time.perf_counter()
            done: list[int] = []
            for idx, (worker, deadline) in list(in_flight.items()):
                if now >= deadline:
                    worker.kill()
                    results[idx] = _failed_result("subprocess_timeout")
                    done.append(idx)
                    continue
                have, msg = worker.try_recv()
                if not have:
                    continue
                if msg is None:
                    worker.kill()
                    results[idx] = _failed_result("worker_died")
                    done.append(idx)
                    continue
                worker.calls += 1
                try:
                    tag, payload_back = msg
                    if tag == "ok":
                        results[idx] = FitnessResult(
                            metrics=payload_back["metrics"],
                            is_valid=payload_back["is_valid"],
                            weights=payload_back["weights"],
                        )
                    else:
                        results[idx] = _failed_result(
                            "worker_exception", str(payload_back)
                        )
                except Exception:
                    results[idx] = _failed_result("worker_protocol")
                done.append(idx)
                if (worker.calls >= self.max_calls_per_worker
                        or not worker.is_alive()):
                    worker.kill()
                else:
                    self._workers.append(worker)
            for idx in done:
                in_flight.pop(idx, None)

            while next_idx < n and len(in_flight) < self.n_workers:
                if payloads[next_idx] is not None:
                    _send(next_idx)
                next_idx += 1

            if in_flight:
                time.sleep(0.005)

        return [
            r if r is not None else _failed_result("missing")
            for r in results
        ]

    # ── Precise (shortlist) evaluation ────────────────────────────────
    # The subprocess worker doesn't implement a tighter precise-SER loop;
    # we approximate it by re-evaluating the shortlist with a temporarily
    # inflated trial count and a generous timeout.  Workers are recycled
    # before/after to make sure the larger config is in effect.

    def _with_temp_config(self, **overrides):
        """Context-manager-like helper that swaps cfg fields and recycles workers."""
        import copy
        old_cfg = self.config
        new_cfg = copy.copy(old_cfg)
        for k, v in overrides.items():
            setattr(new_cfg, k, v)
        self.config = new_cfg
        # Force workers to be respawned with the new cfg.
        for w in self._workers:
            w.kill()
        self._workers.clear()
        return old_cfg

    def evaluate_precise(
        self,
        genome: AlgorithmGenome,
        *,
        target_errors: int = 100,
        max_symbols: int = 200000,
        timeout_sec: float | None = None,
        seed: int | None = None,
    ) -> FitnessResult:
        old = self._with_temp_config(
            n_trials=max(self.config.n_trials, 8),
            timeout_sec=float(timeout_sec) if timeout_sec is not None else max(self.config.timeout_sec, 8.0),
        )
        try:
            return self.evaluate(genome)
        finally:
            self._with_temp_config(
                n_trials=old.n_trials,
                timeout_sec=old.timeout_sec,
            )

    def evaluate_precise_batch(
        self,
        genomes: list[AlgorithmGenome],
        *,
        target_errors: int = 100,
        max_symbols: int = 200000,
        timeout_sec: float | None = None,
        desc: str = "Precise SER",
    ) -> list[FitnessResult]:
        if not genomes:
            return []
        old = self._with_temp_config(
            n_trials=max(self.config.n_trials, 8),
            timeout_sec=float(timeout_sec) if timeout_sec is not None else max(self.config.timeout_sec, 8.0),
        )
        try:
            return self.evaluate_batch(genomes)
        finally:
            self._with_temp_config(
                n_trials=old.n_trials,
                timeout_sec=old.timeout_sec,
            )


def _failed_result(reason: str, detail: str | None = None) -> FitnessResult:
    return FitnessResult(metrics={"ser": 1.0, reason: 1.0}, is_valid=False)
