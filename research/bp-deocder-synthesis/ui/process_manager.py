"""Cross-process lifecycle manager.

On Windows uses a Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE so
that *all* child processes (including grandchildren — the 16 evolution
workers) are forcibly terminated when this Python process exits.

No orphan processes can survive a crash of the UI server.

Public API:
    pm = ProcessManager()
    rid = pm.launch(cmdline=[...], cwd=..., log_stdout=Path, log_stderr=Path,
                    label=str, meta=dict)
    pm.kill(rid)
    pm.list() -> [RunRecord]
    pm.get(rid) -> RunRecord | None
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_IS_WINDOWS = sys.platform.startswith("win")

if _IS_WINDOWS:
    import win32api
    import win32con
    import win32job
    import win32process

# Reserve this many logical CPUs for the OS / UI / browser so that the
# system stays responsive even when every active run is maxed out.
_OS_RESERVE_CORES = 4
# Hard floor on the worker budget; never allow zero.
_MIN_WORKER_BUDGET = 2

# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class RunRecord:
    rid: str                       # short id (also used as run-name)
    label: str                     # display label
    cmdline: List[str]
    cwd: str
    pid: int
    started_at: float
    stopped_at: Optional[float] = None
    returncode: Optional[int] = None
    log_stdout: Optional[str] = None
    log_stderr: Optional[str] = None
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)
    status: str = "running"        # running | exited | killed | error

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["started_at_iso"] = _dt.datetime.fromtimestamp(
            self.started_at).strftime("%Y-%m-%d %H:%M:%S")
        if self.stopped_at:
            d["stopped_at_iso"] = _dt.datetime.fromtimestamp(
                self.stopped_at).strftime("%Y-%m-%d %H:%M:%S")
            d["elapsed_s"] = self.stopped_at - self.started_at
        else:
            d["elapsed_s"] = time.time() - self.started_at
        return d


# --------------------------------------------------------------------------- #
# ProcessManager
# --------------------------------------------------------------------------- #
class ProcessManager:
    def __init__(self, max_concurrent: int = 4) -> None:
        self.max_concurrent = max_concurrent
        # Global worker budget: the sum of `--workers` across all active
        # evolution runs must not exceed this.  Default leaves
        # `_OS_RESERVE_CORES` logical CPUs for the OS / UI / browser to
        # prevent USB-HID starvation and watchdog reboots that have been
        # observed when many `pop=200` runs saturate every core.
        self.total_worker_budget = max(
            _MIN_WORKER_BUDGET,
            (os.cpu_count() or 4) - _OS_RESERVE_CORES,
        )
        self._lock = threading.Lock()
        self._records: Dict[str, RunRecord] = {}
        self._popens: Dict[str, subprocess.Popen] = {}
        # rid -> declared worker count (for budgeting).  Watcher / parser
        # children declare 0 because they are I/O-bound.
        self._workers: Dict[str, int] = {}
        # rid -> Win32 Job Object handle.  Each run gets its own job so
        # that `TerminateJobObject` kills the *entire* tree atomically,
        # including `multiprocessing.spawn`-created workers whose parent
        # linkage `taskkill /T` cannot follow reliably.  Each per-run
        # job has KILL_ON_JOB_CLOSE set, so when the UI process exits
        # the OS closes every handle and kills every child --> no orphans.
        self._jobs: Dict[str, Any] = {}

    @staticmethod
    def _make_job():
        """Create a Win32 Job Object configured for atomic kill +
        kill-on-close + no breakaway."""
        if not _IS_WINDOWS:
            return None
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation)
        flags = info["BasicLimitInformation"]["LimitFlags"]
        flags |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        flags &= ~win32job.JOB_OBJECT_LIMIT_BREAKAWAY_OK
        flags &= ~win32job.JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
        info["BasicLimitInformation"]["LimitFlags"] = flags
        win32job.SetInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation, info)
        return job

    # ----- budget helpers -------------------------------------------------- #
    def _active_worker_load(self) -> int:
        return sum(
            n for rid, n in self._workers.items()
            if self._records.get(rid) is not None
            and self._records[rid].status == "running"
        )

    def budget_status(self) -> Dict[str, int]:
        with self._lock:
            used = self._active_worker_load()
        return {
            "used": used,
            "budget": self.total_worker_budget,
            "free": max(0, self.total_worker_budget - used),
            "cpu_count": os.cpu_count() or 0,
        }

    # ----- public API ------------------------------------------------------ #
    def launch(self,
               cmdline: List[str],
               cwd: str,
               log_stdout: Path,
               log_stderr: Path,
               label: str,
               rid: str,
               meta: Optional[Dict[str, Any]] = None,
               workers: int = 0,
               below_normal_priority: bool = True) -> RunRecord:
        with self._lock:
            active = sum(1 for r in self._records.values()
                         if r.status == "running")
            if active >= self.max_concurrent:
                raise RuntimeError(
                    f"max concurrent runs reached ({self.max_concurrent}); "
                    "stop a run before starting another."
                )
            if rid in self._records and self._records[rid].status == "running":
                raise RuntimeError(f"run id `{rid}` already running")
            used = self._active_worker_load()
            if workers > 0 and used + workers > self.total_worker_budget:
                raise RuntimeError(
                    f"CPU worker budget exceeded: requested {workers}, "
                    f"already in use {used}, budget {self.total_worker_budget} "
                    f"(reserved {_OS_RESERVE_CORES} cores for OS/UI). "
                    f"Stop a run or lower --workers."
                )

        log_stdout.parent.mkdir(parents=True, exist_ok=True)
        f_out = open(log_stdout, "wb", buffering=0)
        f_err = open(log_stderr, "wb", buffering=0)

        creationflags = 0
        if _IS_WINDOWS:
            # NOTE: we intentionally do NOT pass CREATE_NEW_PROCESS_GROUP.
            # That flag would let the child install its own Ctrl+C handler
            # but it also subtly breaks Job-Object inheritance for some
            # multiprocessing spawn paths.  CREATE_NO_WINDOW is enough.
            creationflags = (
                0x08000000  # CREATE_NO_WINDOW
                | (0x00004000 if below_normal_priority else 0)
                # BELOW_NORMAL_PRIORITY_CLASS keeps the UI / browser /
                # mouse-input thread responsive under sustained 100% CPU.
            )

        popen = subprocess.Popen(
            cmdline,
            cwd=cwd,
            stdout=f_out,
            stderr=f_err,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=False,
        )

        if _IS_WINDOWS:
            # Create a per-run Job Object.  Assign the freshly-started
            # process to it; all children (the 16 evolution workers
            # spawned later, including those launched via
            # multiprocessing.spawn) inherit the job automatically since
            # Vista and cannot break out (BREAKAWAY_OK is masked off).
            run_job = self._make_job()
            try:
                h_proc = win32api.OpenProcess(
                    win32con.PROCESS_ALL_ACCESS, False, popen.pid)
                win32job.AssignProcessToJobObject(run_job, h_proc)
                win32api.CloseHandle(h_proc)
            except Exception as exc:
                popen.kill()
                try: win32api.CloseHandle(run_job)
                except Exception: pass
                raise RuntimeError(
                    f"failed to assign process to job: {exc}") from exc
        else:
            run_job = None

        rec = RunRecord(
            rid=rid,
            label=label,
            cmdline=list(cmdline),
            cwd=cwd,
            pid=popen.pid,
            started_at=time.time(),
            log_stdout=str(log_stdout),
            log_stderr=str(log_stderr),
            meta=dict(meta or {}),
            status="running",
        )
        with self._lock:
            self._records[rid] = rec
            self._popens[rid] = popen
            self._workers[rid] = int(workers)
            if run_job is not None:
                self._jobs[rid] = run_job

        # background watcher to update status on exit
        t = threading.Thread(target=self._watch, args=(rid,), daemon=True)
        t.start()
        return rec

    def kill(self, rid: str, timeout: float = 5.0) -> bool:
        with self._lock:
            popen = self._popens.get(rid)
            rec = self._records.get(rid)
            job = self._jobs.get(rid)
        if popen is None or rec is None:
            return False
        if rec.status != "running":
            return True
        killed = False
        if _IS_WINDOWS and job is not None:
            # Atomic kill: TerminateJobObject hits every process in the
            # job (main + 16 workers + any multiprocessing.spawn tracker)
            # in a single syscall, regardless of parent-child linkage.
            try:
                win32job.TerminateJobObject(job, 1)
                killed = True
            except Exception as exc:
                print(f"[pm] TerminateJobObject({rid}) failed: {exc}",
                      file=sys.stderr)
        if not killed and _IS_WINDOWS:
            # Fallback: walk parent-child tree.
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(popen.pid)],
                    capture_output=True, timeout=timeout,
                )
                killed = True
            except Exception:
                pass
        if not killed:
            popen.terminate()
            try:
                popen.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                popen.kill()
        try:
            popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        with self._lock:
            rec.status = "killed"
            rec.stopped_at = time.time()
            rec.returncode = popen.returncode
            if rid in self._jobs and _IS_WINDOWS:
                try:
                    win32api.CloseHandle(self._jobs[rid])
                except Exception:
                    pass
                self._jobs.pop(rid, None)
        return True

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._records.values()]

    def get(self, rid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            r = self._records.get(rid)
            return r.to_dict() if r else None

    def shutdown(self) -> None:
        """Kill everything (called on UI shutdown).  Job object will also
        do this automatically when the UI process exits."""
        with self._lock:
            rids = list(self._popens.keys())
        for rid in rids:
            self.kill(rid)

    # ----- internal -------------------------------------------------------- #
    def _watch(self, rid: str) -> None:
        with self._lock:
            popen = self._popens.get(rid)
            rec = self._records.get(rid)
        if popen is None or rec is None:
            return
        rc = popen.wait()
        with self._lock:
            if rec.status == "running":
                rec.status = "exited" if rc == 0 else "error"
                rec.stopped_at = time.time()
                rec.returncode = rc
            # Release the per-run Job Object handle once the main process
            # has exited.  All children inside the job have already been
            # killed by KILL_ON_JOB_CLOSE when we close the last handle.
            if rid in self._jobs and _IS_WINDOWS:
                try:
                    win32api.CloseHandle(self._jobs[rid])
                except Exception:
                    pass
                self._jobs.pop(rid, None)
