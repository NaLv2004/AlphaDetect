"""FastAPI backend for the Push-GP evolution UI.

Pure HTTP+polling (no WebSocket) — keeps deps minimal and is plenty fast
for log/jsonl tailing at the rates we generate.

Endpoints
---------
GET  /                              -> index.html (single-file frontend)
GET  /api/health
GET  /api/scripts                   -> static info: default flags, choices
POST /api/runs                      -> launch new run (body = full param dict)
GET  /api/runs                      -> list (active + finished)
GET  /api/runs/{rid}                -> single record
DELETE /api/runs/{rid}              -> kill (taskkill /T)
GET  /api/runs/{rid}/log?stream=    -> stdout|stderr, ?offset=N
GET  /api/runs/{rid}/baseline       -> baseline.json
GET  /api/runs/{rid}/meta           -> meta.json
GET  /api/runs/{rid}/summary?since= -> incremental gen_summary rows
GET  /api/runs/{rid}/population?gen=
GET  /api/runs/{rid}/individual?gen=&idx=
GET  /api/runs/{rid}/champion       -> champion.json (when finished)
GET  /api/runs/{rid}/final          -> final_summary.json (when finished)
GET  /api/discover                  -> list of run-dirs on disk not yet
                                       tracked by this session
POST /api/attach                    -> attach to an existing run-dir
                                       (read-only — no process to manage)
"""
from __future__ import annotations

import atexit
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_HERE = Path(__file__).resolve().parent
_BPS = _HERE.parent
_REPO = _BPS.parent.parent
_RESULTS = _BPS / "results" / "logged_evolution"
_PY = sys.executable  # the python launching this UI is also used for children
_EVO_SCRIPT = _BPS / "experiments" / "run_logged_evolution.py"
_PARSE_SCRIPT = _BPS / "experiments" / "parse_individuals_math.py"

# Make `pushgp.*` importable so the parsed-math endpoint can re-use the
# trace / math helpers (read-only — no modification to existing code).
if str(_BPS) not in sys.path:
    sys.path.insert(0, str(_BPS))

from experiments.parse_individuals_math import _parse_record  # noqa: E402

from ui.process_manager import ProcessManager  # noqa: E402

# --------------------------------------------------------------------------- #
# Globals
# --------------------------------------------------------------------------- #
pm = ProcessManager(max_concurrent=4)
app = FastAPI(title="PushGP Evolution UI", version="1.0")

atexit.register(pm.shutdown)

# Cache for /population: (rid, gen) -> (mtime, size, rows)
_POP_CACHE: Dict[tuple, tuple] = {}
_POP_CACHE_LOCK = __import__("threading").Lock()
_POP_CACHE_MAX = 32  # LRU cap (FIFO drop)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class LaunchParams(BaseModel):
    run_name: Optional[str] = None
    pop_size: int = 200
    gens: int = 20
    seed: int = 2025
    elitism: int = 3
    tournament_k: int = 4
    p_crossover: float = 0.7
    n_mutations: int = 2
    p_const_tweak: float = 0.25
    snr_list: str = "-3,-2,-1"
    n_frames: int = 6
    max_iter: int = 8
    rand_min_size: int = 4
    rand_max_size: int = 30
    workers: int = 16
    bgn: int = 2
    set_idx: int = 1
    zc: int = 2

    # DCE
    dce_bp: bool = True
    dce_bp_max_iter: int = 8
    dce_bp_decimals: int = 6
    dce_bp_max_passes: int = 800
    dce_bp_max_decode_evals: int = -1
    dce_bp_threads: int = 0
    dce_bp_use_cpp: bool = True
    dce_bp_snr_db: float = -2.0
    dce_bp_n_frames: int = 1

    # other
    cpp_seeder: bool = True
    from_scratch: bool = True
    label: Optional[str] = None
    start_watcher: bool = True

    def build_cmdline(self, run_name: str) -> List[str]:
        cmd = [
            _PY, "-B", "-u", str(_EVO_SCRIPT),
            "--run-name", run_name,
            "--pop-size", str(self.pop_size),
            "--gens", str(self.gens),
            "--seed", str(self.seed),
            "--elitism", str(self.elitism),
            "--tournament-k", str(self.tournament_k),
            "--p-crossover", str(self.p_crossover),
            "--n-mutations", str(self.n_mutations),
            "--p-const-tweak", str(self.p_const_tweak),
            "--snr-list=" + self.snr_list,
            "--n-frames", str(self.n_frames),
            "--max-iter", str(self.max_iter),
            "--rand-min-size", str(self.rand_min_size),
            "--rand-max-size", str(self.rand_max_size),
            "--workers", str(self.workers),
            "--bgn", str(self.bgn),
            "--set-idx", str(self.set_idx),
            "--zc", str(self.zc),
        ]
        if self.cpp_seeder:
            cmd.append("--cpp-seeder")
        if not self.from_scratch:
            cmd.append("--use-oms-seed")
        if self.dce_bp:
            cmd += [
                "--dce-bp",
                "--dce-bp-max-iter", str(self.dce_bp_max_iter),
                "--dce-bp-decimals", str(self.dce_bp_decimals),
                "--dce-bp-max-passes", str(self.dce_bp_max_passes),
                "--dce-bp-max-decode-evals", str(self.dce_bp_max_decode_evals),
                "--dce-bp-threads", str(self.dce_bp_threads),
                "--dce-bp-snr-db=" + str(self.dce_bp_snr_db),
                "--dce-bp-n-frames", str(self.dce_bp_n_frames),
            ]
            if not self.dce_bp_use_cpp:
                cmd.append("--dce-bp-no-cpp")
        return cmd


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run_dir(rid: str) -> Path:
    return _RESULTS / rid


def _read_jsonl(p: Path, since: int = 0) -> tuple[list[dict], int]:
    """Return (rows, new_offset_in_bytes).  since = byte offset to start
    reading from."""
    if not p.exists():
        return [], 0
    out: list[dict] = []
    with p.open("rb") as fh:
        fh.seek(since)
        for raw in fh:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        new_offset = fh.tell()
    return out, new_offset


def _read_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _tail_bytes(p: Path, offset: int = 0, max_bytes: int = 64 * 1024
                ) -> tuple[str, int]:
    if not p.exists():
        return "", 0
    size = p.stat().st_size
    if offset >= size:
        return "", size
    with p.open("rb") as fh:
        fh.seek(offset)
        chunk = fh.read(max_bytes)
        new_offset = fh.tell()
    return chunk.decode("utf-8", errors="replace"), new_offset


# --------------------------------------------------------------------------- #
# Static
# --------------------------------------------------------------------------- #
@app.get("/", response_class=FileResponse)
def root():
    return FileResponse(_HERE / "index.html")


app.mount("/ui", StaticFiles(directory=str(_HERE)), name="ui")


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"ok": True, "python": _PY, "results_dir": str(_RESULTS)}


@app.get("/api/scripts")
def scripts():
    return {
        "python": _PY,
        "evo_script": str(_EVO_SCRIPT),
        "parse_script": str(_PARSE_SCRIPT),
        "max_concurrent": pm.max_concurrent,
    }


@app.get("/api/budget")
def budget():
    """CPU worker budget snapshot for the UI to render & validate."""
    return pm.budget_status()


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
@app.post("/api/runs")
def launch_run(params: LaunchParams):
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = params.run_name or f"ui_run_{ts}"
    rid = run_name
    log_stdout = _RESULTS / f"{run_name}.console.log"
    log_stderr = _RESULTS / f"{run_name}.console.err.log"
    cmd = params.build_cmdline(run_name)
    try:
        rec = pm.launch(
            cmdline=cmd,
            cwd=str(_BPS),
            log_stdout=log_stdout,
            log_stderr=log_stderr,
            label=params.label or run_name,
            rid=rid,
            meta={"params": params.model_dump(),
                  "watcher_rid": None},
            workers=params.workers,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # optionally start watcher (--watch)
    # NOTE: the watcher exits immediately if individuals.jsonl is missing,
    # which is always the case in the first few seconds before the
    # evolution main loop writes the first generation.  We therefore
    # defer the actual `pm.launch` until the file appears (or 120s).
    if params.start_watcher:
        w_rid = rid + "__watcher"
        w_stdout = _RESULTS / f"{run_name}.watcher.console.log"
        w_stderr = _RESULTS / f"{run_name}.watcher.console.err.log"
        w_cmd = [
            _PY, "-B", "-u", str(_PARSE_SCRIPT), run_name,
            "--watch", "--poll-interval", "3.0", "--max-chars", "500",
        ]
        rec.meta["watcher_rid"] = w_rid
        rec.meta["watcher_status"] = "deferred"

        def _deferred_launch_watcher():
            import time as _t
            wait_for = _run_dir(rid) / "individuals.jsonl"
            deadline = _t.time() + 120.0
            while _t.time() < deadline:
                if wait_for.exists() and wait_for.stat().st_size > 0:
                    break
                _t.sleep(1.0)
            else:
                rec.meta["watcher_status"] = "timeout"
                return
            try:
                pm.launch(cmdline=w_cmd, cwd=str(_BPS),
                          log_stdout=w_stdout, log_stderr=w_stderr,
                          label=f"watcher[{run_name}]", rid=w_rid,
                          meta={"parent_rid": rid},
                          workers=0)
                rec.meta["watcher_status"] = "running"
            except RuntimeError as e:
                rec.meta["watcher_status"] = f"failed: {e}"

        import threading as _th
        _th.Thread(target=_deferred_launch_watcher, daemon=True).start()
    return rec.to_dict()


@app.get("/api/runs")
def list_runs():
    return pm.list()


@app.get("/api/runs/{rid}")
def get_run(rid: str):
    rec = pm.get(rid)
    if rec is None:
        # maybe attached / discoverable
        if _run_dir(rid).exists():
            return {"rid": rid, "label": rid, "status": "external",
                    "pid": None,
                    "log_stdout": str(_RESULTS / f"{rid}.console.log"),
                    "log_stderr": str(_RESULTS / f"{rid}.console.err.log"),
                    "started_at_iso": "",
                    "meta": {}}
        raise HTTPException(status_code=404, detail="run not found")
    return rec


@app.delete("/api/runs/{rid}")
def kill_run(rid: str):
    rec = pm.get(rid)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    # also kill watcher if any
    w_rid = (rec.get("meta") or {}).get("watcher_rid")
    if w_rid:
        pm.kill(w_rid)
    ok = pm.kill(rid)
    return {"ok": ok, "rid": rid}


@app.get("/api/runs/{rid}/log")
def get_log(rid: str,
            stream: str = Query("stdout", pattern="^(stdout|stderr)$"),
            offset: int = 0,
            max_bytes: int = 64 * 1024):
    p = _RESULTS / (f"{rid}.console.log" if stream == "stdout"
                    else f"{rid}.console.err.log")
    text, new_off = _tail_bytes(p, offset=offset, max_bytes=max_bytes)
    return {"text": text, "offset": new_off, "size": (p.stat().st_size
                                                      if p.exists() else 0)}


@app.get("/api/runs/{rid}/watcher_log")
def get_watcher_log(rid: str, offset: int = 0, max_bytes: int = 64 * 1024):
    p = _RESULTS / f"{rid}.watcher.console.log"
    text, new_off = _tail_bytes(p, offset=offset, max_bytes=max_bytes)
    return {"text": text, "offset": new_off, "size": (p.stat().st_size
                                                      if p.exists() else 0)}


@app.get("/api/runs/{rid}/baseline")
def get_baseline(rid: str):
    d = _read_json(_run_dir(rid) / "baseline.json")
    if d is None:
        raise HTTPException(status_code=404, detail="no baseline yet")
    return d


@app.get("/api/runs/{rid}/meta")
def get_meta(rid: str):
    d = _read_json(_run_dir(rid) / "meta.json")
    if d is None:
        raise HTTPException(status_code=404, detail="no meta yet")
    return d


@app.get("/api/runs/{rid}/summary")
def get_summary(rid: str, since: int = 0):
    rows, new_off = _read_jsonl(_run_dir(rid) / "gen_summary.jsonl",
                                since=since)
    return {"rows": rows, "offset": new_off}


@app.get("/api/runs/{rid}/population")
def get_population(rid: str, gen: int = Query(...)):
    """Return one row per individual at the given generation, with the
    heavy push-code stripped (only sizes + fitness + ber).  Suitable for
    the population panel / scatter / histogram.

    Cached by (rid, gen, mtime, size).  Population for a given (rid, gen)
    is append-only once that generation is written, so an mtime+size key
    is a safe invalidation criterion.
    """
    p = _run_dir(rid) / "individuals.jsonl"
    if not p.exists():
        return {"gen": gen, "rows": [], "count": 0}
    st = p.stat()
    key = (rid, gen)
    with _POP_CACHE_LOCK:
        cached = _POP_CACHE.get(key)
        if cached is not None and cached[0] == st.st_mtime and cached[1] == st.st_size:
            rows = cached[2]
            return {"gen": gen, "rows": rows, "count": len(rows),
                    "cached": True}
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(rec.get("gen", -1)) != gen:
                continue
            out.append({
                "idx": rec.get("idx"),
                "fitness": rec.get("fitness"),
                "valid": rec.get("valid"),
                "v2c_size": rec.get("v2c_size"),
                "c2v_size": rec.get("c2v_size"),
                "v2c_max_depth": rec.get("v2c_max_depth"),
                "c2v_max_depth": rec.get("c2v_max_depth"),
                "ber_per_snr": rec.get("ber_per_snr"),
                "fer_per_snr": rec.get("fer_per_snr"),
                "error": rec.get("error"),
            })
    out.sort(key=lambda r: (r.get("fitness") if r.get("fitness") is not None
                            else float("inf")))
    with _POP_CACHE_LOCK:
        if len(_POP_CACHE) >= _POP_CACHE_MAX:
            # FIFO eviction
            try:
                _POP_CACHE.pop(next(iter(_POP_CACHE)))
            except StopIteration:
                pass
        _POP_CACHE[key] = (st.st_mtime, st.st_size, out)
    return {"gen": gen, "rows": out, "count": len(out), "cached": False}


@app.get("/api/runs/{rid}/generations")
def get_generations(rid: str):
    """Return the set of generations present in individuals.jsonl
    (for the gen-selector dropdown)."""
    p = _run_dir(rid) / "individuals.jsonl"
    if not p.exists():
        return {"generations": []}
    gens: Dict[int, int] = {}
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            g = int(rec.get("gen", -1))
            gens[g] = gens.get(g, 0) + 1
    return {"generations": sorted(gens.items())}


# (rid) -> (mtime, size, list[dict])  for /best_sizes
_BEST_SIZES_CACHE: Dict[str, tuple] = {}
_BEST_SIZES_LOCK = __import__("threading").Lock()


@app.get("/api/runs/{rid}/best_sizes")
def get_best_sizes(rid: str):
    """Per-generation best-fitness individual's v2c/c2v size (post-DCE).

    `gen_summary.jsonl` doesn't record these directly, so we derive them
    by scanning `individuals.jsonl` once and grouping by gen.  Cached on
    mtime+size of the source file.
    """
    p = _run_dir(rid) / "individuals.jsonl"
    if not p.exists():
        return {"rows": []}
    st = p.stat()
    with _BEST_SIZES_LOCK:
        cached = _BEST_SIZES_CACHE.get(rid)
        if cached is not None and cached[0] == st.st_mtime and cached[1] == st.st_size:
            return {"rows": cached[2], "cached": True}
    by_gen: Dict[int, dict] = {}
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                g = int(rec.get("gen", -1))
                f = rec.get("fitness")
            except (TypeError, ValueError):
                continue
            if f is None:
                continue
            cur = by_gen.get(g)
            if cur is None or float(f) < cur["best_fit"]:
                by_gen[g] = {
                    "gen": g,
                    "best_fit": float(f),
                    "best_v2c_size": rec.get("v2c_size"),
                    "best_c2v_size": rec.get("c2v_size"),
                    "best_idx": rec.get("idx"),
                    "best_ber_per_snr": rec.get("ber_per_snr"),
                }
    rows = [by_gen[k] for k in sorted(by_gen.keys())]
    with _BEST_SIZES_LOCK:
        _BEST_SIZES_CACHE[rid] = (st.st_mtime, st.st_size, rows)
    return {"rows": rows, "cached": False}


@app.get("/api/runs/{rid}/individual")
def get_individual(rid: str, gen: int, idx: int):
    """Return the raw record + parsed math + push-code dump for the
    requested (gen, idx)."""
    p = _run_dir(rid) / "individuals.jsonl"
    if not p.exists():
        raise HTTPException(status_code=404, detail="no individuals.jsonl")
    target: Optional[dict] = None
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(rec.get("gen", -1)) == gen and int(rec.get("idx", -1)) == idx:
                target = rec
                break
    if target is None:
        raise HTTPException(status_code=404,
                            detail=f"individual gen={gen} idx={idx} not found")
    parsed = _parse_record(target)
    return {
        "raw": {k: target[k] for k in target
                if k not in ("v2c", "c2v")},
        "v2c_push": target.get("v2c"),
        "c2v_push": target.get("c2v"),
        "parsed": parsed,
    }


@app.get("/api/runs/{rid}/champion")
def get_champion(rid: str):
    d = _read_json(_run_dir(rid) / "champion.json")
    if d is None:
        raise HTTPException(status_code=404, detail="champion not written yet")
    return d


@app.get("/api/runs/{rid}/final")
def get_final(rid: str):
    d = _read_json(_run_dir(rid) / "final_summary.json")
    if d is None:
        raise HTTPException(status_code=404, detail="final not written yet")
    return d


# --------------------------------------------------------------------------- #
# Discover existing run dirs on disk (so the UI can attach to a run that
# the user started outside the UI, e.g. via the original evo_run.bat).
# --------------------------------------------------------------------------- #
@app.get("/api/discover")
def discover():
    if not _RESULTS.exists():
        return {"runs": []}
    out: List[Dict[str, Any]] = []
    tracked = {r["rid"] for r in pm.list()}
    for entry in _RESULTS.iterdir():
        if not entry.is_dir():
            continue
        if not (entry / "meta.json").exists():
            continue
        rid = entry.name
        out.append({
            "rid": rid,
            "tracked": rid in tracked,
            "has_champion": (entry / "champion.json").exists(),
            "has_final": (entry / "final_summary.json").exists(),
            "mtime": entry.stat().st_mtime,
        })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return {"runs": out}
