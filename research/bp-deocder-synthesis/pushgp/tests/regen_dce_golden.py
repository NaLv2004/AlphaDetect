"""Pre-generate golden Python-reference DCE outputs.

This is the **source of truth** for cpp DCE tests.  It runs the slow
Python reference loop once and pickles the inputs + reduced outputs to
``pushgp/tests/data/dce_golden.pkl``.

Tests then load this file and assert the cpp implementation produces
structurally identical reduced programs.  Cpp must not silently fall
back to Python — see ``dce.behavioral_reduce_bp(use_cpp=True)``.

Run from the project root::

    python -m pushgp.tests.regen_dce_golden

Expensive — takes minutes.  Re-run only when the Python reference
algorithm or oracle inputs change.
"""
from __future__ import annotations

from pathlib import Path
import json
import pickle
import sys
import time

import numpy as np


_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent.parent  # bp-deocder-synthesis/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pushgp.dce import behavioral_reduce_bp, reduce_populations_bp
from pushgp.program import program_length
from pushgp.serialize import dict_to_program, program_to_dict


# ---------------------------------------------------------------------------
# Shared fixture knobs.  Must match those used by the tests.
# ---------------------------------------------------------------------------

JSONL_PATH = (
    _ROOT / "results" / "logged_evolution"
    / "fromscratch_pop100_dedup" / "individuals.jsonl"
)
OUT_DIR = _THIS.parent / "data"
OUT_PATH = OUT_DIR / "dce_golden.pkl"

# T4 (per-pair) fixture
T4_N_PAIRS         = 10
T4_SNR_LIST        = (-2.0, -1.0, 0.0)
T4_MAX_ITER        = 3
T4_MAX_PASSES      = 800
T4_MAX_DECODE_EVALS = -1   # unlimited
T4_DECIMALS        = 6
T4_ORACLE_SEED     = 20_000

# Pop-batch fixture
POP_N              = 4
POP_PERM_SEED      = 9876
POP_RX_SEED        = 1234
POP_N_FRAMES       = 2
POP_MAX_ITER       = 4
POP_MAX_PASSES     = 200
POP_MAX_DECODE_EVALS = 2000
POP_DECIMALS       = 6
POP_BGN, POP_SET, POP_ZC = 2, 1, 2


def _load_pairs(n_pairs: int):
    pairs = []
    with JSONL_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if len(pairs) >= n_pairs:
                break
            rec = json.loads(line)
            if not rec.get("valid"):
                continue
            if rec.get("v2c_size", 0) < 8 or rec.get("c2v_size", 0) < 8:
                continue
            pairs.append((
                dict_to_program(rec["v2c"]),
                dict_to_program(rec["c2v"]),
                np.asarray(rec["log_constants"], dtype=np.float64),
            ))
    if len(pairs) < n_pairs:
        raise RuntimeError(
            f"only got {len(pairs)} valid pairs from {JSONL_PATH}"
        )
    return pairs


def _build_t4_oracle(par, snr_list, seed):
    """Same construction as test_dce._build_channel_data (1 frame/SNR)."""
    from ldpc_5g import HTYPE, bpsk_modulate, bpsk_llr
    from pushgp_ldpc.eval import _random_codeword
    htype = HTYPE[par.bgn - 1][par.set_idx - 1]
    rate = 0.5
    rx_llrs = []
    for snr_db in snr_list:
        rng = np.random.default_rng(seed + abs(int(snr_db * 1000)))
        sigma2 = 1.0 / (2.0 * rate * 10.0 ** (snr_db / 10.0))
        sigma = float(np.sqrt(sigma2))
        cw = _random_codeword(par, htype, rng)
        tx = bpsk_modulate(cw[2 * par.zc:])
        rx = tx + sigma * rng.standard_normal(tx.shape)
        llr_part = bpsk_llr(rx, sigma2)
        llr = np.zeros(par.cols, dtype=np.float64)
        llr[2 * par.zc:] = llr_part
        rx_llrs.append(llr)
    return rx_llrs


def _t4_meta():
    return {
        "schema": 1,
        "par_args": {"bgn": 2, "set_idx": 1, "zc": 2},
        "snr_list": list(T4_SNR_LIST),
        "oracle_seed": T4_ORACLE_SEED,
        "max_iter": T4_MAX_ITER,
        "max_passes": T4_MAX_PASSES,
        "max_decode_evals": T4_MAX_DECODE_EVALS,
        "decimals": T4_DECIMALS,
        "n_pairs": T4_N_PAIRS,
    }


def _load_cached_t4():
    """Return (entries, rx_llrs) from existing pickle if metadata matches."""
    if not OUT_PATH.exists():
        return [], None
    try:
        with OUT_PATH.open("rb") as fh:
            blob = pickle.load(fh)
    except Exception:
        return [], None
    t4 = blob.get("t4") if isinstance(blob, dict) else None
    if not isinstance(t4, dict):
        return [], None
    want = _t4_meta()
    for k, v in want.items():
        if k == "n_pairs":
            continue
        if t4.get(k) != v:
            print(f"[t4] cache meta mismatch on {k!r}; ignoring cache",
                  flush=True)
            return [], None
    entries = t4.get("entries") or []
    rx_llrs = [np.asarray(a, dtype=np.float64)
               for a in (t4.get("rx_llrs") or [])]
    return entries, rx_llrs


def _dump_t4(entries, rx_llrs):
    meta = _t4_meta()
    meta.pop("n_pairs", None)
    meta["rx_llrs"] = [a.tolist() for a in rx_llrs]
    meta["entries"] = entries
    blob = {"t4": meta, "format": 1}
    tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(OUT_PATH)


def gen_t4_golden():
    from ldpc_5g import build_parity
    print(f"[t4] loading {T4_N_PAIRS} pairs ...", flush=True)
    pairs = _load_pairs(T4_N_PAIRS)
    par = build_parity(bgn=2, set_idx=1, zc=2)
    cached_entries, cached_rx = _load_cached_t4()
    if cached_rx is not None:
        rx_llrs = cached_rx
        print(f"[t4] resuming from cache: {len(cached_entries)} entries",
              flush=True)
    else:
        rx_llrs = _build_t4_oracle(par, T4_SNR_LIST, T4_ORACLE_SEED)
    entries = list(cached_entries)
    done_keys = {(json.dumps(e["v_orig"], sort_keys=True),
                  json.dumps(e["c_orig"], sort_keys=True))
                 for e in entries}
    for i, (v, c, k) in enumerate(pairs):
        v_key = json.dumps(program_to_dict(v), sort_keys=True)
        c_key = json.dumps(program_to_dict(c), sort_keys=True)
        if (v_key, c_key) in done_keys:
            print(f"  [{i+1:2d}/{T4_N_PAIRS}] cached, skip", flush=True)
            continue
        t0 = time.time()
        v_red = behavioral_reduce_bp(
            v, "v2c", peer_prog=c, log_constants=k, par=par,
            rx_llrs=rx_llrs, max_iter=T4_MAX_ITER,
            max_passes=T4_MAX_PASSES,
            max_decode_evals=(None if T4_MAX_DECODE_EVALS < 0
                              else T4_MAX_DECODE_EVALS),
            decimals=T4_DECIMALS, use_cpp=False,
        )
        c_red = behavioral_reduce_bp(
            c, "c2v", peer_prog=v, log_constants=k, par=par,
            rx_llrs=rx_llrs, max_iter=T4_MAX_ITER,
            max_passes=T4_MAX_PASSES,
            max_decode_evals=(None if T4_MAX_DECODE_EVALS < 0
                              else T4_MAX_DECODE_EVALS),
            decimals=T4_DECIMALS, use_cpp=False,
        )
        dt = time.time() - t0
        print(f"  [{i+1:2d}/{T4_N_PAIRS}] "
              f"v {program_length(v):2d}->{program_length(v_red):2d}  "
              f"c {program_length(c):2d}->{program_length(c_red):2d}  "
              f"dt={dt:.1f}s", flush=True)
        entries.append({
            "v_orig": program_to_dict(v),
            "c_orig": program_to_dict(c),
            "log_constants": k.tolist(),
            "v_red_py": program_to_dict(v_red),
            "c_red_py": program_to_dict(c_red),
        })
        _dump_t4(entries, rx_llrs)
    _dump_t4(entries, rx_llrs)
    return {
        "schema": 1,
        "par_args": {"bgn": 2, "set_idx": 1, "zc": 2},
        "snr_list": list(T4_SNR_LIST),
        "oracle_seed": T4_ORACLE_SEED,
        "max_iter": T4_MAX_ITER,
        "max_passes": T4_MAX_PASSES,
        "max_decode_evals": T4_MAX_DECODE_EVALS,
        "decimals": T4_DECIMALS,
        "rx_llrs": [a.tolist() for a in rx_llrs],
        "entries": entries,
    }


def gen_pop_golden():
    from ldpc_5g import build_parity
    print(f"[pop] loading {POP_N} pairs ...", flush=True)
    pairs = _load_pairs(POP_N)
    par = build_parity(bgn=POP_BGN, set_idx=POP_SET, zc=POP_ZC)
    pop_v = [p[0] for p in pairs]
    pop_c = [p[1] for p in pairs]
    pop_k = [p[2] for p in pairs]
    rng_rx = np.random.default_rng(POP_RX_SEED)
    rx = [rng_rx.standard_normal(par.cols).astype(np.float64)
          for _ in range(POP_N_FRAMES)]
    t0 = time.time()
    new_v, new_c, st_v, st_c = reduce_populations_bp(
        pop_v, pop_c, pop_k,
        par=par, rx_llrs=rx,
        max_iter=POP_MAX_ITER, max_passes=POP_MAX_PASSES,
        max_decode_evals=POP_MAX_DECODE_EVALS, decimals=POP_DECIMALS,
        threads=1, use_cpp=False,
        rng=np.random.default_rng(POP_PERM_SEED),
    )
    dt = time.time() - t0
    print(f"  python serial done in {dt:.1f}s", flush=True)
    for i in range(POP_N):
        print(f"  [V{i}] {st_v[i].size_before}->{st_v[i].size_after}  "
              f"[C{i}] {st_c[i].size_before}->{st_c[i].size_after}",
              flush=True)
    return {
        "schema": 1,
        "par_args": {"bgn": POP_BGN, "set_idx": POP_SET, "zc": POP_ZC},
        "perm_seed": POP_PERM_SEED,
        "rx_seed": POP_RX_SEED,
        "n_frames": POP_N_FRAMES,
        "max_iter": POP_MAX_ITER,
        "max_passes": POP_MAX_PASSES,
        "max_decode_evals": POP_MAX_DECODE_EVALS,
        "decimals": POP_DECIMALS,
        "rx_llrs": [a.tolist() for a in rx],
        "pop_v_orig": [program_to_dict(p) for p in pop_v],
        "pop_c_orig": [program_to_dict(p) for p in pop_c],
        "log_constants": [k.tolist() for k in pop_k],
        "pop_v_red_py": [program_to_dict(p) for p in new_v],
        "pop_c_red_py": [program_to_dict(p) for p in new_c],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=== T4 per-pair golden ===", flush=True)
    t4 = gen_t4_golden()
    blob = {"t4": t4, "format": 1}
    with OUT_PATH.open("wb") as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.1f} KB)  "
          f"in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
