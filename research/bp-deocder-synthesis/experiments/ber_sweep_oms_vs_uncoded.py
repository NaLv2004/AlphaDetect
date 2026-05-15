"""BER sweep for OMS BP decoder + uncoded BPSK on the SAME LDPC code
the from-scratch evolution used (BG2 set 1, Zc=2, N=104, M=84, R=0.5).

Per user request:
  - Eb/N0 sweep: -1.0 : 0.25 : 1.0 dB  (9 points)
  - Min frame errors per point: 50
  - OMS BER reported on info bits (K = 20 per codeword)
  - Uncoded BPSK BER reported on transmitted symbols (100 per codeword)
    via direct sign decision on the SAME y used by OMS

Channel model matches `pushgp_ldpc/eval.py::_channel_inputs`:
  sigma^2 = 1 / (2 * R * 10^(EbN0/10))
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # add bp-deocder-synthesis to path

from ldpc_5g import (
    HTYPE,
    bpsk_llr, bpsk_modulate,
    build_oms_context, build_parity,
    decode_oms_fast, encode_codeblock,
)


def random_codeword(par, htype: int, rng: np.random.Generator) -> np.ndarray:
    Kb = 10 if par.bgn == 2 else 22
    K = Kb * par.zc
    info = rng.integers(0, 2, size=K, dtype=np.int8)
    cw_punct = encode_codeblock(info, par, htype)
    cw_full = np.concatenate([info[: 2 * par.zc], cw_punct]).astype(np.int8)
    return cw_full, info  # full cw (length 104), info (length K=20)


def run_point(ebno_db: float, par, ctx, htype: int,
              max_iter: int, min_frame_err: int, max_frames: int,
              seed: int) -> dict:
    rng = np.random.default_rng(seed + abs(int(ebno_db * 1000)))
    R = 0.5
    sigma2 = 1.0 / (2.0 * R * 10.0 ** (ebno_db / 10.0))
    sigma = math.sqrt(sigma2)
    Kb = 10 if par.bgn == 2 else 22
    K = Kb * par.zc          # info bits per codeword (= 20)
    N_tx = par.cols - 2 * par.zc  # transmitted bits (= 100)

    # OMS counters
    oms_frames = 0
    oms_frame_err = 0
    oms_bit_err = 0
    oms_iters = 0
    # Uncoded BPSK counters (on tx symbols)
    uncoded_bits = 0
    uncoded_bit_err = 0
    uncoded_frames = 0
    uncoded_frame_err = 0  # frame-error if any bit wrong

    t0 = time.time()
    while oms_frame_err < min_frame_err and oms_frames < max_frames:
        cw_full, info = random_codeword(par, htype, rng)
        tx_bits = cw_full[2 * par.zc :].astype(np.int64)  # 100 bits
        tx_sym = bpsk_modulate(tx_bits)                    # +1/-1
        rx = tx_sym + sigma * rng.standard_normal(tx_sym.shape)

        # ---- (1) Uncoded BPSK: hard-decide on rx ----
        # bpsk_modulate convention: bit=0 -> +1, bit=1 -> -1
        # so hard decision: bit_hat = (rx < 0)
        bit_hat = (rx < 0.0).astype(np.int64)
        bit_err = int(np.sum(bit_hat != tx_bits))
        uncoded_bits += N_tx
        uncoded_bit_err += bit_err
        uncoded_frames += 1
        if bit_err > 0:
            uncoded_frame_err += 1

        # ---- (2) OMS BP decode ----
        llr_part = bpsk_llr(rx, sigma2)
        llr_in = np.zeros(par.cols, dtype=np.float64)
        llr_in[2 * par.zc :] = llr_part
        llr_post, iters = decode_oms_fast(llr_in, ctx, max_iter, offset=0.25)
        oms_iters += iters
        # Decode info bits (positions 0..K-1 in cw_full)
        info_hat = (llr_post[:K] < 0.0).astype(np.int8)
        bit_err_oms = int(np.sum(info_hat != info))
        oms_bit_err += bit_err_oms
        oms_frames += 1
        if bit_err_oms > 0:
            oms_frame_err += 1

        if oms_frames % 200 == 0:
            elapsed = time.time() - t0
            print(f"    EbN0={ebno_db:+.2f}dB  cw={oms_frames:6d}  "
                  f"OMS_FE={oms_frame_err:4d}  BE={oms_bit_err:5d}  "
                  f"BER={oms_bit_err/(oms_frames*K):.3e}  "
                  f"unc_BER={uncoded_bit_err/uncoded_bits:.3e}  "
                  f"avg_it={oms_iters/oms_frames:.1f}  t={elapsed:.0f}s",
                  flush=True)

    return {
        "ebno_db": ebno_db,
        "oms": {
            "frames": oms_frames,
            "frame_errors": oms_frame_err,
            "bit_errors": oms_bit_err,
            "info_bits": oms_frames * K,
            "BER": oms_bit_err / (oms_frames * K),
            "FER": oms_frame_err / oms_frames,
            "avg_iter": oms_iters / oms_frames,
        },
        "uncoded": {
            "bits": uncoded_bits,
            "bit_errors": uncoded_bit_err,
            "frame_errors": uncoded_frame_err,
            "frames": uncoded_frames,
            "BER": uncoded_bit_err / uncoded_bits,
            "FER": uncoded_frame_err / uncoded_frames,
        },
    }


def main():
    par = build_parity(bgn=2, set_idx=1, zc=2)
    ctx = build_oms_context(par)
    htype = HTYPE[par.bgn - 1][par.set_idx - 1]
    print(f"[code] BG2 set1 Zc=2  N={par.cols} M={par.rows}  R=0.5  "
          f"K_info={10*par.zc}  N_tx={par.cols - 2*par.zc}")

    ebno_list = [round(-1.0 + 0.25 * k, 2) for k in range(9)]  # -1, -0.75, ..., 1.0
    print(f"[sweep] Eb/N0 = {ebno_list}  min_frame_err=50  max_iter=8")

    results = []
    for ebno in ebno_list:
        print(f"\n=== Eb/N0 = {ebno:+.2f} dB ===", flush=True)
        r = run_point(ebno, par, ctx, htype,
                      max_iter=8, min_frame_err=50, max_frames=200000,
                      seed=12345)
        results.append(r)
        o = r["oms"]; u = r["uncoded"]
        print(f"  -> OMS:     BER={o['BER']:.4e}  FER={o['FER']:.4e}  "
              f"({o['bit_errors']}/{o['info_bits']} bits, "
              f"{o['frame_errors']}/{o['frames']} frames, avg_it={o['avg_iter']:.1f})")
        print(f"  -> Uncoded: BER={u['BER']:.4e}  FER={u['FER']:.4e}  "
              f"({u['bit_errors']}/{u['bits']} bits, "
              f"{u['frame_errors']}/{u['frames']} frames)")

    # ---- summary table ----
    print("\n\n" + "=" * 88)
    print(f"{'Eb/N0(dB)':>10} | {'OMS BER':>12} {'OMS FER':>12} {'avg_it':>7} | "
          f"{'Uncoded BER':>12} {'Uncoded FER':>12}")
    print("-" * 88)
    for r in results:
        o = r["oms"]; u = r["uncoded"]
        print(f"{r['ebno_db']:>+10.2f} | {o['BER']:>12.4e} {o['FER']:>12.4e} "
              f"{o['avg_iter']:>7.2f} | {u['BER']:>12.4e} {u['FER']:>12.4e}")
    print("=" * 88)

    # also dump JSON
    import json
    out_path = os.path.join(_HERE, "..", "..", "results", "logged_evolution",
                            "ber_sweep_oms_vs_uncoded.json")
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
