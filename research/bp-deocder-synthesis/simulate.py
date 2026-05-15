"""End-to-end LDPC simulation: 3GPP encoder → BPSK → AWGN → OMS BP decoder.

Default sweep matches the user acceptance target (code length 1024, rate 0.5,
single CB, BPSK SISO AWGN).  Stops at each Eb/N0 once a configurable minimum
number of frame errors is collected, mirroring the C++ ``thres_frame`` loop.

Usage:
    python simulate.py                 # run default sweep
    python simulate.py --min 0 --max 4 --step 0.5 --thres 50

Run from inside the ``AutoGenOld`` conda environment so numpy is available.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

# Allow `python simulate.py` from the package directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ldpc_5g import (
    derive_params, build_parity, build_oms_context,
    encode_codeblock, cbs_rate_match, cbs_rate_recover,
    compute_k0, compute_E_array,
    bpsk_modulate, awgn, bpsk_llr,
    decode_oms_fast, parity_check_syndrome, HTYPE,
)


def _sigma_from_ebno(ebno_db: float, code_rate: float) -> float:
    """sigma s.t. for BPSK with Es=1 over real AWGN, Eb/N0 = ebno_db.

    Es = 1, Eb = Es / R = 1/R, N0 = 2*sigma^2 (real channel),
    Eb/N0 = 1 / (2 R sigma^2)  →  sigma^2 = 1 / (2 R 10^(EbN0/10)).
    """
    return math.sqrt(1.0 / (2.0 * code_rate * 10.0 ** (ebno_db / 10.0)))


def _gen_cb_infobits(K_cb_bit: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Generate K-long info vector with first K_cb_bit random bits and
    trailing K-K_cb_bit filler positions marked as -1.  CRC is omitted in
    this study (we measure raw decoded-bit FER on the K_cb_bit info bits)."""
    bits = -np.ones(K, dtype=np.int64)
    bits[:K_cb_bit] = rng.integers(0, 2, size=K_cb_bit)
    return bits


def run_point(ebno_db: float, params, par, ctx,
              max_iter: int, min_frame_errors: int, max_frames: int,
              seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sigma = _sigma_from_ebno(ebno_db, params.code_rate)
    sigma2 = sigma * sigma
    htype = HTYPE[params.bgn - 1][params.set_idx - 1]

    k0 = compute_k0(params.bgn, params.zc, params.N_punctured, rv=0)
    e_arr = compute_E_array(params.C, params.outlen, qm=1, nlayers=1)
    if params.C != 1:
        raise NotImplementedError("This simulator only exercises the single-CB path (C==1).")
    E = e_arr[0]
    Kd = params.K_cb_bit - 2 * params.zc  # filler region start (post-puncture)
    K_after_punct = params.K - 2 * params.zc

    n_frames = 0
    n_frame_errors = 0
    n_bit_errors = 0
    iters_total = 0
    t0 = time.time()

    while n_frame_errors < min_frame_errors and n_frames < max_frames:
        # --- TX ---
        info = _gen_cb_infobits(params.K_cb_bit, params.K, rng)
        cw = encode_codeblock(info, par, htype)  # length N_punctured, -1 fillers
        tx_bits = cbs_rate_match(cw, E, k0, params.N_punctured, qm=1)
        tx_sym = bpsk_modulate(tx_bits)

        # --- Channel ---
        rx = awgn(tx_sym, sigma, rng)

        # --- RX: channel LLR → rate recover → (prepend 2*Zc punctured
        #         zero LLRs to obtain length-N input) → decode ---
        llr_chan = bpsk_llr(rx, sigma2)
        llr_punctured = cbs_rate_recover(llr_chan, params.N_punctured, E,
                                         K_after_punct, Kd, k0,
                                         params.N_punctured, qm=1)
        # The C++ ``modified_RateRecoverLDPC`` writes LLR_in_C[0..2*Zc-1]=0
        # and LLR_in_C[2*Zc..N-1] = rate-recovered.  Mirror that here.
        llr_in_full = np.concatenate(
            (np.zeros(2 * params.zc, dtype=np.float64), llr_punctured)
        )
        llr_post, iters = decode_oms_fast(llr_in_full, ctx, max_iter)
        iters_total += iters

        # Compare ALL K_cb_bit info bits (incl. the first 2*Zc that were
        # punctured at TX and reconstructed by BP) against the source info.
        # Filler positions (info==-1, indices K_cb_bit..K-1) are not part of
        # the user payload.
        info_payload = np.where(info[:params.K_cb_bit] == -1, 0, info[:params.K_cb_bit])
        decoded = (llr_post[:params.K_cb_bit] < 0.0).astype(np.int64)
        bit_err = int(np.sum(decoded != info_payload))
        n_bit_errors += bit_err
        if bit_err > 0:
            n_frame_errors += 1
        n_frames += 1

        if n_frames % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  EbN0={ebno_db:5.2f}dB  frames={n_frames:6d}  "
                f"FE={n_frame_errors:4d}  BE={n_bit_errors:6d}  "
                f"FER={n_frame_errors / n_frames:.3e}  "
                f"BER={n_bit_errors / (n_frames * params.K_cb_bit):.3e}  "
                f"avg_it={iters_total / n_frames:.1f}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    info_per_frame = params.K_cb_bit
    return {
        "ebno_db": ebno_db,
        "frames": n_frames,
        "frame_errors": n_frame_errors,
        "bit_errors": n_bit_errors,
        "fer": n_frame_errors / n_frames if n_frames else float("nan"),
        "ber": n_bit_errors / (n_frames * info_per_frame) if n_frames else float("nan"),
        "avg_iter": iters_total / n_frames if n_frames else float("nan"),
        "elapsed_s": elapsed,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--A", type=int, default=512, help="Info bits per TB (no TB-CRC)")
    p.add_argument("--rate", type=float, default=0.5, help="Code rate (R = A / outlen)")
    p.add_argument("--min", dest="ebno_min", type=float, default=3.0)
    p.add_argument("--max", dest="ebno_max", type=float, default=3.0)
    p.add_argument("--step", dest="ebno_step", type=float, default=0.5)
    p.add_argument("--max-iter", type=int, default=25, help="BP iterations")
    p.add_argument("--thres", type=int, default=100, help="Min frame errors per point")
    p.add_argument("--max-frames", type=int, default=200000)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    params = derive_params(args.A, args.rate)
    print("=== 3GPP NR LDPC parameters ===")
    print(f"  A={params.A}  rate={params.code_rate}  L(CRC, unused here)={params.L}  B={params.B}")
    print(f"  bgn={params.bgn}  setIdx={params.set_idx}  Zc={params.zc}  Kb={params.Kb}")
    print(f"  C={params.C}  K_cb_bit={params.K_cb_bit}  K={params.K}  fillers/CB={params.num_filler_per_cb}")
    print(f"  Mb={params.Mb}  Nb={params.Nb}  N={params.N}  M={params.M}")
    print(f"  N_punctured={params.N_punctured}  outlen={params.outlen}")

    par = build_parity(params.bgn, params.set_idx, params.zc)

    # Self-check: encode 32 random CBs and verify H @ c = 0 over GF(2)
    # (filler positions zeroed for the check).
    rng = np.random.default_rng(0)
    htype = HTYPE[params.bgn - 1][params.set_idx - 1]
    print("Encoder self-test (H @ c = 0 over GF(2) for 32 random CBs)...")
    for trial in range(32):
        info = _gen_cb_infobits(params.K_cb_bit, params.K, rng)
        cw = encode_codeblock(info, par, htype)  # length N_punctured w/ -1 fillers
        # Reconstruct the full lifted codeword (length N) for the syndrome
        # check.  Filler bits behave as zeros for parity-check purposes.
        full = np.zeros(params.N, dtype=np.int64)
        info_zeroed = np.where(info == -1, 0, info)
        full[:params.K] = info_zeroed
        # The first 2*Zc bits of the codeword == the first 2*Zc info bits
        # (they are punctured at TX but still part of the codeword for H).
        # The rest (positions 2*Zc .. N-1) is what ``encode_codeblock``
        # returned, with filler -1 replaced by 0 here.
        parity_part = np.where(cw == -1, 0, cw)
        full[2 * params.zc:params.N] = parity_part
        syn = parity_check_syndrome(par, full)
        if syn.any():
            raise RuntimeError(
                f"Encoder self-check failed at trial {trial}: syndrome weight {int(syn.sum())}"
            )
    print("  OK: 32/32 trials produce H @ c = 0.\n")

    # Build vectorised decoder context once.
    ctx = build_oms_context(par)

    # Eb/N0 sweep.
    n_steps = int(round((args.ebno_max - args.ebno_min) / args.ebno_step)) + 1
    points = []
    for k in range(n_steps):
        ebno = args.ebno_min + k * args.ebno_step
        print(f"--- Eb/N0 = {ebno:.2f} dB ---")
        res = run_point(
            ebno_db=ebno, params=params, par=par, ctx=ctx,
            max_iter=args.max_iter, min_frame_errors=args.thres,
            max_frames=args.max_frames, seed=args.seed + k,
        )
        print(
            f"  >> Eb/N0={res['ebno_db']:.2f}dB  frames={res['frames']}  "
            f"FE={res['frame_errors']}  FER={res['fer']:.3e}  "
            f"BER={res['ber']:.3e}  avg_it={res['avg_iter']:.2f}  "
            f"elapsed={res['elapsed_s']:.1f}s"
        )
        points.append(res)

    # Persist results.
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"sim_A{params.A}_R{params.code_rate:.2f}_OMS_iter{args.max_iter}.txt",
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# 3GPP NR LDPC, BPSK, AWGN, Offset Min-Sum (offset=0.25)\n")
        fh.write(
            f"# A={params.A}  rate={params.code_rate}  bgn={params.bgn}  "
            f"setIdx={params.set_idx}  Zc={params.zc}  outlen={params.outlen}\n"
        )
        fh.write("# EbN0_dB  frames  frame_errors  bit_errors  FER  BER  avg_iter  elapsed_s\n")
        for r in points:
            fh.write(
                f"{r['ebno_db']:.3f}  {r['frames']}  {r['frame_errors']}  "
                f"{r['bit_errors']}  {r['fer']:.6e}  {r['ber']:.6e}  "
                f"{r['avg_iter']:.3f}  {r['elapsed_s']:.2f}\n"
            )
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
