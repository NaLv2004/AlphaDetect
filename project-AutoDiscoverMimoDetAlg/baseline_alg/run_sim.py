"""
Monte Carlo MIMO detection simulation.
Python equivalent of main.m — supports MMSE, AMP, EP, and K-Best detectors.

Usage examples:
  python run_sim.py --detector EP  --iter 7  --rx 32 --tx 16
  python run_sim.py --detector AMP --iter 10 --rx 32 --tx 16
  python run_sim.py --detector AMP --iter 10 --rx 32 --tx 16 --snr 12
  python run_sim.py --detector KBest --kbest-k 128 --rx 32 --tx 16
"""
import numpy as np
import argparse
import time

from sim_utils import (get_qam_params, gray_map_qam, gray_demap,
                       acwgn_eb_n0, generate_channel, complex_to_real)
from mmse_det import mmse_detect
from amp_det import amp_detect
from ep_det import ep_detect
from kbest_det import kbest_detect_wrapper


def run_simulation(args):
    np.random.seed(args.seed)

    sym, norm_f = get_qam_params(args.mod)
    slen = len(sym)
    info_len = args.mod * args.tx   # bits per frame

    if args.snr is not None:
        snr_range = [args.snr]
    else:
        snr_range = np.arange(args.snr_min, args.snr_max + args.snr_step / 2,
                              args.snr_step)

    print(f"Detector: {args.detector}, {args.rx}x{args.tx}, "
          f"{2**(args.mod//2)}-PAM / {2**args.mod}-QAM")
    print(f"SNR range: {[float(f'{s:.1f}') for s in snr_range]}")
    print(f"Max samples: {args.max_samples}, Max frame errors: {args.max_frames}")
    if args.detector in ('AMP', 'EP'):
        print(f"Iterations: {args.iter}, Delta: {args.delta}")
    elif args.detector == 'KBest':
        print(f"K: {args.kbest_k}")
    print("-" * 65)

    ber_results = []
    fer_results = []
    t_start = time.time()

    for snr_db in snr_range:
        error_bits = 0
        error_frames = 0
        loop = 0

        while error_frames < args.max_frames and loop < args.max_samples:
            loop += 1

            # --- Transmitter ---
            tx_bits = np.random.randint(0, 2, info_len)
            tx_symbol = gray_map_qam(tx_bits, args.mod, sym)
            tx_real = np.concatenate([np.real(tx_symbol), np.imag(tx_symbol)])

            # --- Channel ---
            H_complex = generate_channel(args.rx, args.tx)

            # --- Receiver ---
            rx_complex, Nv = acwgn_eb_n0(
                H_complex @ tx_symbol, 1, snr_db, args.tx, args.rx, args.mod, 1)

            H_real = complex_to_real(H_complex)
            y_real = np.concatenate([np.real(rx_complex), np.imag(rx_complex)])

            # --- Detection ---
            if args.detector == 'MMSE':
                symest = mmse_detect(args.tx, args.rx, slen, y_real,
                                     H_real, Nv, sym, args.delta, args.iter)
            elif args.detector == 'AMP':
                symest = amp_detect(args.tx, args.rx, slen, y_real,
                                    H_real, Nv, sym, args.delta, args.iter)
            elif args.detector == 'EP':
                symest = ep_detect(args.tx, args.rx, slen, y_real,
                                   H_real, Nv, sym, args.delta, args.iter)
            elif args.detector == 'KBest':
                symest = kbest_detect_wrapper(args.tx, args.rx, slen, y_real,
                                              H_real, Nv, sym, args.delta,
                                              args.iter, args.kbest_k)

            # --- Demapping and error counting ---
            rx_bits = gray_demap(symest, sym, args.tx)
            err = int(np.sum(rx_bits != tx_bits))
            error_bits += err
            error_frames += (err > 0)

            if loop % 500 == 0:
                cur_ber = error_bits / (loop * info_len)
                print(f"  SNR={snr_db:5.1f}dB  loop={loop:6d}  "
                      f"BER={cur_ber:.4e}  FER={error_frames/loop:.4e}  "
                      f"err_frames={error_frames}")

        ber = error_bits / (info_len * loop)
        fer = error_frames / loop
        ber_results.append(ber)
        fer_results.append(fer)
        print(f"SNR={snr_db:5.1f}dB: BER={ber:.4e}  FER={fer:.4e}  "
              f"samples={loop}")

    elapsed = time.time() - t_start
    print("-" * 65)
    print(f"Total time: {elapsed:.1f}s\n")
    print(f"{'SNR(dB)':>8} | {'BER':>12} | {'FER':>12}")
    print("-" * 40)
    for snr, ber, fer in zip(snr_range, ber_results, fer_results):
        print(f"{snr:8.1f} | {ber:12.4e} | {fer:12.4e}")

    return list(zip(snr_range, ber_results, fer_results))


def main():
    parser = argparse.ArgumentParser(description='MIMO detection baseline simulation')
    parser.add_argument('--detector', type=str, default='EP',
                        choices=['MMSE', 'AMP', 'EP', 'KBest'])
    parser.add_argument('--tx', type=int, default=16)
    parser.add_argument('--rx', type=int, default=32)
    parser.add_argument('--mod', type=int, default=4,
                        help='Bits per symbol (2=QPSK, 4=16QAM, 6=64QAM)')
    parser.add_argument('--snr', type=float, default=None,
                        help='Single SNR point (overrides range)')
    parser.add_argument('--snr-min', type=float, default=6)
    parser.add_argument('--snr-max', type=float, default=14)
    parser.add_argument('--snr-step', type=float, default=2)
    parser.add_argument('--max-samples', type=int, default=100000)
    parser.add_argument('--max-frames', type=int, default=50)
    parser.add_argument('--iter', type=int, default=7)
    parser.add_argument('--delta', type=float, default=0.7)
    parser.add_argument('--kbest-k', type=int, default=128)
    parser.add_argument('--seed', type=int, default=114514)
    args = parser.parse_args()
    run_simulation(args)


if __name__ == '__main__':
    main()
