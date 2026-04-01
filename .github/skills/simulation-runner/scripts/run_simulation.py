"""
Run a simulation executable with parameter sweeps.
Captures output and organizes results by SNR point.

Usage:
    python run_simulation.py <executable> [options]

Options:
    --snr-min <float>     Minimum SNR in dB (default: 0)
    --snr-max <float>     Maximum SNR in dB (default: 10)
    --snr-step <float>    SNR step size in dB (default: 1.0)
    --frames <int>        Number of frames per SNR point (default: 10000)
    --output-dir <path>   Directory to store output files
    --args <string>       Additional arguments passed to the executable
    --timeout <int>       Timeout per SNR point in seconds (default: 3600)

Example:
    python run_simulation.py "sim.exe" --snr-min 0 --snr-max 8 --snr-step 0.5 --frames 50000 --output-dir "results/raw/"

Note:
    If the simulation handles its own SNR sweep internally (as in the mimo2D codebase),
    simply run without --snr-min/max/step and the script will execute the binary once
    and capture all output.
"""

import argparse
import os
import subprocess
import sys
import time


def run_single(executable: str, extra_args: list[str], output_dir: str,
               label: str, timeout: int) -> dict:
    """Run the simulation once, capture output."""
    os.makedirs(output_dir, exist_ok=True)
    stdout_path = os.path.join(output_dir, f"{label}_stdout.txt")
    stderr_path = os.path.join(output_dir, f"{label}_stderr.txt")

    cmd = [executable] + extra_args
    print(f"Running: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(executable) or '.',
        )
        elapsed = time.time() - start

        with open(stdout_path, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        if result.stderr:
            with open(stderr_path, 'w', encoding='utf-8') as f:
                f.write(result.stderr)

        return {
            'label': label,
            'returncode': result.returncode,
            'elapsed_s': elapsed,
            'stdout_file': stdout_path,
            'stderr_file': stderr_path if result.stderr else None,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.1f}s")
        return {
            'label': label,
            'returncode': -1,
            'elapsed_s': elapsed,
            'stdout_file': None,
            'stderr_file': None,
            'error': 'timeout',
        }


def main():
    parser = argparse.ArgumentParser(description="Run communication simulations.")
    parser.add_argument("executable", help="Path to the simulation executable")
    parser.add_argument("--snr-min", type=float, default=None)
    parser.add_argument("--snr-max", type=float, default=None)
    parser.add_argument("--snr-step", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--output-dir", default="results/raw/")
    parser.add_argument("--args", nargs='*', default=[], help="Extra args for the executable")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    if not os.path.isfile(args.executable):
        print(f"Error: Executable not found: {args.executable}", file=sys.stderr)
        sys.exit(1)

    results = []

    if args.snr_min is not None and args.snr_max is not None:
        # Parameter sweep mode: run once per SNR point
        snr = args.snr_min
        while snr <= args.snr_max + 1e-9:
            extra = list(args.args)
            # Common conventions: pass SNR as first arg or named
            extra.extend([str(snr)])
            if args.frames:
                extra.append(str(args.frames))
            label = f"snr_{snr:.1f}".replace('.', 'p')
            print(f"\n=== SNR = {snr:.1f} dB ===")
            r = run_single(args.executable, extra, args.output_dir, label, args.timeout)
            results.append(r)
            if r['returncode'] != 0:
                print(f"  Warning: non-zero return code {r['returncode']}")
            else:
                print(f"  Completed in {r['elapsed_s']:.1f}s")
            snr += args.snr_step
    else:
        # Single run mode: simulation handles its own parameter sweep
        print("\n=== Running simulation (internal parameter sweep) ===")
        r = run_single(args.executable, list(args.args), args.output_dir, "full_run", args.timeout)
        results.append(r)

    # Summary
    print("\n=== Summary ===")
    total_time = sum(r['elapsed_s'] for r in results)
    failed = sum(1 for r in results if r['returncode'] != 0)
    print(f"Total runs: {len(results)}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Output directory: {args.output_dir}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
