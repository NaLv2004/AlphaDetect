"""
Parse simulation output files and collect results into structured CSV format.
Supports common output patterns from communications simulation code.

Usage:
    python collect_results.py <input_dir_or_file> [options]

Options:
    --output <path>       Output CSV file path (default: results.csv)
    --pattern <regex>     Custom regex pattern to match result lines
    --format <type>       Input format: auto, mimo2d, csv, whitespace (default: auto)

Common output formats detected:
    - SNR BER FER (whitespace-separated columns)
    - "SNR = X dB, BER = Y, FER = Z" style
    - mimo2D platform output format (from result.txt)

Example:
    python collect_results.py "results/raw/" --output "results/processed/results.csv"
    python collect_results.py "result.txt" --format mimo2d --output "results.csv"
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path


def parse_whitespace_table(text: str) -> list[dict]:
    """Parse whitespace-separated columns (SNR BER FER ...)."""
    results = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith(('#', '%', '//')):
            continue
        parts = line.split()
        try:
            numbers = [float(p) for p in parts]
        except ValueError:
            continue
        if len(numbers) >= 2:
            entry = {'SNR_dB': numbers[0], 'BER': numbers[1]}
            if len(numbers) >= 3:
                entry['FER'] = numbers[2]
            if len(numbers) >= 4:
                entry['Frames'] = int(numbers[3])
            if len(numbers) >= 5:
                entry['Errors'] = int(numbers[4])
            results.append(entry)
    return results


def parse_keyword_format(text: str) -> list[dict]:
    """Parse 'SNR = X, BER = Y' style output."""
    results = []
    snr_pattern = re.compile(
        r'SNR\s*[=:]\s*([-\d.]+).*?BER\s*[=:]\s*([\d.eE+-]+)(?:.*?FER\s*[=:]\s*([\d.eE+-]+))?',
        re.IGNORECASE,
    )
    for match in snr_pattern.finditer(text):
        entry = {
            'SNR_dB': float(match.group(1)),
            'BER': float(match.group(2)),
        }
        if match.group(3):
            entry['FER'] = float(match.group(3))
        results.append(entry)
    return results


def parse_mimo2d_format(text: str) -> list[dict]:
    """Parse the mimo2D platform output format from result.txt."""
    results = []
    # The mimo2D code typically outputs: SNR BER FER total_frames error_frames timing
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith(('#', '%', '//', 'SNR')):
            continue
        parts = line.split()
        try:
            if len(parts) >= 3:
                entry = {
                    'SNR_dB': float(parts[0]),
                    'BER': float(parts[1]),
                    'FER': float(parts[2]),
                }
                if len(parts) >= 4:
                    entry['Frames'] = int(float(parts[3]))
                if len(parts) >= 5:
                    entry['Time_s'] = float(parts[4])
                results.append(entry)
        except (ValueError, IndexError):
            continue
    return results


def auto_detect_and_parse(text: str) -> list[dict]:
    """Auto-detect format and parse."""
    # Try keyword format first
    results = parse_keyword_format(text)
    if results:
        return results
    # Try whitespace table
    results = parse_whitespace_table(text)
    if results:
        return results
    return []


def collect_from_dir(input_dir: str, fmt: str) -> list[dict]:
    """Collect results from all text files in a directory."""
    all_results = []
    txt_files = sorted(Path(input_dir).glob("*.txt"))
    if not txt_files:
        txt_files = sorted(Path(input_dir).glob("*_stdout.txt"))

    for fp in txt_files:
        text = fp.read_text(encoding='utf-8', errors='replace')
        if fmt == 'auto':
            results = auto_detect_and_parse(text)
        elif fmt == 'mimo2d':
            results = parse_mimo2d_format(text)
        elif fmt == 'whitespace':
            results = parse_whitespace_table(text)
        else:
            results = auto_detect_and_parse(text)

        for r in results:
            r['source_file'] = fp.name
        all_results.extend(results)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Collect simulation results into CSV.")
    parser.add_argument("input", help="Input directory or file")
    parser.add_argument("--output", default="results.csv", help="Output CSV path")
    parser.add_argument("--format", choices=["auto", "mimo2d", "csv", "whitespace"], default="auto")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        results = collect_from_dir(args.input, args.format)
    elif os.path.isfile(args.input):
        text = open(args.input, 'r', encoding='utf-8', errors='replace').read()
        parsers = {
            'auto': auto_detect_and_parse,
            'mimo2d': parse_mimo2d_format,
            'whitespace': parse_whitespace_table,
        }
        parse_fn = parsers.get(args.format, auto_detect_and_parse)
        results = parse_fn(text)
    else:
        print(f"Error: Path not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("Warning: No results parsed from input.", file=sys.stderr)
        sys.exit(0)

    # Write CSV
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fieldnames = list(results[0].keys())
    # Ensure standard columns come first
    priority = ['SNR_dB', 'BER', 'FER', 'Frames', 'Errors', 'Time_s']
    ordered = [f for f in priority if f in fieldnames] + [f for f in fieldnames if f not in priority]

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(results)

    print(f"Collected {len(results)} data points → {args.output}")

    # Print summary
    if results:
        snrs = [r['SNR_dB'] for r in results]
        print(f"SNR range: {min(snrs):.1f} to {max(snrs):.1f} dB")
        bers = [r['BER'] for r in results if r.get('BER', 0) > 0]
        if bers:
            print(f"BER range: {min(bers):.2e} to {max(bers):.2e}")


if __name__ == "__main__":
    main()
