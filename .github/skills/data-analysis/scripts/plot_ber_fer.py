"""
Generate BER/FER waterfall curve plots for communications simulations.
Produces publication-quality figures suitable for IEEE papers.

Usage:
    python plot_ber_fer.py <data_file> [options]

Options:
    --output <path>       Output file path (without extension; saves .pdf and .png)
    --metric <type>       Metric to plot: BER, FER, or both (default: BER)
    --title <text>        Plot title (optional, omit for clean IEEE style)
    --xlabel <text>       X-axis label (default: "$E_b/N_0$ (dB)")
    --ylabel <text>       Y-axis label (auto-detected from metric)
    --legend-loc <pos>    Legend position (default: lower left)
    --column-snr <name>   Column name for SNR (default: SNR_dB)
    --column-ber <name>   Column name for BER (default: BER)
    --column-fer <name>   Column name for FER (default: FER)
    --multi <file1,file2> Multiple data files with labels: "file1:Label1,file2:Label2"
    --width <inches>      Figure width (default: 3.5 for single column)
    --ymin <float>        Minimum y-axis value (default: 1e-6)
    --ymax <float>        Maximum y-axis value (default: 1)

Example:
    python plot_ber_fer.py "results.csv" --output "ber_plot" --metric BER
    python plot_ber_fer.py --multi "proposed.csv:Proposed,baseline.csv:Baseline" --output "comparison"
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# IEEE-friendly plot style
IEEE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,  # Set True if LaTeX is installed
}

# Line style cycle for multiple curves
LINE_STYLES = [
    {'color': 'b', 'marker': 'o', 'linestyle': '-'},
    {'color': 'r', 'marker': 's', 'linestyle': '--'},
    {'color': 'g', 'marker': '^', 'linestyle': '-.'},
    {'color': 'm', 'marker': 'D', 'linestyle': ':'},
    {'color': 'k', 'marker': 'v', 'linestyle': '-'},
    {'color': 'c', 'marker': '<', 'linestyle': '--'},
    {'color': 'orange', 'marker': '>', 'linestyle': '-.'},
    {'color': 'brown', 'marker': 'p', 'linestyle': ':'},
]


def load_data(filepath: str, snr_col: str, metric_col: str) -> tuple:
    """Load data from CSV or whitespace-separated file."""
    try:
        df = pd.read_csv(filepath)
    except Exception:
        df = pd.read_csv(filepath, sep=r'\s+', header=None, comment='#')
        if snr_col == 'SNR_dB':
            df.columns = ['SNR_dB', 'BER'] + [f'col{i}' for i in range(2, len(df.columns))]
            if len(df.columns) > 2 and 'FER' not in df.columns:
                df = df.rename(columns={'col2': 'FER'})

    snr = df[snr_col].values
    metric = df[metric_col].values
    # Filter out zero or negative values (can't plot on log scale)
    mask = metric > 0
    return snr[mask], metric[mask]


def plot_waterfall(datasets: list[dict], metric: str, output: str,
                   title: str | None, xlabel: str, ylabel: str | None,
                   legend_loc: str, width: float, ymin: float, ymax: float):
    """Generate the waterfall plot."""
    plt.rcParams.update(IEEE_STYLE)

    height = width * 0.8
    fig, ax = plt.subplots(figsize=(width, height))

    for i, ds in enumerate(datasets):
        style = LINE_STYLES[i % len(LINE_STYLES)]
        ax.semilogy(ds['snr'], ds['data'], label=ds['label'],
                     markerfacecolor='none', markeredgewidth=1.2, **style)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)

    ax.set_ylim(ymin, ymax)
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.1)
    ax.legend(loc=legend_loc)

    # Save in both formats
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(f"{output}.pdf")
    fig.savefig(f"{output}.png")
    plt.close(fig)
    print(f"Saved: {output}.pdf, {output}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot BER/FER waterfall curves.")
    parser.add_argument("data_file", nargs='?', help="Input CSV data file")
    parser.add_argument("--output", default="ber_plot", help="Output path (no extension)")
    parser.add_argument("--metric", choices=["BER", "FER", "both"], default="BER")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xlabel", default="$E_b/N_0$ (dB)")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--legend-loc", default="lower left")
    parser.add_argument("--column-snr", default="SNR_dB")
    parser.add_argument("--column-ber", default="BER")
    parser.add_argument("--column-fer", default="FER")
    parser.add_argument("--multi", default=None, help="file1:Label1,file2:Label2")
    parser.add_argument("--width", type=float, default=3.5)
    parser.add_argument("--ymin", type=float, default=1e-6)
    parser.add_argument("--ymax", type=float, default=1.0)
    args = parser.parse_args()

    datasets = []

    if args.multi:
        # Multi-file mode
        for entry in args.multi.split(','):
            parts = entry.split(':')
            filepath = parts[0].strip()
            label = parts[1].strip() if len(parts) > 1 else os.path.basename(filepath)
            snr, data = load_data(filepath, args.column_snr,
                                   args.column_ber if args.metric != 'FER' else args.column_fer)
            datasets.append({'snr': snr, 'data': data, 'label': label})
    elif args.data_file:
        if args.metric == 'both':
            for m, col in [('BER', args.column_ber), ('FER', args.column_fer)]:
                try:
                    snr, data = load_data(args.data_file, args.column_snr, col)
                    datasets.append({'snr': snr, 'data': data, 'label': m})
                except (KeyError, IndexError):
                    print(f"Warning: Column {col} not found, skipping {m}.")
        else:
            col = args.column_ber if args.metric == 'BER' else args.column_fer
            snr, data = load_data(args.data_file, args.column_snr, col)
            datasets.append({'snr': snr, 'data': data, 'label': args.metric})
    else:
        parser.print_help()
        sys.exit(1)

    if not datasets:
        print("Error: No data to plot.", file=sys.stderr)
        sys.exit(1)

    ylabel = args.ylabel
    if ylabel is None:
        ylabel = args.metric if args.metric != 'both' else 'Error Rate'

    plot_waterfall(datasets, args.metric, args.output, args.title,
                   args.xlabel, ylabel, args.legend_loc, args.width,
                   args.ymin, args.ymax)


if __name__ == "__main__":
    main()
