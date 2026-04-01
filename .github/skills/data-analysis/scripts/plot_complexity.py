"""
Generate complexity comparison plots (bar charts or grouped bars).
Compares computational complexity or runtime across different methods.

Usage:
    python plot_complexity.py <data_file> [options]

Options:
    --output <path>       Output file path (without extension)
    --mode <type>         Chart type: bar, grouped, stacked (default: grouped)
    --ylabel <text>       Y-axis label (default: "Runtime (s)")
    --title <text>        Plot title
    --width <inches>      Figure width (default: 3.5)
    --log-scale           Use logarithmic y-axis

Input CSV format:
    Method,Metric1,Metric2,...
    Proposed,0.5,100,...
    Baseline,1.2,300,...

Example:
    python plot_complexity.py "complexity.csv" --output "complexity_chart" --ylabel "Operations per frame"
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IEEE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}

BAR_COLORS = ['#2171b5', '#cb181d', '#238b45', '#6a3d9a', '#ff7f00', '#a65628']
HATCH_PATTERNS = ['', '//', '\\\\', 'xx', '..', '++']


def plot_grouped_bars(df: pd.DataFrame, output: str, ylabel: str,
                      title: str | None, width: float, log_scale: bool):
    """Generate grouped bar chart."""
    plt.rcParams.update(IEEE_STYLE)

    methods = df.iloc[:, 0].values
    metrics = df.columns[1:]
    n_methods = len(methods)
    n_metrics = len(metrics)

    x = np.arange(n_methods)
    bar_width = 0.8 / n_metrics

    height = width * 0.7
    fig, ax = plt.subplots(figsize=(width, height))

    for i, metric in enumerate(metrics):
        values = df[metric].values.astype(float)
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                       label=metric,
                       color=BAR_COLORS[i % len(BAR_COLORS)],
                       hatch=HATCH_PATTERNS[i % len(HATCH_PATTERNS)],
                       edgecolor='black', linewidth=0.5)

    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0 if n_methods <= 5 else 30)
    if n_metrics > 1:
        ax.legend()
    if log_scale:
        ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.3)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(f"{output}.pdf")
    fig.savefig(f"{output}.png")
    plt.close(fig)
    print(f"Saved: {output}.pdf, {output}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot complexity comparison charts.")
    parser.add_argument("data_file", help="Input CSV file")
    parser.add_argument("--output", default="complexity_chart")
    parser.add_argument("--mode", choices=["bar", "grouped", "stacked"], default="grouped")
    parser.add_argument("--ylabel", default="Runtime (s)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--width", type=float, default=3.5)
    parser.add_argument("--log-scale", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.data_file)
    plot_grouped_bars(df, args.output, args.ylabel, args.title, args.width, args.log_scale)


if __name__ == "__main__":
    main()
