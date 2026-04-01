"""
Generate LaTeX tables from CSV simulation data.

Usage:
    python generate_table.py <data_file> [options]

Options:
    --output <path>       Output .tex file path
    --caption <text>      Table caption
    --label <text>        Table label for \ref
    --format <spec>       Number format for values (default: auto-detect)
    --columns <list>      Comma-separated column names to include (default: all)
    --transpose           Swap rows and columns

Example:
    python generate_table.py "results.csv" --output "table.tex" --caption "BER performance comparison"
"""

import argparse
import os
import sys

import pandas as pd


def format_value(val) -> str:
    """Format a numeric value for LaTeX."""
    if pd.isna(val):
        return '--'
    if isinstance(val, str):
        return val
    try:
        fval = float(val)
    except (ValueError, TypeError):
        return str(val)

    if fval == 0:
        return '0'
    if abs(fval) < 0.01 or abs(fval) >= 1e6:
        # Scientific notation for very small/large numbers
        exp = f"{fval:.2e}"
        mantissa, exponent = exp.split('e')
        mantissa = float(mantissa)
        exponent = int(exponent)
        return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"
    if fval == int(fval) and abs(fval) < 1e9:
        return f"{int(fval)}"
    if abs(fval) < 1:
        return f"{fval:.4f}"
    return f"{fval:.2f}"


def generate_latex_table(df: pd.DataFrame, caption: str | None, label: str | None,
                          columns: list[str] | None) -> str:
    """Generate a complete LaTeX table environment."""
    if columns:
        df = df[[c for c in columns if c in df.columns]]

    n_cols = len(df.columns)
    col_spec = 'c' * n_cols
    col_spec = '|' + '|'.join(col_spec) + '|'

    lines = []
    lines.append('\\begin{table}[!t]')
    lines.append('\\centering')
    if caption:
        lines.append(f'\\caption{{{caption}}}')
    if label:
        lines.append(f'\\label{{{label}}}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\hline')

    # Header
    headers = [f'\\textbf{{{col}}}' for col in df.columns]
    lines.append(' & '.join(headers) + ' \\\\')
    lines.append('\\hline\\hline')

    # Data rows
    for _, row in df.iterrows():
        cells = [format_value(row[col]) for col in df.columns]
        lines.append(' & '.join(cells) + ' \\\\')
        lines.append('\\hline')

    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from CSV.")
    parser.add_argument("data_file", help="Input CSV file")
    parser.add_argument("--output", help="Output .tex file (default: stdout)")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--columns", default=None, help="Comma-separated column names")
    parser.add_argument("--transpose", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.data_file)

    if args.transpose:
        df = df.set_index(df.columns[0]).T.reset_index()
        df = df.rename(columns={'index': 'Metric'})

    columns = args.columns.split(',') if args.columns else None
    tex = generate_latex_table(df, args.caption, args.label, columns)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(tex)
        print(f"LaTeX table written to: {args.output}")
    else:
        print(tex)


if __name__ == "__main__":
    main()
