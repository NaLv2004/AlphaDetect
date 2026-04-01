---
name: data-analysis
description: "Analyze simulation data and generate publication-quality plots for communications research. Use when: plotting BER/FER waterfall curves, generating complexity comparison charts, creating LaTeX tables from data, formatting figures for IEEE papers, analyzing Monte Carlo simulation results, comparing algorithm performance."
argument-hint: "Describe what data to analyze and what plots/tables to generate"
---

# Data Analysis for Communications Research

## When to Use

- Generating BER/FER/throughput plots from simulation data
- Creating complexity comparison charts
- Converting data to LaTeX tables
- Formatting plots for IEEE paper submission
- Analyzing and comparing multiple simulation results

## Prerequisites

```
pip install numpy matplotlib pandas scipy
```

## Procedure

### 1. Load and Inspect Data

Read CSV or text data files from `research/<topic>/results/processed/`.
Verify data integrity: correct columns, reasonable value ranges, sufficient data points.

### 2. Generate Plots

#### BER/FER Waterfall Curves
```
python "<skill-path>/scripts/plot_ber_fer.py" "<data.csv>" --output "figures/ber_curve" [options]
```

#### Complexity Comparison
```
python "<skill-path>/scripts/plot_complexity.py" "<data.csv>" --output "figures/complexity" [options]
```

### 3. Generate LaTeX Tables
```
python "<skill-path>/scripts/generate_table.py" "<data.csv>" --output "tables/results.tex" [options]
```

### 4. IEEE Plot Formatting Guidelines

See [plot-styles.md](./references/plot-styles.md) for detailed IEEE formatting standards.

Key rules:
- Figure width: single column = 3.5 in, double column = 7 in
- Font size: 8-10 pt in figures
- Line styles: combine color + marker + line style for B&W compatibility
- Labels: include units, use LaTeX math mode for symbols
- Grid: major grid on, minor grid optional
- Legend: inside plot area, avoid overlapping data
