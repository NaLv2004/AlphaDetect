---
description: "Use when writing or modifying LaTeX manuscripts for IEEE papers in communications/signal processing. Covers notation consistency, figure placement, reference formatting, and academic writing style."
applyTo: "**/*.tex"
---

# LaTeX Academic Writing Conventions

## Format

- Use `\documentclass[journal]{IEEEtran}` for transactions
- Use `\documentclass[conference]{IEEEtran}` for conference papers
- Packages: `amsmath, amssymb, graphicx, cite, tikz, pgfplots, algorithmic`

## Notation Consistency

Maintain these conventions (consistent with `manuscript/main.tex`):

| Symbol | Meaning | LaTeX |
|--------|---------|-------|
| Bold uppercase | Matrix | `\mathbf{H}` |
| Bold lowercase | Vector | `\mathbf{x}` |
| Italic | Scalar | `x, N, K` |
| Calligraphic | Set | `\mathcal{S}` |
| Hat | Estimate | `\hat{x}` |
| Tilde | Transformed | `\tilde{x}` |

## Cross-References

Always use `\label` and `\ref`:
```latex
\section{System Model}\label{sec:sysmodel}
\begin{equation}\label{eq:received_signal}
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
\end{equation}
As shown in \eqref{eq:received_signal}, ...
```

## Figures

- Placement: `\begin{figure}[!t]` (top of column)
- Width: `\columnwidth` for single column, `\textwidth` for double
- Use pgfplots with `.dat` data files for BER/FER curves
- Always include `\caption{}` and `\label{fig:...}`

## Writing Style

- Flowing paragraphs, not bullet points in the body
- Define notation before using it
- One contribution per paragraph in the introduction
- Begin sections with context before diving into equations
- "It can be shown that" → show it or cite it
