# IEEE Publication Plot Style Guide

## Figure Dimensions

| Type | Width | Height Ratio |
|------|-------|-------------|
| Single column | 3.5 in (88 mm) | 0.7–0.85 × width |
| Double column | 7.16 in (181 mm) | 0.5–0.7 × width |
| Quarter page | 3.5 × 2.5 in | fixed |

## Font Specifications

- **Family**: Serif (Times New Roman preferred, DejaVu Serif fallback)
- **Axis labels**: 9–10 pt
- **Tick labels**: 8 pt
- **Legend**: 8 pt
- **Title**: 10 pt (avoid titles in IEEE figures — use captions instead)

## Line Styles

Use combinations that remain distinguishable in both color and B&W print:

| Priority | Color | Marker | Line Style |
|----------|-------|--------|------------|
| 1 | Blue (#2171b5) | ○ (circle) | Solid (─) |
| 2 | Red (#cb181d) | □ (square) | Dashed (╌) |
| 3 | Green (#238b45) | △ (triangle up) | Dash-dot (╍) |
| 4 | Purple (#6a3d9a) | ◇ (diamond) | Dotted (…) |
| 5 | Black | ▼ (triangle down) | Solid |
| 6 | Orange | ◁ (triangle left) | Dashed |

- Line width: 1.5 pt
- Marker size: 5 pt
- Marker face: none (open markers) for clarity
- Marker edge width: 1.2 pt

## BER/FER Waterfall Plots

```python
ax.semilogy(snr, ber, 'b-o', markerfacecolor='none')
ax.set_xlabel('$E_b/N_0$ (dB)')
ax.set_ylabel('BER')
ax.set_ylim(1e-6, 1)
ax.grid(True, which='major', alpha=0.3)
ax.grid(True, which='minor', alpha=0.1)
```

Key rules:
- Always use semilogy (log y-axis) for error rates
- Y-axis range: typically 1e-6 to 1 (adjust to data)
- X-axis: SNR in dB, integer ticks
- Grid: major on, minor optional
- Legend: lower left for waterfall curves

## Complexity Comparison Plots

- Use grouped bar charts for discrete comparisons
- Add hatching patterns for B&W distinguishability
- Y-axis label should include units
- Use log scale for large range differences

## Saving

- Save as both **PDF** (vector, for LaTeX) and **PNG** (raster, for review)
- PDF: savefig with `bbox_inches='tight'`, `pad_inches=0.05`
- PNG: 300 DPI minimum
- For LaTeX inclusion: `\includegraphics[width=\columnwidth]{figure.pdf}`

## LaTeX Integration (pgfplots alternative)

When generating TikZ/pgfplots directly:
```latex
\begin{tikzpicture}
\begin{semilogyaxis}[
    width=\columnwidth,
    height=0.75\columnwidth,
    xlabel={$E_b/N_0$ (dB)},
    ylabel={BER},
    grid=major,
    legend pos=south west,
    legend style={font=\footnotesize},
    tick label style={font=\footnotesize},
    label style={font=\small},
]
% Data plotted here
\end{semilogyaxis}
\end{tikzpicture}
```
