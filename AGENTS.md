# Automated Communications Research System

This workspace hosts an AI-driven research automation system for the communications/signal processing domain. An **Orchestrator** agent coordinates specialized sub-agents to conduct end-to-end research: literature review, idea generation, code development, experimentation, paper writing, and peer review.

## Workspace Structure

```
research/
  memory/          # Persistent memory files (state, logs, experience)
  <topic-name>/    # Per-topic research folders
    papers/        # Downloaded PDFs and extracted text
    code/          # Simulation source code (C++/Python/MATLAB)
    results/       # Experiment outputs, data files, plots
    paper/         # LaTeX manuscript drafts
    references.bib # BibTeX for this topic
mimo2D/            # Existing C++ simulation codebase (reference)
manuscript/        # Existing paper (reference)
theme.txt          # Current research theme
```

## Research Directory Conventions

- Each research thread lives under `research/<topic-name>/` with a descriptive kebab-case name (e.g., `research/polar-mimo-idd/`).
- Orchestrator creates new topic folders when starting a new research thread.
- Multiple topics can be active simultaneously; Orchestrator tracks them in `research/memory/state.json`.
- Never modify `mimo2D/` or `manuscript/` directly — copy and adapt into `research/<topic>/code/` when needed.

## Memory System

All persistent memory resides in `research/memory/`. Agents MUST read relevant memory files before starting work and update them after completing tasks.

| File | Purpose | Updated By |
|------|---------|------------|
| `state.json` | Current research state, active threads, phase per topic | Orchestrator |
| `experiment-log.md` | Chronological experiment records with parameters and results | Experiment agent |
| `idea-bank.md` | Research ideas with status tracking | Ideator, Orchestrator |
| `decision-history.md` | Major decisions with rationale | Orchestrator |
| `experience-base.md` | Lessons learned, patterns, reusable insights | Orchestrator, all agents |
| `literature-notes.md` | Paper summaries, key findings, citation info | LiteratureSearch agent |

### Memory Update Protocol

1. **Read before act**: Always load the relevant memory files at session start.
2. **Append, don't overwrite**: Add new entries; only modify existing entries to update status.
3. **Timestamp all entries**: Use ISO 8601 format (`YYYY-MM-DD HH:MM`).
4. **Cross-reference**: Link experiment results to ideas, decisions to literature, etc.

## Cross-Agent Communication

Sub-agents receive structured prompts from Orchestrator and return structured reports. Standard output sections:

```
## Summary
{One-paragraph overview of what was accomplished}

## Details
{Full content: idea proposal / code description / experiment results / paper draft / review}

## Artifacts
{List of files created or modified, with paths}

## Recommendations
{Suggested next steps for Orchestrator}

## Memory Updates
{Entries to add to memory files — Orchestrator will commit these}
```

## Code Conventions

- **C++**: Follow patterns in `mimo2D/platform_1/` — Eigen3 for linear algebra, OpenMP for parallelism, structured parameter headers (see `setting.h`).
- **Python**: Use numpy/scipy/matplotlib. Virtual environment in `.venv/`.
- **MATLAB**: Standard function-file structure with clear parameter documentation.

## LaTeX Conventions

- IEEE two-column format for papers.
- Notation consistency: refer to `manuscript/main.tex` for established symbols ($\mathbf{H}$, $N_h$, $N_v$, etc.).
- Figures via TikZ/pgfplots with data files.
