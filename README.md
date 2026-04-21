<p align="center">
  <h1 align="center">AlphaDetect</h1>
  <p align="center"><b>Explainable MIMO Detection Algorithm Discovery via IR-Based Evolutionary Search</b></p>
  <p align="center">
    <a href="README.md">English</a> | <a href="README_CN.md">中文</a>
  </p>
</p>

---

## Overview

**AlphaDetect** is a research project for the **automated discovery of MIMO detection algorithms** via compiler intermediate representation (IR) and evolutionary search. The core idea is to compile existing detector implementations (LMMSE, ZF, OSIC, K-Best, BP, EP, AMP, etc.) into a structure-neutral SSA IR, then apply a **two-level evolutionary engine** to discover new detectors by cross-family structural grafting and per-slot micro-mutation.

> **Current status**: Active research. The `research/algorithm-IR/` folder contains the full framework — IR compiler, evolutionary engine with GNN-guided grafting, 8 baseline detectors, Monte Carlo MIMO evaluator, and 242 tests.

## Research Vision

The core research question is:

> *Can a machine systematically discover conceptual innovations in algorithm design, rather than merely optimizing within a fixed algorithm space?*

State-of-the-art MIMO detectors (ZF, MMSE, V-BLAST, sphere decoding, K-Best, BP, GTA, EP, AMP, etc.) were discovered by humans over decades. Each breakthrough required not just mathematical derivation, but the **invention of new concepts**: representing the detection problem as probabilistic inference on a graph, modeling the search space as a tree, or introducing cavity distributions.

AlphaDetect's Algorithm-IR framework targets **automated structural innovation** through:

1. **IR Compilation** — Python detector functions are compiled to SSA IR (`FunctionIR`), making their computational structure explicitly manipulable.
2. **Skeleton Grafting** — Cross-family structural transfer: e.g., injecting LMMSE pre-filtering into K-Best tree search, or replacing EP cavity updates with BP message sweeps — all performed at IR level via `graft_general()`.
3. **Two-Level Evolution** — Macro-level evolves skeleton structures (via grafting); micro-level evolves per-slot implementations (via IR mutation/crossover).
4. **GNN-Guided Proposals** — A Graph Attention Network learns online (via REINFORCE) which host-donor pairs produce viable detectors, replacing random grafting with learned structural intuition.
5. **Materialization** — Any evolved `AlgorithmGenome` can be materialized back to executable Python source code.

For the full technical details, see [research/research-proposal/research_proposal.tex](research/research-proposal/research_proposal.tex).

## Project Structure

```
AlphaDetect/
├── .github/                        # AI agent system for vibe-coding research
│   ├── agents/                     # 8 specialized agent definitions
│   │   ├── orchestrator.agent.md   # Central research coordinator
│   │   ├── code-generation.agent.md
│   │   ├── experiment.agent.md
│   │   ├── ideator.agent.md
│   │   ├── literature-search.agent.md
│   │   ├── math-deduction.agent.md
│   │   ├── paper-writing.agent.md
│   │   └── review.agent.md
│   ├── instructions/               # Coding conventions & memory protocols
│   └── skills/                     # Domain-specific skill modules
├── AGENTS.md                       # Agent system overview & conventions
├── research/
│   ├── memory/                     # Persistent research memory
│   │   ├── state.json              # Current research threads & phases
│   │   ├── experiment-log.md       # Chronological experiment records
│   │   ├── idea-bank.md            # Research ideas with status tracking
│   │   ├── decision-history.md     # Major decisions with rationale
│   │   └── experience-base.md      # Lessons learned & patterns
│   ├── research-proposal/          # Core research proposal (LaTeX)
│   │   ├── research_proposal.tex   # Full AlphaDetect vision (English)
│   │   └── research_proposal_cn.tex  # Chinese version
│   └── algorithm-IR/              # ★ Main research implementation
│       ├── algorithm_ir/           #   IR compiler, grafting, runtime
│       ├── evolution/              #   Two-level evolutionary engine
│       ├── tests/                  #   242 tests (unit + integration)
│       ├── e2e_experiment.py       #   End-to-end evolution experiment
│       ├── train_gnn.py            #   GNN pattern matcher trainer
│       ├── baseline_eval.py        #   Baseline detector evaluation
│       └── osic_sweep.py           #   SNR sweep for OSIC baseline
└── code_for_reference/             # External reference code (AutoML-Zero etc.)
```

### AI Agent System (`.github/`)

This project uses a **vibe-coding research** workflow powered by VS Code Copilot custom agents. An **Orchestrator** agent coordinates 7 specialized sub-agents:

| Agent | Role |
|-------|------|
| **Orchestrator** | Central research director — loads research state, makes strategic decisions, delegates to sub-agents, maintains memory |
| **Ideator** | Generates novel research ideas, performs gap analysis, assesses novelty |
| **Literature Search** | Searches arXiv/IEEE/Scholar, downloads PDFs, builds literature reviews |
| **Code Generation** | Writes simulation code (Python/C++), implements algorithms |
| **Experiment** | Compiles, runs, and analyzes simulations; collects BER/SER results |
| **Math Deduction** | Performs rigorous step-by-step mathematical derivations |
| **Paper Writing** | Drafts IEEE-format LaTeX manuscripts, creates TikZ figures |
| **Review** | Simulates peer review, evaluates novelty and technical correctness |

## Algorithm-IR: IR-Based MIMO Detector Evolution

`research/algorithm-IR/` is the main implementation. It compiles Python MIMO detectors into SSA IR, evolves them structurally, and evaluates fitness via Monte Carlo simulation.

### Architecture Overview

```
Python Detectors (LMMSE, ZF, OSIC, K-Best, BP, EP, AMP, Stack)
        ↓  compile_source_to_ir()
   FunctionIR  (SSA — Values, Ops, Blocks)
        ↓
   AlgorithmGenome = structural_ir + slot_populations
        ↓
   Two-Level Evolution
   ├── Macro: graft_general()  ← GNN / Expert / Static pattern matchers
   └── Micro: mutate_ir() / crossover_ir()  ← per-slot IR mutation
        ↓
   MIMOFitnessEvaluator  (16×16 16-QAM Monte Carlo)
        ↓
   materialize()  →  Executable Python source
```

### 8 Baseline Detectors

| Algorithm | Tags | Evolvable Slots |
|-----------|------|-----------------|
| `lmmse` | `linear` | `slot_regularizer`, `slot_hard_decision` |
| `zf` | `linear` | `slot_hard_decision` |
| `osic` | `sic` | `slot_ordering`, `slot_sic_step` |
| `kbest` | `tree_search` | `slot_expand`, `slot_prune` |
| `stack` | `tree_search` | `slot_expand`, `slot_prune` |
| `bp` | `message_passing` | `slot_bp_sweep`, `slot_bp_final` |
| `ep` | `inference` | `slot_ep_site`, `slot_ep_final` |
| `amp` | `inference` | `slot_amp_denoise`, `slot_amp_final` |

### Quick Start

```bash
conda activate AutoGenOld
cd research/algorithm-IR

# 1. Evaluate all 8 baseline detectors (16×16 16-QAM, SNR=24dB)
python baseline_eval.py

# 2. OSIC SNR sweep (18–28 dB, 2000 trials)
python osic_sweep.py

# 3. End-to-end evolution experiment (50 generations, Expert+Static+GNN matcher)
python e2e_experiment.py

# 4. GNN-guided trainer (large-scale, 500 graft proposals/generation)
python train_gnn.py [--gens 200] [--snr-start 20] [--snr-target 24] \
    [--proposals 500] [--pool-size 141] [--n-trials 5] [--timeout 1.5] \
    [--warmstart-gens 1] [--warmstart-eval-workers 8] [--seed 42] \
    [--resume path/to/checkpoint.pt]

# 5. Run all tests
python -m pytest tests/ -v   # 242 tests, ~30s
```

### `train_gnn.py` Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gens` | 200 | Total generations |
| `--snr-start` | 20.0 | Initial training SNR (dB) |
| `--snr-target` | 24.0 | Final target SNR (dB) |
| `--proposals` | 500 | GNN graft proposals per generation |
| `--pool-size` | 141 | Population size (91 original + 50 survivors) |
| `--n-trials` | 5 | Evaluation trials per genome (320 bits/trial) |
| `--timeout` | 1.5 | Per-genome evaluation timeout (seconds) |
| `--warmstart-gens` | 1 | Warm-start generations (exhaustive pair sweep) |
| `--warmstart-eval-workers` | 8 | Parallel workers for warm-start evaluation |
| `--ckpt-interval` | 10 | GNN checkpoint every N generations |
| `--seed` | 42 | Random seed |
| `--resume` | None | Path to GNN checkpoint to resume from |

Output is written to `results/gnn_training/` (JSONL log + checkpoints).

### Prerequisites

- Python 3.12+ (`conda activate AutoGenOld`)
- `numpy`, `scipy`, `torch`, `torch_geometric`, `xdsl`
- `pytest` for running the test suite

## License

This project is for academic research purposes.

## Citation

If you reference this work, please cite the research proposal:

```
AlphaDetect: Explainable MIMO Detection Algorithm Discovery
via Neural-Symbolic Automated Reasoning. Research Proposal, April 2026.
```
