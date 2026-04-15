<p align="center">
  <h1 align="center">AlphaDetect</h1>
  <p align="center"><b>Explainable MIMO Detection Algorithm Discovery via Neural-Symbolic Automated Reasoning</b></p>
  <p align="center">
    <a href="README.md">English</a> | <a href="README_CN.md">中文</a>
  </p>
</p>

---

## Overview

**AlphaDetect** is a research project aiming to build a unified framework for the **automated discovery of explainable MIMO detection algorithms**. Unlike neural architecture search or LLM-based code evolution, AlphaDetect grounds algorithm discovery in *formal reasoning* — every step in a discovered algorithm is traceable to a formal derivation, a type-correct composition of primitives, or a verified algebraic manipulation.

The ultimate vision comprises four tightly integrated components:

1. **Typed Domain-Specific Language (DSL)** — encodes signal-processing and probabilistic-inference primitives with precise mathematical types and algorithmic combinators.
2. **Problem Transformation Library** — recasts the invention of intermediate concepts (e.g., recognizing a MIMO model as a factor graph) as witness generation from typed transformations.
3. **Computer Algebra System (CAS)** — performs step-by-step symbolic derivation of algorithm update equations once a high-level structure is selected.
4. **LLM-augmented Structure Mapping Engine (SME)** — discovers cross-domain analogies (e.g., mapping between factor-graph inference and MIMO detection) and proposes new conceptual hypotheses.

These components jointly address the central challenge of algorithm discovery — the invention of novel *intermediate concepts* that cannot be reduced to search over a fixed concept space.

> **Current status**: This project is in active early-stage research. The `research/mimo-push-gp/` folder contains a proof-of-concept genetic programming system that evolves MIMO detection algorithms — a stepping stone towards the full AlphaDetect vision.

## Research Vision

The core research question is:

> *Can a machine systematically discover conceptual innovations in algorithm design, rather than merely optimizing within a fixed algorithm space?*

State-of-the-art MIMO detectors (ZF, MMSE, V-BLAST, sphere decoding, K-Best, BP, GTA, EP, AMP, etc.) were discovered by humans over decades. Each breakthrough required not just mathematical derivation, but the **invention of new concepts**: representing the detection problem as probabilistic inference on a graph, modeling the search space as a tree, approximating a loopy factor graph with a Chow-Liu spanning tree, or introducing cavity distributions.

AlphaDetect targets **explainable** algorithm discovery. The system operates by:

1. Encoding MIMO problems and source domains (e.g., factor-graph inference) as predicate structures via an LLM
2. Using Structure Mapping to discover cross-domain analogies and generate *candidate inferences*
3. Searching the space of DSL compositions and problem transformations guided by type compatibility and complexity constraints
4. Deriving exact update equations via a CAS once a high-level structure is found
5. Evaluating discovered algorithms for performance and complexity, feeding successful discoveries back into the system

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
│   │   ├── cpp-simulation.instructions.md
│   │   ├── latex-writing.instructions.md
│   │   └── research-memory.instructions.md
│   ├── prompts/                    # Reusable prompt templates
│   └── skills/                     # Domain-specific skill modules
│       ├── data-analysis/          # Plotting BER curves, generating tables
│       ├── literature-search/      # Searching arXiv/IEEE, downloading PDFs
│       └── simulation-runner/      # Compiling & running C++/Python sims
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
│   └── mimo-push-gp/              # PoC: GP-evolved MIMO detection
│       ├── code/                   # Implementation (Python)
│       ├── results/                # Experiment outputs (JSON)
│       ├── papers/                 # Related references
│       └── logs/                   # Evolution logs
└── code_for_reference/             # External reference code (AutoML-Zero etc.)
```

### AI Agent System (`.github/`)

This project uses a **vibe-coding research** workflow powered by a team of VS Code Copilot custom agents. An **Orchestrator** agent coordinates 7 specialized sub-agents to conduct end-to-end research:

| Agent | Role |
|-------|------|
| **Orchestrator** | Central research director — loads research state, makes strategic decisions, delegates to sub-agents, maintains memory |
| **Ideator** | Generates novel research ideas, performs gap analysis, assesses novelty |
| **Literature Search** | Searches arXiv/IEEE/Scholar, downloads PDFs, builds literature reviews |
| **Code Generation** | Writes simulation code (Python/C++), implements algorithms |
| **Experiment** | Compiles, runs, and analyzes simulations; collects BER/FER results |
| **Math Deduction** | Performs rigorous step-by-step mathematical derivations |
| **Paper Writing** | Drafts IEEE-format LaTeX manuscripts, creates TikZ figures |
| **Review** | Simulates peer review, evaluates novelty and technical correctness |

The agents share a persistent **memory system** (`research/memory/`) that tracks research state, experiment logs, ideas, decisions, and lessons learned across sessions.

## Proof-of-Concept: MIMO-Push-GP

The `research/mimo-push-gp/` folder contains a working proof-of-concept that uses **genetic programming** to automatically discover MIMO detection algorithms. While far from the full AlphaDetect vision (which targets formal reasoning-based discovery), this GP system demonstrates the feasibility of automatically finding competitive detection algorithms.

### How It Works

The system evolves small programs that control a **Belief-Propagation (BP) augmented stack decoder** for MIMO signal detection:

```
MIMO Channel (y = Hx + n)
        ↓
   QR Decomposition (H = QR)
        ↓
   Tree Search with Evolved BP
        ↓
   Detected Symbols (x̂)
```

**Key components:**

- **Push-style Virtual Machine** (`vm.py`): A typed-stack VM with ~80 instructions across float, int, bool, vector, matrix, node, and graph types
- **4-Program Genome**: Each individual has four evolved programs that define detection behavior:

  | Program | Purpose |
  |---------|---------|
  | `F_down` | DOWN sweep: propagate parent message to child |
  | `F_up` | UP sweep: aggregate children messages for parent |
  | `F_belief` | Score frontier nodes using distances and BP messages |
  | `H_halt` | Decide whether to continue BP iterations |

- **Structured BP Decoder** (`bp_decoder_v2.py`): A stack decoder augmented with full-tree BP message passing
- **Evolution Engine** (`evolution.py`): Tournament selection, mutation, crossover, with stagnation detection and hard restarts

### Running a Quick Experiment

```bash
conda activate AutoGenOld
cd research/mimo-push-gp/code

# Quick test (4×4 MIMO, QPSK, 5 generations)
python -u bp_main_v2.py --generations 5 --population 20 --train-samples 20 \
    --train-max-nodes 500 --train-flops-max 500000 --step-max 1000 \
    --train-snrs "24" --eval-snrs "20,24,28" --eval-trials 50 \
    --train-nt 4 --train-nr 4 --log-suffix quick_test --mod-order 4

# Full experiment (16×16 MIMO, 16-QAM, with C++ acceleration)
python -u bp_main_v2.py --generations 60 --population 100 --train-samples 80 \
    --train-max-nodes 1000 --train-flops-max 1200000 --step-max 1000 \
    --train-snrs "24,28" --eval-snrs "18,20,22,24,28" --eval-trials 500 \
    --eval-max-nodes 2000 --eval-flops-max 5000000 --eval-step-max 5000 \
    --train-nt 16 --train-nr 16 --log-suffix test_run --use-cpp
```

### Prerequisites

- Python 3.8+ with `numpy`, `scipy`, `matplotlib`, `tqdm`
- Conda environment: `AutoGenOld`
- (Optional) C++ compiler for accelerated fitness evaluation

## License

This project is for academic research purposes.

## Citation

If you reference this work, please cite the research proposal:

```
AlphaDetect: Explainable MIMO Detection Algorithm Discovery
via Neural-Symbolic Automated Reasoning. Research Proposal, April 2026.
```
