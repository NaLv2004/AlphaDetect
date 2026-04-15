# MIMO-Push-GP: Evolving MIMO Detection Algorithms via Genetic Programming

## Overview

This project uses **genetic programming (GP)** to automatically discover MIMO detection algorithms. The core idea is to evolve small programs that control a **Belief-Propagation (BP) augmented stack decoder** for MIMO signal detection. The system combines:

- A **Push-style virtual machine** with typed stacks (float, int, bool, vector, matrix, node, graph)
- A **4-program genome** that defines how messages pass through a tree-structured search
- **QR decomposition** to convert the MIMO detection problem into a tree search
- **C++ acceleration** for fast fitness evaluation during evolution

The evolved programs define *how* the decoder computes messages, scores nodes, and decides when to stop BP iterations — replacing hand-designed heuristics with automatically discovered ones.

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Evolution Engine                     │
│   (population, tournament select, mutation, xover)    │
├──────────────────┬───────────────────────────────────┤
│   Evaluator      │   Structured BP Stack Decoder      │
│   (Python/C++)   │   (bp_decoder_v2.py)               │
├──────────────────┴───────────────────────────────────┤
│                  Push VM (vm.py)                       │
│   float_stack │ int_stack │ bool_stack │ vec_stack     │
│   mat_stack   │ node_stack│ graph_stack│ exec_stack    │
├──────────────────────────────────────────────────────┤
│              MIMO Channel Model                       │
│   H·x + n = y  →  QR decomposition  →  Tree Search   │
└──────────────────────────────────────────────────────┘
```

## The 4-Program Genome

Each individual in the population is a **Genome** with four evolved programs and four evolved constants:

| Program | Signature | Purpose |
|---------|-----------|---------|
| **F_down** | `f(M_parent_down, C_i) → M_down` | DOWN sweep: propagate parent's message to child, incorporating local distance |
| **F_up** | `f({C_j, M_j_up}) → M_up` | UP sweep: aggregate children's messages and distances to compute parent message |
| **F_belief** | `f(D_i, M_down, M_up) → score` | Score each frontier node using cumulative distance and BP messages |
| **H_halt** | `f(old_M_up, new_M_up) → bool` | Decide whether to continue BP iterations (convergence check) |

**Evolved Constants**: Four log-domain constants (EC0–EC3) available to all programs via `Float.EvoConst0`–`Float.EvoConst3`.

## BP Message Passing Flow

After each node expansion in the stack decoder:

```
1. Expand node → create M children with local_dist, cum_dist
2. UP sweep  (leaves → root): parent.m_up = F_up(children's m_up, local_dist)
3. DOWN sweep (root → leaves): child.m_down = F_down(parent.m_down, child.local_dist)
4. Score ALL frontier: node.score = F_belief(cum_dist, m_down, m_up)
5. Check halt: if H_halt(old_root_m_up, new_root_m_up) → repeat BP (goto 2)
6. Pop best-scored frontier node → expand → goto 1
```

Key design: **all** frontier nodes are re-scored after every BP cycle, and previously expanded nodes can be re-explored through the priority queue's `queue_version` mechanism.

## File Structure

```
code/
├── bp_main_v2.py          # Main evolution runner, formula conversion, evaluation
├── bp_decoder_v2.py       # Structured BP stack decoder (detection engine)
├── vm.py                  # Push-style virtual machine with typed stacks
├── stacks.py              # TreeNode, SearchTreeGraph, TypedStack
├── evolution.py           # Mutation, crossover, selection, fitness
├── stack_decoder.py       # Baseline detectors (LMMSE, K-Best)
├── cpp_bridge.py          # C++ DLL bridge for fast genome evaluation
├── eval_genomes.py        # Standalone genome evaluation script
├── cpp/                   # C++ source for accelerated evaluator
│   ├── bp_evaluator.cpp   # C++ BP evaluator implementation
│   └── ...
├── seed_genomes/          # JSON files for warm-starting evolution
├── run_test6.bat          # Example batch file for running experiments
└── test_*.py              # Various test and debugging scripts

logs/                      # Evolution log files
results/                   # Evaluation results (JSON)
papers/                    # Related papers
bp_stack_design.tex        # LaTeX design document
```

## Key Components

### Push VM (`vm.py`)

A typed-stack virtual machine with ~80 instructions across categories:

- **Arithmetic**: `Float.Add`, `Float.Mul`, `Float.Exp`, `Float.Log`, `Float.Max`, `Float.Min`, ...
- **Comparison**: `Float.GT`, `Float.LT`, `Int.LT`, ...
- **Stack ops**: `Float.Swap`, `Float.Dup`, `Node.Pop`, ...
- **Environment access**: `Mat.PeekAt` (read R matrix), `Vec.PeekAt` (read y_tilde), `Float.GetMMSELB` (MMSE lower bound), `Float.GetNoiseVar`, ...
- **BP messages**: `Node.GetMUp`, `Node.GetMDown`, `Node.SetMDown`, `Node.SetScore`, ...
- **Control flow**: `Exec.If`, `Exec.ForEachSymbol`, `Exec.MinOverSymbols`

### BP Decoder (`bp_decoder_v2.py`)

Implements the structured stack decoder with BP augmentation:

- QR decomposition of channel matrix H
- Tree search with priority queue (min-score ordering)
- Full UP/DOWN BP sweeps after each node expansion
- Re-exploration of previously expanded nodes via `queue_version`
- Configurable node budget, flops budget, and step limit

### Evolution Engine (`evolution.py`, `bp_main_v2.py`)

- **Population**: Tournament selection with elitism
- **Mutation**: Instruction swap, insertion, deletion, block mutation, constant perturbation
  - Escalating mutations: later attempts try more aggressive changes
  - Fallback: generates fresh random genome instead of returning unmutated copy
- **Crossover**: Single-point crossover between programs
- **Hall of Fame**: Stores up to 15 diverse genomes (signature-based deduplication)
- **Stagnation detection**: Hard restart when no improvement for N generations
- **BP dependency filter**: Rejects mutations that break BP message dependencies in F_up/F_belief

### Formula Tracing

Two methods for converting programs to human-readable formulas:

1. **Symbolic tracer** (`program_to_formula`): Static symbolic execution on the stack
2. **Trace-based tracer** (`program_to_formula_trace`): Runs the actual VM with perturbation analysis to determine real dependencies and values

## Running Experiments

### Prerequisites

- Python 3.8+ with numpy, scipy, matplotlib, tqdm
- Conda environment: `AutoGenOld`
- (Optional) C++ compiler for acceleration

### Quick Test

```bash
conda activate AutoGenOld
python -u bp_main_v2.py --generations 5 --population 20 --train-samples 20 \
    --train-max-nodes 500 --train-flops-max 500000 --step-max 1000 \
    --train-snrs "24" --eval-snrs "20,24,28" --eval-trials 50 \
    --train-nt 4 --train-nr 4 --log-suffix quick_test --mod-order 4
```

### Full Experiment (16×16 MIMO, 16-QAM)

```bash
python -u bp_main_v2.py --generations 60 --population 100 --train-samples 80 \
    --train-max-nodes 1000 --train-flops-max 1200000 --step-max 1000 \
    --train-snrs "24,28" --eval-snrs "18,20,22,24,28" --eval-trials 500 \
    --eval-max-nodes 2000 --eval-flops-max 5000000 --eval-step-max 5000 \
    --train-nt 16 --train-nr 16 --log-suffix test_run --use-cpp
```

### Evaluate Specific Genomes

```bash
python -u eval_genomes.py
```

Edit `eval_genomes.py` to define the genomes you want to evaluate.

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--generations` | Number of evolution generations | 30 |
| `--population` | Population size | 80 |
| `--train-samples` | Training samples per fitness evaluation | 50 |
| `--train-max-nodes` | Max tree nodes during training | 500 |
| `--train-flops-max` | Max flops budget during training | 500000 |
| `--step-max` | Max VM execution steps per program | 2000 |
| `--train-snrs` | Comma-separated training SNR values | "20,24,28" |
| `--eval-snrs` | Comma-separated evaluation SNR values | "16,18,20,22,24,26,28" |
| `--eval-trials` | Number of trials per SNR for evaluation | 200 |
| `--use-cpp` | Enable C++ accelerated evaluation | False |
| `--continuous` | Run in continuous mode (batch + periodic eval) | False |
| `--batch-gens` | Generations per batch in continuous mode | 5 |
| `--seed-genome-json` | JSON file for warm-starting from a known genome | None |

## Log Output

Each generation logs:

```
Gen 0042 [12.3s] | BER=0.05234 ratio=0.197 FLOPs=45230 len=14 uniq=78/100 faults=0.02 bp_up=67.3
```

- **BER**: Best individual's bit error rate on training data
- **ratio**: BER ratio vs LMMSE baseline
- **FLOPs**: Average flops used by best individual
- **uniq**: Unique individuals in population (diversity metric)
- **faults**: Fraction of population causing VM faults
- **bp_up**: Percentage of fitness evaluations where F_up modifies m_up (BP activity)

Top-10 individuals are logged every generation with their formulas and evolved constants.

Periodic full evaluations (every `batch_gens` in continuous mode) compare the best evolved algorithm against LMMSE, K-Best-16, and K-Best-32 baselines with ≥500 bit errors per SNR point for statistical significance.

## Baseline Comparisons

The system evaluates against three baselines:

| Baseline | Description |
|----------|-------------|
| **LMMSE** | Linear Minimum Mean Square Error detector |
| **K-Best 16** | K-Best tree search with K=16 |
| **K-Best 32** | K-Best tree search with K=32 |

Baseline evaluation is performed at experiment start and included in all subsequent comparison reports.

## Changes Log (Latest Session)

### Bug Fixes
- **`program_to_formula`**: Added handlers for `Mat.PeekAt`, `Vec.PeekAt`, `PeekAtIm`, `SecondPeekAt`; unknown ops are now silently skipped instead of showing `[?]`
- **`mutate_genome`**: Fixed fallback that returned unmutated copies (major stagnation source). Now uses escalating mutation counts and generates fresh random genomes on fallback
- **Hall of Fame**: Expanded from 5 to 15 entries with oneliner-signature-based deduplication to maintain diversity

### New Features
- **`program_to_formula_trace`**: Trace-based formula conversion using actual VM execution with perturbation analysis
- **`baseline_evaluation`**: Standalone baseline eval with min-bit-errors stopping criterion
- **`full_evaluation`**: Rewritten with `min_bit_errors=500` parameter for statistical significance
- **Baseline eval at experiment start**: Runs LMMSE, K-Best 16, K-Best 32 before evolution begins
- **Periodic full-SNR evaluation**: Every `batch_gens` generations in continuous mode
- **Diversity logging**: `[Diversity] unique=X/Y stagnant_gens=Z` written to log file
- **Evolved constants in log**: Top-1 individual's EC0–EC3 values logged every generation
- **BP flow diagram**: Printed with formula output showing message passing structure
- **`eval_genomes.py`**: Standalone script for evaluating specific genomes against baselines

## Research Context

This project is part of the **AlphaDetect** research initiative, which aims to automatically discover novel MIMO detection algorithms using machine learning and evolutionary methods. The approach is inspired by AutoML-Zero but specialized for communications systems, where the search space is structured around the physics of wireless channel models and tree-search decoding.
