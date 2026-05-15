"""Selection operators for the GA.

* `tournament_select(pop, fitnesses, k, rng)` — k-way tournament,
  smaller fitness wins (we minimise FER / regret).

* `lexicase_select(pop, case_fitnesses, rng)` — optional lexicase
  selection over a list of per-case fitnesses (each row = one
  individual, each column = one fitness case).  Smaller is better.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def tournament_select(
    pop: Sequence,
    fitnesses: Sequence[float],
    k: int,
    rng: np.random.Generator,
):
    n = len(pop)
    if n == 0:
        raise ValueError("empty population")
    k = max(1, min(k, n))
    idx = rng.choice(n, size=k, replace=False)
    fits = np.asarray([fitnesses[int(i)] for i in idx], dtype=np.float64)
    # NaN/inf treated as worst.
    fits = np.where(np.isfinite(fits), fits, np.inf)
    winner = int(idx[int(np.argmin(fits))])
    return pop[winner]


def lexicase_select(
    pop: Sequence,
    case_fitnesses: np.ndarray,  # shape (n_individuals, n_cases)
    rng: np.random.Generator,
    epsilon: float = 0.0,
):
    n, c = case_fitnesses.shape
    if n == 0:
        raise ValueError("empty population")
    survivors = np.arange(n)
    case_order = rng.permutation(c)
    for ci in case_order:
        col = case_fitnesses[survivors, ci]
        col = np.where(np.isfinite(col), col, np.inf)
        best = col.min()
        keep = survivors[col <= best + epsilon]
        if len(keep) == 1:
            return pop[int(keep[0])]
        survivors = keep
    return pop[int(rng.choice(survivors))]


__all__ = ["tournament_select", "lexicase_select"]
