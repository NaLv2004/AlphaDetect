# Optimal Correction Derivation for QR-Based Stack Decoder

## Summary

The optimal "correction" for a residual-scoring stack decoder (score = d_cum + correction) 
is the **negative of the A\* admissible future cost heuristic**.

## Key Result

For a node at layer k with decided symbols x[k:Nt-1]:

```
score(node) = d_cum(k) + h_{k-1}
h_{k-1} = min_{s ∈ Ω} |ŷ'_{k-1} - R_{k-1,k-1} * s|²
```

where the interference-cancelled observation is:
```
ŷ'_{k-1} = ŷ_{k-1} - Σ_{j=k}^{Nt-1} R_{k-1,j} * x_j
```

## What the GP needs to compute this

| To compute | R-matrix access needed |
|---|---|
| ŷ'_{k-1} (interference cancellation) | R[k-1, j] for j = k, ..., Nt-1 (row k-1, cols k to Nt-1) |
| h_{k-1} (nearest-point search) | R[k-1, k-1] (diagonal element of next layer) |
| Multi-layer h^(L) | R[i,j] for i=k-L,...,k-1 and j=i,...,Nt-1 |

## Simplest proxy for GP

Just use -|R[k-1,k-1]|² as correction:
```
Node.GetLayer ; Int.Dec ; Int.Dup ; Mat.PeekAt ; Float.Square ; Float.Neg
```
This is 6 instructions — achievable with Mat.PeekAt instruction.

## Key insight for evolution

The discovered program (v2, gen 10) computes:
```
correction = min(2*num_siblings, local_dist + Re(symbol))
```
= approx. local_dist + Re(symbol) (double-weighting + symbol preference)

The OPTIMAL correction is:
```
correction = -min_{s ∈ Ω} |ŷ'_{k-1} - R_{k-1,k-1}*s|²
```

Gap: the discovered correction doesn't access R matrix at all.
Room for improvement: R-matrix-aware corrections could match or beat K-Best-32.

## Dated: 2026-04-12
