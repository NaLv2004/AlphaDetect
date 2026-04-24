"""Inspect the live behavior of typed-binding inside graft_general.

Builds the IR pool, picks several (host, donor) pairs from real
detector genomes, runs ``graft_general`` on each, and dumps the typed
binding diagnostics that the new bipartite layer recorded on each
GraftArtifact.

Verifies:
  1. The typed binder fires (artifact.typed_binding is not None).
  2. Donor argument types map to host candidates whose lattice types
     are compatible (subtype-or-equal in either direction, or share a
     non-trivial unify).
  3. Cross-IR grafts that previously bound by name-hint-then-positional
     now show *type-aware* assignments.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np

from algorithm_ir.grafting.graft_general import graft_general
from algorithm_ir.ir.type_lattice import is_subtype, unify
from evolution.ir_pool import build_ir_pool
from evolution.pattern_matchers import _build_region_from_ops, _fresh_id
from evolution.pool_types import GraftProposal


def lattice_compatible(donor_t: str, host_t: str) -> bool:
    if donor_t == "any" or host_t == "any":
        return True
    if donor_t == host_t:
        return True
    if is_subtype(host_t, donor_t) or is_subtype(donor_t, host_t):
        return True
    return unify(donor_t, host_t) != "any"


def attempt_graft(host, donor, rng):
    """Build a small random region in host and graft donor.ir into it."""
    block_ids = list(host.ir.blocks.keys())
    if not block_ids:
        return None
    bid = block_ids[0]
    block = host.ir.blocks[bid]
    non_term = [
        oid for oid in block.op_ids
        if host.ir.ops.get(oid) and host.ir.ops[oid].opcode not in
        ("return", "branch", "jump")
    ]
    if len(non_term) < 2:
        return None
    length = int(rng.integers(1, min(4, len(non_term)) + 1))
    start = int(rng.integers(0, max(1, len(non_term) - length + 1)))
    region_ops = non_term[start:start + length]
    region = _build_region_from_ops(host.ir, region_ops)
    proposal = GraftProposal(
        proposal_id=_fresh_id("inspect"),
        host_algo_id=host.algo_id,
        donor_algo_id=donor.algo_id,
        region=region,
        contract=None,
        donor_ir=donor.ir,
        dependency_overrides=[],
    )
    try:
        return graft_general(host.ir, proposal)
    except Exception as exc:
        return ("error", repr(exc))


def main() -> int:
    print("Building IR pool ...")
    pool = build_ir_pool()
    print(f"Pool size: {len(pool)}")
    rng = np.random.default_rng(42)

    n_attempts = 30
    n_typed = 0
    n_legacy = 0
    n_errors = 0
    sample_diags = []  # collect first few diagnostics for printing
    type_check_total = 0
    type_check_mismatch = 0

    for i in range(n_attempts):
        h, d = rng.choice(len(pool), size=2, replace=False)
        host, donor = pool[h], pool[d]
        result = attempt_graft(host, donor, rng)
        if result is None:
            continue
        if isinstance(result, tuple) and result[0] == "error":
            n_errors += 1
            continue
        artifact = result
        tb = artifact.typed_binding
        if tb is not None and tb.get("used"):
            n_typed += 1
            diag = tb.get("diagnostics") or []
            if len(sample_diags) < 5:
                sample_diags.append((host.algo_id, donor.algo_id, diag,
                                     tb.get("cost")))
            for donor_vid, host_vid, dt, ht, c in diag:
                if c >= 1e8:
                    continue
                type_check_total += 1
                if not lattice_compatible(dt, ht):
                    type_check_mismatch += 1
        else:
            n_legacy += 1

    print("\n" + "=" * 78)
    print(f"Attempts: {n_attempts}   typed-binder fired: {n_typed}   "
          f"legacy fallback: {n_legacy}   errors: {n_errors}")
    print("=" * 78)
    print(f"Lattice compatibility check: {type_check_total} bindings, "
          f"{type_check_mismatch} mismatches "
          f"({100 * type_check_mismatch / max(1, type_check_total):.1f}%)")

    print("\n--- Sample diagnostics (first 5 grafts that used typed binding) ---")
    for host_id, donor_id, diag, cost in sample_diags:
        print(f"\nhost {host_id[:18]:<18} <-- donor {donor_id[:18]:<18}  "
              f"cost={cost:.2f}")
        for donor_vid, host_vid, dt, ht, c in diag[:8]:
            ok = "OK" if lattice_compatible(dt, ht) else "MISMATCH"
            print(f"  {ok:<8} donor[{donor_vid:<10}: {dt:>10}]  "
                  f"-> host[{host_vid:<10}: {ht:>10}]  cost={c:6.2f}")

    if n_typed == 0:
        print("\nFAIL: typed binder never fired.")
        return 1
    print("\nOK: typed binder is active in the live graft pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
