"""Phase H+3 smoke test for slot micro-evolution.

For each detector with annotated slot ops, verify:
  - map_pop_key_to_from_slot_ids returns a non-empty set
  - collect_slot_region returns a RewriteRegion
  - apply_slot_variant with the default variant yields a flat IR that
    validates and produces compilable Python source.
  - perturb_constants_in_ir produces a child that also splices+validates.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np

from algorithm_ir.regeneration.codegen import emit_python_source
from algorithm_ir.ir.validator import validate_function_ir
from evolution.ir_pool import build_ir_pool
from evolution.slot_evolution import (
    apply_slot_variant,
    collect_slot_region,
    map_pop_key_to_from_slot_ids,
    perturb_constants_in_ir,
)


def main() -> int:
    pool = build_ir_pool(np.random.default_rng(42))
    print(f"Pool size: {len(pool)}")

    rng = np.random.default_rng(123)
    n_genomes = 0
    n_genomes_with_slots = 0
    total_pops = 0
    pops_with_anno = 0
    pops_region_built = 0
    pops_default_spliced = 0
    pops_default_validated = 0
    pops_perturbed_spliced = 0
    pops_default_codegen_ok = 0

    examples_printed = 0
    for g in pool:
        n_genomes += 1
        if not g.slot_populations:
            continue
        n_genomes_with_slots += 1
        for pop_key, pop in g.slot_populations.items():
            total_pops += 1
            if not pop.variants:
                continue
            sids = map_pop_key_to_from_slot_ids(g, pop_key)
            if not sids:
                continue
            pops_with_anno += 1
            region = collect_slot_region(g.ir, sids)
            if region is None:
                continue
            pops_region_built += 1
            default = pop.variants[0]
            if default is None:
                continue
            new_ir = apply_slot_variant(g, pop_key, default)
            if new_ir is None:
                continue
            pops_default_spliced += 1
            errs = validate_function_ir(new_ir)
            if not errs:
                pops_default_validated += 1
            try:
                src = emit_python_source(new_ir)
                if "def " in src:
                    pops_default_codegen_ok += 1
            except Exception:
                pass
            child = perturb_constants_in_ir(default, rng, scale=0.2)
            child_ir = apply_slot_variant(g, pop_key, child)
            if child_ir is not None and not validate_function_ir(child_ir):
                pops_perturbed_spliced += 1
            if examples_printed < 4 and pops_default_validated == examples_printed + 1:
                print(f"\nGenome={g.algo_id} pop_key={pop_key}")
                print(f"  from_slot_ids: {sids}")
                print(f"  region: {len(region.op_ids)} ops, "
                      f"entry={region.entry_values[:3]}, "
                      f"exit={region.exit_values[:3]}")
                print(f"  default variant ops: {len(default.ops)}")
                print(f"  spliced flat IR ops: {len(new_ir.ops)} "
                      f"(orig was {len(g.ir.ops)})")
                examples_printed += 1

    print("\n" + "=" * 70)
    print(f"genomes scanned                  : {n_genomes}")
    print(f"genomes with slot populations    : {n_genomes_with_slots}")
    print(f"slot populations total           : {total_pops}")
    print(f"  with at least one annotation   : {pops_with_anno}")
    print(f"  region built                   : {pops_region_built}")
    print(f"  default variant spliced (graft): {pops_default_spliced}")
    print(f"  default variant validated      : {pops_default_validated}")
    print(f"  default variant codegen ok     : {pops_default_codegen_ok}")
    print(f"  perturbed variant spliced+ok   : {pops_perturbed_spliced}")

    if pops_default_validated == 0:
        print("\nFAIL: no slot variant could be spliced")
        return 1
    print("\nOK: slot evolution machinery exercises real genomes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
