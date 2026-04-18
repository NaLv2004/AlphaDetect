"""Debug slot_id matching between IR AlgSlot ops and SlotPopulations."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolution.ir_pool import build_ir_pool, find_algslot_ops

pool = build_ir_pool()
for g in pool:
    if g.algo_id in ('bp', 'amp', 'kbest', 'osic'):
        print(f"=== {g.algo_id} ===")
        slot_ops = find_algslot_ops(g.structural_ir)
        print("  AlgSlot ops:")
        for op in slot_ops:
            print(f"    op_id={op.id}, slot_id={op.attrs.get('slot_id')}")
        print("  SlotPopulation keys:")
        for key, pop in g.slot_populations.items():
            print(f"    key={key}, pop.slot_id={pop.slot_id}, has_source={pop.source_variants is not None and len(pop.source_variants) > 0}")
            if pop.source_variants and pop.source_variants[0]:
                print(f"      source_variant[0] starts with: {pop.source_variants[0][:60]}")
        print()
