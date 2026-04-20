"""Diagnostic: verify that grafted children inherit donor slot populations."""
import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from copy import deepcopy
from evolution.ir_pool import build_ir_pool, find_algslot_ops
from evolution.pool_types import AlgorithmGenome, SlotPopulation
from evolution.algorithm_engine import AlgorithmEvolutionEngine
from evolution.pattern_matchers import ExpertPatternMatcher
from evolution.materialize import materialize

rng = np.random.default_rng(42)
pool = build_ir_pool(rng)
print(f"Pool: {len(pool)} genomes")
for g in pool:
    slots = list(g.slot_populations.keys())
    slot_info = {k: len(v.variants) for k, v in g.slot_populations.items()}
    print(f"  {g.algo_id:10s} slots={slot_info}")

# Build entries for pattern matcher
entries = [g.to_entry(None) for g in pool]

# Run ExpertPatternMatcher
pm = ExpertPatternMatcher(max_proposals_per_gen=10)
proposals = pm(entries, generation=1)
print(f"\nExpert proposals: {len(proposals)}")
for p in proposals:
    print(f"  {p.proposal_id}: host={p.host_algo_id} donor={p.donor_algo_id}")
    print(f"    region ops: {len(p.region.op_ids)}")
    print(f"    donor_ir ops: {len(p.donor_ir.ops) if p.donor_ir else 'None'}")
    donor_slots = [op for op in p.donor_ir.ops.values() if op.opcode == 'slot'] if p.donor_ir else []
    print(f"    donor slot ops: {[op.attrs.get('slot_id') for op in donor_slots]}")

# Execute each graft proposal manually to check slot inheritance
print("\n=== Testing graft execution ===")
for p in proposals:
    from algorithm_ir.grafting.graft_general import graft_general

    # Find host and donor genomes
    host_genome = next((g for g in pool if g.algo_id == p.host_algo_id), None)
    donor_genome = next((g for g in pool if g.algo_id == p.donor_algo_id), None)

    if host_genome is None or donor_genome is None:
        print(f"\n  SKIP {p.proposal_id}: missing genome")
        continue

    # Execute graft
    try:
        artifact = graft_general(host_genome.structural_ir, p)
    except Exception as e:
        print(f"\n  FAIL {p.proposal_id}: graft_general error: {e}")
        continue

    print(f"\n  Proposal: {p.proposal_id}")
    print(f"    new_slot_ids from artifact: {artifact.new_slot_ids}")
    print(f"    inlined ops: {len(artifact.inlined_op_ids)}")
    print(f"    replaced ops: {len(artifact.replaced_op_ids)}")

    # Check what slot ops exist in the grafted IR
    grafted_slot_ops = find_algslot_ops(artifact.ir)
    grafted_slot_ids = [op.attrs.get('slot_id') for op in grafted_slot_ops]
    print(f"    slot ops in grafted IR: {grafted_slot_ids}")

    # Build child with the fix
    child = AlgorithmGenome(
        algo_id=AlgorithmGenome._make_id(),
        structural_ir=artifact.ir,
        slot_populations=deepcopy(host_genome.slot_populations),
        constants=host_genome.constants.copy(),
        generation=1,
        parent_ids=[host_genome.algo_id, p.donor_algo_id or ""],
        graft_history=[],
        tags=set(host_genome.tags) | {"grafted"},
        metadata={},
    )

    # Use the engine's static helper to find donor slots
    for slot_id in artifact.new_slot_ids:
        if slot_id in child.slot_populations:
            continue
        donor_pop = AlgorithmEvolutionEngine._find_donor_slot_population(
            donor_genome, slot_id,
        )
        if donor_pop is not None:
            child.slot_populations[slot_id] = SlotPopulation(
                slot_id=slot_id,
                spec=donor_pop.spec,
                variants=[deepcopy(v) for v in donor_pop.variants],
                fitness=list(donor_pop.fitness),
                best_idx=donor_pop.best_idx,
                source_variants=list(donor_pop.source_variants) if donor_pop.source_variants else [],
            )
            print(f"    INHERITED slot '{slot_id}' from donor: {len(donor_pop.variants)} variants")
        else:
            print(f"    MISSING slot '{slot_id}' — no donor population found!")

    # Check all slot ops have populations
    all_ok = True
    for slot_op in grafted_slot_ops:
        sid = slot_op.attrs.get('slot_id')
        pop = None
        for k, v in child.slot_populations.items():
            if v.slot_id == sid or k == sid or k.endswith(f".{sid}") or sid.endswith(k.split('.')[-1]):
                pop = v
                break
        if pop is None or not pop.variants:
            print(f"    WARNING: slot op '{sid}' has no population!")
            all_ok = False

    # Try to materialize
    try:
        src = materialize(child)
        has_stub = "_slot_" in src and "*args" in src
        if has_stub:
            print(f"    PROBLEM: materialize produced stub pass-through!")
        else:
            print(f"    OK: materialize produced {len(src)} chars, no stubs")
    except Exception as e:
        print(f"    FAIL: materialize error: {e}")
        all_ok = False

    if all_ok:
        print(f"    RESULT: PASS")
    else:
        print(f"    RESULT: NEEDS ATTENTION")

print("\nDone.")
