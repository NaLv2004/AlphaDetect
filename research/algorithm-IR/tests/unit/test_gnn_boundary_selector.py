from __future__ import annotations

import numpy as np
import torch

from algorithm_ir.region.slicer import validate_boundary_region
from evolution.gnn_pattern_matcher import GNNPatternMatcher
from evolution.ir_pool import build_ir_pool


def _proposal_boundary_signature(proposal):
    return {
        "host_algo_id": proposal.host_algo_id,
        "donor_algo_id": proposal.donor_algo_id,
        "host_region_ops": tuple(proposal.region.op_ids),
        "host_entry_values": tuple(proposal.region.entry_values),
        "host_exit_values": tuple(proposal.region.exit_values),
        "host_selected_outputs": tuple(proposal.region.provenance.get("selected_output_values", [])),
        "host_effective_outputs": tuple(proposal.region.provenance.get("effective_output_values", [])),
        "host_selected_cuts": tuple(proposal.region.provenance.get("selected_cut_values", [])),
        "host_effective_cuts": tuple(proposal.region.provenance.get("effective_cut_values", [])),
        "donor_region_ops": tuple(proposal.donor_region.op_ids if proposal.donor_region else []),
        "donor_entry_values": tuple(proposal.donor_region.entry_values if proposal.donor_region else []),
        "donor_exit_values": tuple(proposal.donor_region.exit_values if proposal.donor_region else []),
        "donor_selected_outputs": tuple(
            proposal.donor_region.provenance.get("selected_output_values", [])
            if proposal.donor_region else []
        ),
        "donor_effective_outputs": tuple(
            proposal.donor_region.provenance.get("effective_output_values", [])
            if proposal.donor_region else []
        ),
        "donor_selected_cuts": tuple(
            proposal.donor_region.provenance.get("selected_cut_values", [])
            if proposal.donor_region else []
        ),
        "donor_effective_cuts": tuple(
            proposal.donor_region.provenance.get("effective_cut_values", [])
            if proposal.donor_region else []
        ),
    }


def test_bcir_experience_records_effective_boundaries_only_for_valid_proposals():
    genomes = build_ir_pool(np.random.default_rng(6))[:5]
    entries = [genome.to_entry(None) for genome in genomes]
    matcher = GNNPatternMatcher(
        max_proposals_per_gen=20,
        warmstart_generations=1,
        train_interval=10,
    )

    torch.manual_seed(19)
    np.random.seed(19)
    proposals = matcher(entries, generation=1)

    assert len(matcher._experience) == len(proposals)
    for exp in matcher._experience:
        assert exp["host_effective_outputs"]
        assert exp["donor_effective_outputs"]
        assert set(exp["host_effective_outputs"]).issubset(set(exp["host_selected_outputs"]))
        assert set(exp["donor_effective_outputs"]).issubset(set(exp["donor_selected_outputs"]))
        assert set(exp["host_effective_cuts"]).issubset(set(exp["host_selected_cuts"]))
        assert set(exp["donor_effective_cuts"]).issubset(set(exp["donor_selected_cuts"]))
        assert exp["host_region_validity"] == "ok"
        assert exp["donor_region_validity"] == "ok"


def test_bcir_batched_and_unbatched_match_boundary_actions_with_same_seed():
    genomes = build_ir_pool(np.random.default_rng(7))[:6]
    entries = [genome.to_entry(None) for genome in genomes]

    matcher_ref = GNNPatternMatcher(
        max_proposals_per_gen=30,
        warmstart_generations=1,
        train_interval=10,
        enable_batched_proposals=False,
        proposal_batch_size=1,
    )
    matcher_batch = GNNPatternMatcher(
        max_proposals_per_gen=30,
        warmstart_generations=1,
        train_interval=10,
        enable_batched_proposals=True,
        proposal_batch_size=16,
    )
    matcher_batch.encoder.load_state_dict(matcher_ref.encoder.state_dict())
    matcher_batch.scorer.load_state_dict(matcher_ref.scorer.state_dict())
    matcher_batch.boundary_region_policy.load_state_dict(
        matcher_ref.boundary_region_policy.state_dict()
    )

    torch.manual_seed(29)
    np.random.seed(29)
    ref_props = matcher_ref(entries, generation=1)
    torch.manual_seed(29)
    np.random.seed(29)
    batch_props = matcher_batch(entries, generation=1)

    assert [_proposal_boundary_signature(p) for p in ref_props] == [
        _proposal_boundary_signature(p) for p in batch_props
    ]
