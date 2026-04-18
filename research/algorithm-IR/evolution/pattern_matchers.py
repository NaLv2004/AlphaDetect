"""PatternMatcher implementations for structural grafting.

Three strategies:
  1. RandomGraftPatternMatcher  — random host/donor/region selection (testing)
  2. ExpertPatternMatcher       — hardcoded expert graft rules
  3. StaticStructurePatternMatcher — IR-structure-based matching
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from algorithm_ir.ir.model import FunctionIR, Op
from algorithm_ir.region.selector import RewriteRegion
from evolution.pool_types import (
    AlgorithmEntry,
    DependencyOverride,
    GraftProposal,
    PatternMatcherFn,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_id(prefix: str = "pm") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _find_contiguous_region(
    ir: FunctionIR,
    block_id: str,
    start_idx: int,
    length: int,
) -> list[str]:
    """Select a contiguous sub-sequence of ops in a block."""
    block = ir.blocks.get(block_id)
    if block is None:
        return []
    # Exclude terminators
    non_term = [
        oid for oid in block.op_ids
        if ir.ops.get(oid) and ir.ops[oid].opcode not in ("return", "branch", "jump")
    ]
    end = min(start_idx + length, len(non_term))
    return non_term[start_idx:end]


def _build_region_from_ops(
    ir: FunctionIR,
    op_ids: list[str],
) -> RewriteRegion:
    """Build a RewriteRegion from a list of op IDs."""
    block_ids = sorted({ir.ops[oid].block_id for oid in op_ids if oid in ir.ops})

    # Compute entry/exit values
    region_set = set(op_ids)
    region_defined: set[str] = set()
    region_used: set[str] = set()
    for oid in op_ids:
        op = ir.ops.get(oid)
        if op is None:
            continue
        region_defined.update(op.outputs)
        region_used.update(op.inputs)

    entry_values = sorted(v for v in region_used if v not in region_defined)
    exit_values: list[str] = []
    for vid in sorted(region_defined):
        val = ir.values.get(vid)
        if val is None:
            continue
        for use_op in val.use_ops:
            if use_op not in region_set:
                exit_values.append(vid)
                break

    return RewriteRegion(
        region_id=_fresh_id("region"),
        op_ids=op_ids,
        block_ids=block_ids,
        entry_values=entry_values,
        exit_values=exit_values,
        read_set=entry_values,
        write_set=exit_values,
        state_carriers=[],
        schedule_anchors={},
        allows_new_state=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. RandomGraftPatternMatcher
# ═══════════════════════════════════════════════════════════════════════════

class RandomGraftPatternMatcher:
    """Randomly selects host/donor pairs and regions for grafting.

    Primarily for stress-testing graft_general().
    """

    def __init__(
        self,
        proposals_per_gen: int = 2,
        min_region_size: int = 1,
        max_region_size: int = 5,
        seed: int = 42,
    ):
        self.proposals_per_gen = proposals_per_gen
        self.min_region_size = min_region_size
        self.max_region_size = max_region_size
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        entries: list[AlgorithmEntry],
        generation: int,
    ) -> list[GraftProposal]:
        if len(entries) < 2:
            return []

        proposals: list[GraftProposal] = []
        for _ in range(self.proposals_per_gen):
            try:
                proposal = self._make_random_proposal(entries)
                if proposal is not None:
                    proposals.append(proposal)
            except Exception as exc:
                logger.debug("Random proposal failed: %s", exc)

        return proposals

    def _make_random_proposal(
        self, entries: list[AlgorithmEntry],
    ) -> GraftProposal | None:
        indices = self.rng.choice(len(entries), size=2, replace=False)
        host_entry = entries[indices[0]]
        donor_entry = entries[indices[1]]

        # Pick a random block in the host
        block_ids = list(host_entry.ir.blocks.keys())
        if not block_ids:
            return None
        block_id = block_ids[self.rng.integers(0, len(block_ids))]
        block = host_entry.ir.blocks[block_id]

        # Pick a contiguous region
        non_term = [
            oid for oid in block.op_ids
            if host_entry.ir.ops.get(oid)
            and host_entry.ir.ops[oid].opcode not in ("return", "branch", "jump")
        ]
        if not non_term:
            return None

        length = int(self.rng.integers(self.min_region_size, self.max_region_size + 1))
        start = int(self.rng.integers(0, max(1, len(non_term) - length + 1)))
        region_ops = non_term[start:start + length]
        if not region_ops:
            return None

        region = _build_region_from_ops(host_entry.ir, region_ops)

        return GraftProposal(
            proposal_id=_fresh_id("rand_graft"),
            host_algo_id=host_entry.algo_id,
            region=region,
            contract=None,
            donor_algo_id=donor_entry.algo_id,
            donor_ir=donor_entry.ir,
            dependency_overrides=[],
            confidence=0.1,
            rationale=f"Random graft: {len(region_ops)} ops from {block_id}",
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. ExpertPatternMatcher
# ═══════════════════════════════════════════════════════════════════════════

# Expert rules: hardcoded known-effective grafts
@dataclass
class ExpertGraftRule:
    """A hardcoded expert graft rule."""
    rule_id: str
    host_pattern: str          # Tag or name pattern for host
    donor_pattern: str         # Tag or name pattern for donor
    host_opcode_sequence: list[str]   # opcodes to look for in host
    description: str = ""
    confidence: float = 0.7


_EXPERT_RULES: list[ExpertGraftRule] = [
    ExpertGraftRule(
        rule_id="mmse_init_for_kbest",
        host_pattern="kbest",
        donor_pattern="lmmse",
        host_opcode_sequence=["const", "call"],
        description="Replace K-Best initial estimate with MMSE pre-filter",
        confidence=0.8,
    ),
    ExpertGraftRule(
        rule_id="bp_sweep_for_ep",
        host_pattern="ep",
        donor_pattern="bp",
        host_opcode_sequence=["call"],
        description="Replace EP cavity computation with BP message passing sweep",
        confidence=0.6,
    ),
    ExpertGraftRule(
        rule_id="amp_denoise_for_ep",
        host_pattern="ep",
        donor_pattern="amp",
        host_opcode_sequence=["call"],
        description="Replace EP site update with AMP denoiser",
        confidence=0.65,
    ),
    ExpertGraftRule(
        rule_id="osic_ordering_for_kbest",
        host_pattern="kbest",
        donor_pattern="osic",
        host_opcode_sequence=["call"],
        description="Use OSIC ordering strategy in K-Best",
        confidence=0.7,
    ),
    ExpertGraftRule(
        rule_id="lmmse_regularizer_for_zf",
        host_pattern="zf",
        donor_pattern="lmmse",
        host_opcode_sequence=["binary", "call"],
        description="Add LMMSE regularization to ZF",
        confidence=0.9,
    ),
]


class ExpertPatternMatcher:
    """Apply expert-knowledge graft rules.

    Matches host/donor by tags or IR function name, then locates
    matching opcode sequences in the host's IR.
    """

    def __init__(
        self,
        rules: list[ExpertGraftRule] | None = None,
        max_proposals_per_gen: int = 3,
    ):
        self.rules = rules or _EXPERT_RULES
        self.max_proposals_per_gen = max_proposals_per_gen

    def __call__(
        self,
        entries: list[AlgorithmEntry],
        generation: int,
    ) -> list[GraftProposal]:
        proposals: list[GraftProposal] = []

        for rule in self.rules:
            if len(proposals) >= self.max_proposals_per_gen:
                break

            hosts = [e for e in entries if self._matches(e, rule.host_pattern)]
            donors = [e for e in entries if self._matches(e, rule.donor_pattern)]

            for host in hosts:
                for donor in donors:
                    if host.algo_id == donor.algo_id:
                        continue
                    proposal = self._try_rule(rule, host, donor)
                    if proposal is not None:
                        proposals.append(proposal)
                        if len(proposals) >= self.max_proposals_per_gen:
                            break
                if len(proposals) >= self.max_proposals_per_gen:
                    break

        return proposals

    def _matches(self, entry: AlgorithmEntry, pattern: str) -> bool:
        """Check if entry matches a pattern (tag or name substring)."""
        p = pattern.lower()
        if any(p in t.lower() for t in entry.tags):
            return True
        if hasattr(entry.ir, 'name') and p in entry.ir.name.lower():
            return True
        if p in entry.algo_id.lower():
            return True
        return False

    def _try_rule(
        self,
        rule: ExpertGraftRule,
        host: AlgorithmEntry,
        donor: AlgorithmEntry,
    ) -> GraftProposal | None:
        """Try to apply an expert rule to a host/donor pair."""
        # Find matching opcode sequence in host
        for block_id, block in host.ir.blocks.items():
            match_ops = self._find_opcode_sequence(
                host.ir, block.op_ids, rule.host_opcode_sequence,
            )
            if match_ops:
                region = _build_region_from_ops(host.ir, match_ops)
                return GraftProposal(
                    proposal_id=_fresh_id(f"expert_{rule.rule_id}"),
                    host_algo_id=host.algo_id,
                    region=region,
                    contract=None,
                    donor_algo_id=donor.algo_id,
                    donor_ir=donor.ir,
                    dependency_overrides=[],
                    confidence=rule.confidence,
                    rationale=rule.description,
                )
        return None

    def _find_opcode_sequence(
        self,
        ir: FunctionIR,
        block_op_ids: list[str],
        target_opcodes: list[str],
    ) -> list[str] | None:
        """Find the first occurrence of a sequence of opcodes in block ops."""
        ops_list = [
            (oid, ir.ops[oid].opcode)
            for oid in block_op_ids
            if oid in ir.ops
        ]
        n = len(target_opcodes)
        for i in range(len(ops_list) - n + 1):
            window = [ops_list[i + j][1] for j in range(n)]
            if window == target_opcodes:
                return [ops_list[i + j][0] for j in range(n)]
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 3. StaticStructurePatternMatcher
# ═══════════════════════════════════════════════════════════════════════════

class StaticStructurePatternMatcher:
    """IR-structure-based pattern matching.

    Analyses each algorithm's IR to identify structural patterns:
      - Iterative computation (while loop + matrix ops)
      - Sorting/selection patterns
      - Distance computation patterns

    Then proposes cross-algorithm grafts for identified patterns.
    """

    def __init__(
        self,
        max_proposals_per_gen: int = 3,
        min_pattern_ops: int = 2,
    ):
        self.max_proposals_per_gen = max_proposals_per_gen
        self.min_pattern_ops = min_pattern_ops

    def __call__(
        self,
        entries: list[AlgorithmEntry],
        generation: int,
    ) -> list[GraftProposal]:
        # Build structural fingerprints for all entries
        fingerprints = {
            e.algo_id: self._fingerprint(e.ir) for e in entries
        }

        proposals: list[GraftProposal] = []
        for i, host in enumerate(entries):
            if len(proposals) >= self.max_proposals_per_gen:
                break
            for j, donor in enumerate(entries):
                if i == j:
                    continue
                proposal = self._match_structures(
                    host, donor,
                    fingerprints[host.algo_id],
                    fingerprints[donor.algo_id],
                )
                if proposal is not None:
                    proposals.append(proposal)
                    if len(proposals) >= self.max_proposals_per_gen:
                        break

        return proposals

    def _fingerprint(self, ir: FunctionIR) -> dict[str, Any]:
        """Compute a structural fingerprint of an IR."""
        opcode_counts: dict[str, int] = {}
        has_loop = False
        has_branch = False
        call_targets: list[str] = []

        for op in ir.ops.values():
            opcode_counts[op.opcode] = opcode_counts.get(op.opcode, 0) + 1
            if op.opcode == "branch":
                has_branch = True
            if op.opcode == "jump":
                has_loop = True  # jump back often indicates loop
            if op.opcode == "call":
                callee = op.attrs.get("callee") or op.attrs.get("name", "")
                if callee:
                    call_targets.append(callee)

        # Check for back-edges (loops)
        for block in ir.blocks.values():
            for succ in block.succs:
                if succ in block.preds:
                    has_loop = True

        return {
            "opcode_counts": opcode_counts,
            "has_loop": has_loop,
            "has_branch": has_branch,
            "call_targets": call_targets,
            "n_ops": len(ir.ops),
            "n_blocks": len(ir.blocks),
        }

    def _match_structures(
        self,
        host: AlgorithmEntry,
        donor: AlgorithmEntry,
        host_fp: dict[str, Any],
        donor_fp: dict[str, Any],
    ) -> GraftProposal | None:
        """Try to match complementary structures between host and donor."""
        # Strategy: if host has a loop body and donor has a different loop body,
        # propose replacing host's loop body with donor's
        if host_fp["has_loop"] and donor_fp["has_loop"]:
            host_loop_ops = self._find_loop_body(host.ir)
            if len(host_loop_ops) >= self.min_pattern_ops:
                region = _build_region_from_ops(host.ir, host_loop_ops)
                return GraftProposal(
                    proposal_id=_fresh_id("static_loop_swap"),
                    host_algo_id=host.algo_id,
                    region=region,
                    contract=None,
                    donor_algo_id=donor.algo_id,
                    donor_ir=donor.ir,
                    dependency_overrides=[],
                    confidence=0.5,
                    rationale="Replace loop body with donor's iterative computation",
                )

        # Strategy: if host has call ops and donor has different call ops,
        # propose replacing a call with donor's version
        host_calls = [
            oid for oid, op in host.ir.ops.items()
            if op.opcode == "call" and not op.attrs.get("grafted")
        ]
        if host_calls and donor_fp["n_ops"] > 0:
            # Pick first non-grafted call
            target_call = host_calls[0]
            region = _build_region_from_ops(host.ir, [target_call])
            return GraftProposal(
                proposal_id=_fresh_id("static_call_replace"),
                host_algo_id=host.algo_id,
                region=region,
                contract=None,
                donor_algo_id=donor.algo_id,
                donor_ir=donor.ir,
                dependency_overrides=[],
                confidence=0.4,
                rationale="Replace call op with donor implementation",
            )

        return None

    def _find_loop_body(self, ir: FunctionIR) -> list[str]:
        """Find ops that constitute a loop body (block with back-edge)."""
        for block_id, block in ir.blocks.items():
            # Check for back-edge
            is_loop = any(succ in block.preds for succ in block.succs)
            if is_loop:
                return [
                    oid for oid in block.op_ids
                    if ir.ops.get(oid)
                    and ir.ops[oid].opcode not in ("branch", "jump", "return")
                ]
        # No loop found — return empty
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Composite matcher
# ═══════════════════════════════════════════════════════════════════════════

class CompositePatternMatcher:
    """Combine multiple PatternMatchers."""

    def __init__(self, matchers: list[Any]):
        self.matchers = matchers

    def __call__(
        self,
        entries: list[AlgorithmEntry],
        generation: int,
    ) -> list[GraftProposal]:
        proposals: list[GraftProposal] = []
        for matcher in self.matchers:
            proposals.extend(matcher(entries, generation))
        return proposals
