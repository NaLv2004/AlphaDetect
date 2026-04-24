"""Data type definitions for algorithm-level evolution.

Defines AlgorithmEntry, GraftProposal, DependencyOverride, PatternMatcherFn,
SlotDescriptor, SlotPopulation, and AlgorithmGenome — the core types for
the two-level (macro-structure + micro-slot) evolution framework.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from algorithm_ir.ir.model import FunctionIR
from evolution.fitness import FitnessResult
from evolution.skeleton_registry import ProgramSpec

if TYPE_CHECKING:
    from algorithm_ir.factgraph.model import FactGraph
    from algorithm_ir.region.contract import BoundaryContract
    from algorithm_ir.region.selector import RewriteRegion
    from algorithm_ir.runtime.tracer import RuntimeEvent, RuntimeValue


# ---------------------------------------------------------------------------
# Slot Descriptor — describes one evolvable slot in a hierarchical algorithm
# ---------------------------------------------------------------------------

@dataclass
class SlotDescriptor:
    """Describes an evolvable slot in a hierarchical algorithm."""

    # Identity
    slot_id: str               # Global unique, e.g. "kbest.expand.local_cost"
    short_name: str            # e.g. "local_cost"

    # Hierarchy
    level: int                 # 0=atomic, 1=composite, 2=module, 3=algorithm
    depth: int                 # Nesting depth within owning algorithm (0=top)
    parent_slot_id: str | None
    child_slot_ids: list[str] = field(default_factory=list)

    # Type signature
    spec: ProgramSpec = field(default_factory=lambda: ProgramSpec(
        name="unnamed", param_names=[], param_types=[], return_type="float",
    ))

    # Default implementation
    default_impl: FunctionIR | None = None
    default_impl_level: int = 0

    # Evolution control
    mutable: bool = True
    evolution_weight: float = 1.0
    max_complexity: int = 50

    # Meta
    description: str = ""
    domain_tags: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# SlotPopulation — sub-population of variants for one slot
# ---------------------------------------------------------------------------

@dataclass
class SlotPopulation:
    """A sub-population of implementation variants for one slot."""

    slot_id: str
    spec: ProgramSpec
    variants: list[FunctionIR] = field(default_factory=list)
    fitness: list[float] = field(default_factory=list)
    best_idx: int = 0
    # Optional source strings for variants that cannot be compiled to IR
    # (e.g. complex defaults using keyword args unsupported by IR builder).
    # When present, materialization prefers source_variants[i] over
    # emit_python_source(variants[i]).
    source_variants: list[str | None] = field(default_factory=list)

    @property
    def best_variant(self) -> FunctionIR:
        if not self.variants:
            raise ValueError(f"No variants in slot {self.slot_id}")
        return self.variants[self.best_idx]

    @property
    def best_fitness(self) -> float:
        if not self.fitness:
            return float("inf")
        return self.fitness[self.best_idx]

    def __len__(self) -> int:
        return len(self.variants)


# ---------------------------------------------------------------------------
# GraftRecord — provenance of a single graft operation
# ---------------------------------------------------------------------------

@dataclass
class GraftRecord:
    """Records one graft operation in an algorithm's lineage."""

    generation: int
    host_algo_id: str
    donor_algo_id: str | None
    proposal_id: str
    region_summary: str
    new_slots_created: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AlgorithmGenome — single canonical flat IR with slot annotations
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmGenome:
    """An evolved individual = ONE flat annotated FunctionIR.

    Single-representation principle:
      - ``ir``: the SOLE canonical IR. It is fully inlined / flat — every
        slot helper has been expanded at construction time. Per-op slot
        provenance is carried in ``op.attrs["_provenance"]`` (a dict with
        keys ``from_slot_id``, ``call_site_id``, ``variant_idx``,
        ``slot_pop_key``, ``is_slot_boundary``=False, ``boundary_kind``=None).
      - ``slot_populations``: PURELY ANNOTATION-LEVEL metadata. Holds
        snapshot history / micro-evolution candidates for each slot region
        identified by ``from_slot_id``. Does NOT drive any IR rebuild;
        the IR is never re-inlined from variants.

    Grafting and rediscovery operate directly on ``ir``. After a graft,
    annotations on the cloned donor ops persist; the host ops keep their
    annotations; ``maybe_rediscover_slots`` may add new annotations for
    cohesive regions that became unaffiliated.

    The legacy field ``structural_ir`` is exposed as a read-only property
    aliasing ``ir`` for transitional source-compat with code that has not
    yet been migrated.
    """

    algo_id: str
    ir: FunctionIR
    slot_populations: dict[str, SlotPopulation] = field(default_factory=dict)
    constants: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    graft_history: list[GraftRecord] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Backward-compat: ``structural_ir`` aliases ``ir``.
    # New code MUST use ``ir`` directly. Writes are accepted only as a
    # transitional shim; they update ``ir``.
    # ------------------------------------------------------------------
    @property
    def structural_ir(self) -> FunctionIR:
        return self.ir

    @structural_ir.setter
    def structural_ir(self, value: FunctionIR) -> None:
        self.ir = value

    @staticmethod
    def _make_id() -> str:
        return f"algo_{uuid.uuid4().hex[:8]}"

    def clone(self) -> AlgorithmGenome:
        """Deep copy (without caches)."""
        return AlgorithmGenome(
            algo_id=AlgorithmGenome._make_id(),
            ir=deepcopy(self.ir),
            slot_populations={
                k: SlotPopulation(
                    slot_id=v.slot_id,
                    spec=v.spec,
                    variants=[deepcopy(ir) for ir in v.variants],
                    fitness=list(v.fitness),
                    best_idx=v.best_idx,
                    source_variants=list(v.source_variants) if v.source_variants else [],
                )
                for k, v in self.slot_populations.items()
            },
            constants=self.constants.copy(),
            generation=self.generation,
            parent_ids=list(self.parent_ids),
            graft_history=list(self.graft_history),
            tags=set(self.tags),
            metadata=deepcopy(self.metadata),
        )

    def to_entry(
        self,
        fitness: FitnessResult | None = None,
        use_fii_view: bool = True,  # kept for API compat; ignored — single IR is always exposed
    ) -> AlgorithmEntry:
        """Convert this genome into an AlgorithmEntry for PatternMatcher.

        With the single-IR refactor, ``use_fii_view`` is a no-op kept for
        call-site compatibility — the canonical flat annotated IR is the
        only IR there is, and it is always exposed as ``entry.ir``.
        """
        from algorithm_ir.regeneration.codegen import emit_python_source

        active_ir = self.ir
        source = None
        try:
            source = emit_python_source(active_ir)
        except Exception:
            pass

        slot_tree: dict[str, SlotDescriptor] | None = None
        slot_fitness_map: dict[str, float] | None = None
        if self.slot_populations:
            slot_tree = {}
            slot_fitness_map = {}
            for key, pop in self.slot_populations.items():
                slot_tree[key] = SlotDescriptor(
                    slot_id=pop.slot_id,
                    short_name=pop.slot_id.split(".")[-1],
                    level=0,
                    depth=0,
                    parent_slot_id=None,
                    spec=pop.spec,
                )
                slot_fitness_map[key] = pop.best_fitness

        entry = AlgorithmEntry(
            algo_id=self.algo_id,
            ir=active_ir,
            source=source,
            trace=self.metadata.get("_cached_trace"),
            runtime_values=self.metadata.get("_cached_runtime_values"),
            factgraph=self.metadata.get("_cached_factgraph"),
            fitness=fitness,
            generation=self.generation,
            provenance={"parent_ids": list(self.parent_ids)},
            tags=set(self.tags),
            slot_tree=slot_tree,
            slot_fitness=slot_fitness_map,
        )
        # Legacy alias: code that still consults ``entry.fii_ir`` (e.g.
        # the GNN matcher) gets the same canonical IR.
        entry.fii_ir = active_ir
        return entry


# ---------------------------------------------------------------------------
# AlgorithmEntry — what the PatternMatcher sees
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmEntry:
    """An entry in the algorithm pool, visible to the PatternMatcher."""

    algo_id: str
    ir: FunctionIR
    source: str | None = None
    trace: list[Any] | None = None       # list[RuntimeEvent]
    runtime_values: dict[str, Any] | None = None  # dict[str, RuntimeValue]
    factgraph: Any | None = None         # FactGraph
    fitness: FitnessResult | None = None
    generation: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)

    # Hierarchy info
    level: int = 3
    slot_tree: dict[str, SlotDescriptor] | None = None
    slot_fitness: dict[str, float] | None = None

    # FII view: set when ``to_entry(use_fii_view=True)`` succeeded.
    # When non-None, grafting should operate on this IR (which has no
    # algslot ops) and produce flat-genome offspring. When None, fall
    # back to structural_ir flow.
    fii_ir: FunctionIR | None = None


# ---------------------------------------------------------------------------
# DependencyOverride — proposed dependency change
# ---------------------------------------------------------------------------

@dataclass
class DependencyOverride:
    """A proposed change to data-flow dependencies during grafting."""

    target_value: str
    new_dependencies: list[str] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# GraftProposal — a PatternMatcher's suggestion
# ---------------------------------------------------------------------------

@dataclass
class GraftProposal:
    """A grafting suggestion from the PatternMatcher."""

    proposal_id: str

    # Host side
    host_algo_id: str
    region: Any           # RewriteRegion
    contract: Any         # BoundaryContract

    # Donor side
    donor_algo_id: str | None = None
    donor_ir: FunctionIR | None = None
    donor_region: Any | None = None  # RewriteRegion (metadata only)

    # Dependency overrides
    dependency_overrides: list[DependencyOverride] = field(default_factory=list)

    # Port binding: host_value_id → donor_value_id
    port_mapping: dict[str, str] = field(default_factory=dict)

    # Meta
    confidence: float = 0.5
    rationale: str = ""


# ---------------------------------------------------------------------------
# PatternMatcherFn — type alias for the matcher callback
# ---------------------------------------------------------------------------

PatternMatcherFn = Callable[
    [list[AlgorithmEntry], int],    # (current pool, current generation)
    list[GraftProposal],            # returned graft proposals
]


# ---------------------------------------------------------------------------
# AlgorithmFitnessEvaluator — abstract evaluator for complete algorithms
# ---------------------------------------------------------------------------

class AlgorithmFitnessEvaluator(ABC):
    """Evaluates a complete algorithm's fitness.

    Differences from FitnessEvaluator:
      - Input is AlgorithmGenome (complete algorithm), not IRGenome
      - Can access structural_ir and slot_populations for complexity penalty
      - evaluate_single_result() for fast micro-level slot evaluation
    """

    @abstractmethod
    def evaluate(self, genome: AlgorithmGenome) -> FitnessResult:
        """Evaluate a single algorithm."""
        ...

    def evaluate_batch(
        self, genomes: list[AlgorithmGenome],
    ) -> list[FitnessResult]:
        """Batch evaluation. Defaults to sequential evaluate()."""
        return [self.evaluate(g) for g in genomes]

    @abstractmethod
    def evaluate_single_result(self, result: Any) -> float:
        """Fast evaluation of one execution result (for micro-level).

        Returns scalar fitness (lower is better).
        """
        ...


# ---------------------------------------------------------------------------
# AlgorithmEvolutionConfig — config for the two-level engine
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmEvolutionConfig:
    """Configuration for the two-level evolution engine."""

    # Macro layer
    pool_size: int = 20
    n_generations: int = 100

    # Micro layer
    micro_pop_size: int = 32
    micro_generations: int = 3
    micro_mutation_rate: float = 0.6

    # Constant perturbation
    constant_perturb_rate: float = 0.1
    constant_perturb_scale: float = 0.05

    # Hierarchical evolution
    level_mutation_probs: dict[int, float] = field(default_factory=lambda: {
        0: 0.05, 1: 0.30, 2: 0.40, 3: 0.25,
    })
    depth_decay: float = 0.7
    allow_cross_level_graft: bool = True

    # Fully-Inlined IR (FII) view. When True, the GNN sees slot internals
    # as first-class ops (not opaque algslot atoms) and grafts operate on
    # the inlined IR. Grafted offspring are FLAT (no slot populations) —
    # micro-evolution has no effect on them. See ``evolution/fii.py``.
    use_fii_view: bool = False

    seed: int = 42
