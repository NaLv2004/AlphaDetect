"""Deep-tree mutation & crossover tests.

Verifies that:
  * mutations actually reach into nested code_blocks (not just top level)
  * subtree crossover transplants nodes from any depth in donor to any
    depth in recipient, and vice versa
  * the resulting programs remain runnable on the VM (no crashes)
  * tree-depth invariants (max_recur_depth) are preserved
  * Exec.DoTimes inside Exec.DoTimes is reachable by the operators
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from pushgp.crossover import (
    crossover_alternation,
    crossover_program,
    crossover_subtree,
    _all_node_positions as _xover_positions,
    _subtree_max_depth,
)
from pushgp.mutation import (
    _all_node_positions,
    _all_list_positions,
    mut_block,
    mut_delete,
    mut_grow,
    mut_insert,
    mut_point,
    mut_segment,
    mut_swap,
)
from pushgp.program import Instruction, deep_copy_program, program_length
from pushgp.random_program import C2V_INSTR, RandomProgramGenerator, V2C_INSTR
from pushgp.validators import _make_vm, _run
from pushgp.vm import VM


# ============================================================ Fixtures


def I(name, *, b1=None, b2=None):
    return Instruction(name=name, code_block=b1, code_block2=b2)


def _deep_v2c():
    """Hand-crafted V2C with two nested DoTimes (depth=2 children)."""
    return [
        I("FVec.Len"),
        I("Exec.DoTimes", b1=[
            I("FVec.At"),
            I("Float.Add"),
            I("Exec.DoTimes", b1=[
                I("Float.Const1"),
                I("Float.Add"),
            ]),
        ]),
    ]


def _wide_c2v():
    """Hand-crafted C2V with an Exec.If having two non-trivial blocks."""
    return [
        I("Bool.True"),
        I("Exec.If",
          b1=[I("Float.Const0"), I("Float.Add")],
          b2=[I("Float.Const1"), I("Float.Mul")]),
    ]


def _rng(seed):
    return np.random.default_rng(seed)


# ============================================================ Traversal


class TestTraversal:
    def test_node_positions_visit_all_depths(self):
        prog = _deep_v2c()
        positions = _all_node_positions(prog)
        depths = sorted({d for _, _, d in positions})
        assert depths == [0, 1, 2], (
            f"expected nodes at depths 0,1,2 but got {depths}"
        )
        # Total count = top(2) + DoTimes-body(3) + inner-DoTimes-body(2) = 7
        assert len(positions) == 7

    def test_list_positions_include_inner_blocks(self):
        prog = _deep_v2c()
        lists = _all_list_positions(prog)
        depths = sorted({d for _, d in lists})
        assert depths == [0, 1, 2]
        # 1 top + 1 DoTimes body + 1 inner DoTimes body = 3 lists
        assert len(lists) == 3

    def test_two_block_node_exposes_both_blocks(self):
        prog = _wide_c2v()
        lists = _all_list_positions(prog)
        # top + b1 + b2 = 3 lists
        assert len(lists) == 3
        depths = sorted([d for _, d in lists])
        assert depths == [0, 1, 1]

    def test_subtree_max_depth(self):
        atomic = I("Float.Add")
        assert _subtree_max_depth(atomic) == 0
        flat = I("Exec.DoTimes", b1=[I("Float.Add")])
        assert _subtree_max_depth(flat) == 1
        nested = I("Exec.DoTimes", b1=[
            I("Exec.DoTimes", b1=[I("Float.Add")])])
        assert _subtree_max_depth(nested) == 2


# ============================================================ Mutation reaches inside


class TestMutationReachesInside:
    """Run each operator MANY times and verify it actually edits inner blocks."""

    N_TRIALS = 400

    @pytest.fixture
    def gen(self):
        return RandomProgramGenerator(rng=_rng(123))

    def _mutated_inner(self, op, gen, instr_set, prog) -> int:
        """How often does `op` change something inside a nested block?"""
        rng = _rng(7)
        original_top = [ins.name for ins in prog]

        def inner_signature(p):
            # Snapshot of every nested block's name sequence (ignoring outer).
            sigs = []

            def rec(lst):
                for ins in lst:
                    if ins.code_block is not None:
                        sigs.append(tuple(c.name for c in ins.code_block))
                        rec(ins.code_block)
                    if ins.code_block2 is not None:
                        sigs.append(tuple(c.name for c in ins.code_block2))
                        rec(ins.code_block2)

            rec(p)
            return tuple(sigs)

        baseline = inner_signature(prog)
        n_inner_changed = 0
        for _ in range(self.N_TRIALS):
            child = op(prog, rng, gen, instr_set)
            if inner_signature(child) != baseline:
                n_inner_changed += 1
        return n_inner_changed

    def test_point_reaches_inner(self, gen):
        n = self._mutated_inner(mut_point, gen, V2C_INSTR, _deep_v2c())
        # Inner positions are 5 of 7 total (depths 1+2). So ≥ 50% expected.
        assert n > 100, f"mut_point changed inner only {n}/{self.N_TRIALS} times"

    def test_insert_reaches_inner(self, gen):
        n = self._mutated_inner(mut_insert, gen, V2C_INSTR, _deep_v2c())
        # 3 lists total; 2 of them inner. ≥ 30% conservative.
        assert n > 100, f"mut_insert reached inner only {n}/{self.N_TRIALS} times"

    def test_delete_reaches_inner(self, gen):
        n = self._mutated_inner(mut_delete, gen, V2C_INSTR, _deep_v2c())
        assert n > 100, f"mut_delete reached inner only {n}/{self.N_TRIALS} times"

    def test_swap_reaches_inner(self, gen):
        n = self._mutated_inner(mut_swap, gen, V2C_INSTR, _deep_v2c())
        # Either both positions inside or one inside one outside changes
        # the inner signature.  Should be very common.
        assert n > 100, f"mut_swap reached inner only {n}/{self.N_TRIALS} times"

    def test_block_reaches_inner_when_only_inner_control(self, gen):
        # Outer DoTimes is at top, inner DoTimes is also a control.
        # Either being chosen changes inner_signature.
        n = self._mutated_inner(mut_block, gen, V2C_INSTR, _deep_v2c())
        assert n > 50, f"mut_block reached inner only {n}/{self.N_TRIALS} times"

    def test_segment_reaches_inner(self, gen):
        n = self._mutated_inner(mut_segment, gen, V2C_INSTR, _deep_v2c())
        assert n > 80, f"mut_segment reached inner only {n}/{self.N_TRIALS} times"

    def test_grow_reaches_inner(self, gen):
        n = self._mutated_inner(mut_grow, gen, V2C_INSTR, _deep_v2c())
        assert n > 80, f"mut_grow reached inner only {n}/{self.N_TRIALS} times"


# ============================================================ Mutation closure


class TestMutationClosure:
    """Mutated programs must remain runnable; depth invariants preserved."""

    @pytest.fixture
    def gen(self):
        return RandomProgramGenerator(rng=_rng(99), max_recur_depth=2)

    def _max_depth(self, prog):
        d_max = 0

        def rec(lst, d):
            nonlocal d_max
            for ins in lst:
                d_max = max(d_max, d)
                if ins.code_block is not None:
                    rec(ins.code_block, d + 1)
                if ins.code_block2 is not None:
                    rec(ins.code_block2, d + 1)

        rec(prog, 0)
        return d_max

    def test_all_ops_keep_program_runnable(self, gen):
        prog = _deep_v2c()
        rng = _rng(2025)
        evo_consts = np.ones(8)
        for op in (mut_point, mut_insert, mut_delete, mut_swap,
                   mut_block, mut_segment, mut_grow):
            for _ in range(50):
                child = op(prog, rng, gen, V2C_INSTR)
                vm = _make_vm(np.array([1.0, -2.0, 3.0]),
                              channel_llr=0.5,
                              evo_consts=evo_consts, deg=4)
                # Must not raise
                _run(child, vm, "v2c")

    def test_max_depth_respected(self, gen):
        prog = _deep_v2c()
        rng = _rng(11)
        for op in (mut_point, mut_insert, mut_block, mut_segment, mut_grow):
            for _ in range(80):
                child = op(prog, rng, gen, V2C_INSTR)
                d = self._max_depth(child)
                # max_recur_depth=2 means inner-inner blocks (depth=2 nodes,
                # i.e. nodes with parents at depth 1) are allowed but those
                # depth-2 nodes themselves must NOT have code_blocks → max
                # observed depth is 2.
                assert d <= 2, (
                    f"{op.__name__} produced depth-{d} tree, max=2"
                )

    def test_program_length_capped(self, gen):
        prog = _deep_v2c()
        rng = _rng(31)
        for op in (mut_point, mut_insert, mut_block, mut_segment, mut_grow):
            for _ in range(50):
                child = op(prog, rng, gen, V2C_INSTR)
                assert program_length(child) <= 80


# ============================================================ Subtree crossover


class TestSubtreeCrossover:
    """Subtree crossover must transplant from any depth in donor to any
    depth in recipient."""

    @pytest.fixture
    def gen(self):
        return RandomProgramGenerator(rng=_rng(42))

    def test_subtree_xover_can_transplant_inner_to_top(self, gen):
        """Use a recipient with NO controls and a donor whose inner block
        contains a unique marker; verify that marker can land at top of
        the child."""
        recipient = [I("Float.Const0"), I("Float.Add"), I("Float.Mul"),
                     I("Float.Sub")]
        # Donor: top-level innocuous, inner block has marker `Float.Pi`.
        donor = [
            I("Float.Const1"),
            I("Exec.DoTimes", b1=[
                I("Float.Const2"),
                I("Float.ConstPi"),  # the marker
            ]),
        ]
        rng = _rng(0)
        marker_at_top = 0
        for _ in range(500):
            child = crossover_subtree(recipient, donor, rng,
                                       max_recur_depth=2,
                                       instr_set=V2C_INSTR)
            for ins in child:
                if ins.name == "Float.ConstPi":
                    marker_at_top += 1
                    break
        assert marker_at_top > 5, (
            f"inner-marker reached top only {marker_at_top}/500 times"
        )

    def test_subtree_xover_can_transplant_to_inner(self, gen):
        """Donor has a UNIQUE top-level marker; recipient has a DoTimes
        body — verify the marker can land INSIDE that body."""
        recipient = [
            I("FVec.Len"),
            I("Exec.DoTimes", b1=[
                I("Float.Const0"),
                I("Float.Add"),
            ]),
        ]
        donor = [I("Float.ConstPi"), I("Float.Mul"), I("Float.Sub")]
        rng = _rng(1)
        marker_inside = 0
        for _ in range(500):
            child = crossover_subtree(recipient, donor, rng,
                                       max_recur_depth=2,
                                       instr_set=V2C_INSTR)
            for ins in child:
                if ins.code_block is not None:
                    if any(c.name == "Float.ConstPi" for c in ins.code_block):
                        marker_inside += 1
                        break
        assert marker_inside > 5, (
            f"top-marker reached inner only {marker_inside}/500 times"
        )

    def test_subtree_xover_preserves_max_depth(self, gen):
        rng = _rng(5)
        for _ in range(200):
            recipient = gen.random_program(V2C_INSTR, 4, 12)
            donor = gen.random_program(V2C_INSTR, 4, 12)
            child = crossover_subtree(recipient, donor, rng,
                                      max_recur_depth=2,
                                      instr_set=V2C_INSTR)
            d_max = 0

            def rec(lst, d):
                nonlocal d_max
                for ins in lst:
                    d_max = max(d_max, d)
                    if ins.code_block is not None:
                        rec(ins.code_block, d + 1)
                    if ins.code_block2 is not None:
                        rec(ins.code_block2, d + 1)

            rec(child, 0)
            assert d_max <= 2

    def test_subtree_xover_runs_on_vm(self, gen):
        rng = _rng(7)
        for _ in range(100):
            p1 = gen.random_program(V2C_INSTR, 4, 12)
            p2 = gen.random_program(V2C_INSTR, 4, 12)
            child = crossover_subtree(p1, p2, rng,
                                      max_recur_depth=2,
                                      instr_set=V2C_INSTR)
            vm = _make_vm(np.array([1.0, -1.0, 0.5]),
                          channel_llr=0.3,
                          evo_consts=np.ones(8), deg=4)
            _run(child, vm, "v2c")  # must not raise

    def test_subtree_xover_does_not_mutate_parents(self, gen):
        recipient = _deep_v2c()
        donor = _wide_c2v()
        recipient_snap = deep_copy_program(recipient)
        donor_snap = deep_copy_program(donor)
        rng = _rng(13)
        for _ in range(100):
            crossover_subtree(recipient, donor, rng,
                              max_recur_depth=2,
                              instr_set=V2C_INSTR)
        # Compare names + structure
        def sig(prog):
            return [(ins.name,
                     [c.name for c in ins.code_block] if ins.code_block else None,
                     [c.name for c in ins.code_block2] if ins.code_block2 else None)
                    for ins in prog]

        assert sig(recipient) == sig(recipient_snap)
        assert sig(donor) == sig(donor_snap)


# ============================================================ Auto crossover


class TestAutoCrossover:
    @pytest.fixture
    def gen(self):
        return RandomProgramGenerator(rng=_rng(0))

    def test_auto_uses_both_modes(self, gen):
        """With p_subtree=0.5, both modes should be observed."""
        rng = _rng(2)
        # Use programs distinguishable: alternation always preserves
        # exactly the prefix of p1 + suffix of p2 at top level (same
        # internal blocks).  Subtree may produce blocks with structure
        # that wouldn't appear under alternation.
        p1 = [I("Float.Const0"), I("Float.Add")]
        p2 = [
            I("Exec.DoTimes", b1=[I("Float.Const1"), I("Float.Mul")]),
        ]
        seen_with_block = 0
        seen_without_block = 0
        for _ in range(300):
            child = crossover_program(p1, p2, rng,
                                      mode="auto",
                                      p_subtree=0.5,
                                      max_recur_depth=2,
                                      instr_set=V2C_INSTR)
            has_ctrl = any(ins.is_control() for ins in child)
            if has_ctrl:
                seen_with_block += 1
            else:
                seen_without_block += 1
        # Subtree may extract the inner Float.Const1/Float.Mul → no ctrl;
        # alternation always keeps DoTimes whole if cut spans into p2.
        # Both must occur:
        assert seen_with_block > 30 and seen_without_block > 30


# ============================================================ DoTimes nests DoTimes


class TestNestedControl:
    """Verify that Exec.DoTimes can legitimately appear inside another
    Exec.DoTimes after evolutionary operations."""

    def test_random_generator_can_produce_nested_control(self):
        gen = RandomProgramGenerator(rng=_rng(3), max_recur_depth=2)
        nested_seen = False
        for _ in range(200):
            prog = gen.random_program(V2C_INSTR, 6, 18)
            for ins in prog:
                if ins.is_control() and ins.code_block:
                    if any(c.is_control() for c in ins.code_block):
                        nested_seen = True
                        break
            if nested_seen:
                break
        assert nested_seen

    def test_subtree_crossover_can_produce_dotimes_inside_dotimes(self):
        """Recipient has a top-level DoTimes(empty body w/ atomic).  Donor
        has a top-level DoTimes.  Subtree xover that picks the donor
        DoTimes as donor and the body of the recipient DoTimes as the
        recipient site should yield DoTimes-in-DoTimes."""
        recipient = [
            I("FVec.Len"),
            I("Exec.DoTimes", b1=[I("Float.Const0"), I("Float.Add")]),
        ]
        donor = [
            I("Exec.DoTimes", b1=[I("Float.Const1"), I("Float.Mul")]),
        ]
        rng = _rng(0)
        nested_seen = False
        for _ in range(2000):
            child = crossover_subtree(recipient, donor, rng,
                                      max_recur_depth=2,
                                      instr_set=V2C_INSTR)
            for ins in child:
                if ins.is_control() and ins.code_block:
                    if any(c.is_control() for c in ins.code_block):
                        nested_seen = True
                        break
            if nested_seen:
                break
        assert nested_seen, "DoTimes-in-DoTimes never produced by subtree xover"

    def test_mutation_can_produce_nested_control(self):
        gen = RandomProgramGenerator(rng=_rng(8), max_recur_depth=2)
        rng = _rng(11)
        # Start from a flat program; check that mut_point at an inner
        # position can introduce a control there.
        recipient = [
            I("FVec.Len"),
            I("Exec.DoTimes", b1=[I("Float.Const0"), I("Float.Add")]),
        ]
        nested_seen = False
        for _ in range(1500):
            child = mut_point(recipient, rng, gen, V2C_INSTR)
            for ins in child:
                if ins.is_control() and ins.code_block:
                    if any(c.is_control() for c in ins.code_block):
                        nested_seen = True
                        break
            if nested_seen:
                break
        assert nested_seen, "mut_point never introduced nested control"


# ============================================================ End-to-end sanity


class TestEndToEnd:
    def test_combined_evolution_step_runs_clean(self):
        from pushgp.crossover import crossover_genome
        from pushgp.mutation import mutate_genome
        from pushgp.genome import Genome

        gen = RandomProgramGenerator(rng=_rng(0), max_recur_depth=2)
        rng = _rng(0)
        g1 = Genome(prog_v2c=_deep_v2c(), prog_c2v=_wide_c2v())
        g2 = Genome(prog_v2c=gen.random_v2c(6, 14),
                    prog_c2v=gen.random_c2v(6, 14))
        for _ in range(50):
            child = crossover_genome(g1, g2, rng, mode="auto", max_recur_depth=2)
            child = mutate_genome(child, rng, gen, n_mutations=3)
            assert program_length(child.prog_v2c) <= 80
            assert program_length(child.prog_c2v) <= 80
            # Both progs must be runnable
            vm = _make_vm(np.array([1.0, -1.0, 0.5]),
                          channel_llr=0.3,
                          evo_consts=np.ones(8), deg=4)
            _run(child.prog_v2c, vm, "v2c")
            vm2 = _make_vm(np.array([1.0, -1.0, 0.5]),
                           has_channel_llr=False,
                           evo_consts=np.ones(8), deg=4)
            _run(child.prog_c2v, vm2, "c2v")
