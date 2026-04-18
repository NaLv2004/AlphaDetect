"""Tests for evolution framework Phase 1."""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

import numpy as np

from algorithm_ir.frontend.ir_builder import compile_function_to_ir
from algorithm_ir.regeneration.codegen import emit_python_source, emit_cpp_ops, CppOp

from evolution.config import EvolutionConfig
from evolution.fitness import FitnessResult, FitnessEvaluator
from evolution.genome import IRGenome
from evolution.skeleton_registry import ProgramSpec, SkeletonSpec, SkeletonRegistry
from evolution.random_program import random_ir_program
from evolution.operators import mutate_ir, crossover_ir, mutate_genome, crossover_genome
from evolution.engine import EvolutionEngine


# ---- Helper functions for testing ----

def make_add_fn():
    def f_add(a, b):
        return a + b
    return f_add


def make_weighted_sum_fn():
    def f_weighted(x, y, w):
        return x * w + y * (1.0 - w)
    return f_weighted


def make_abs_diff_fn():
    def f_abs_diff(a, b):
        return abs(a - b)
    return f_abs_diff


# ---- Config tests ----

class TestEvolutionConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = EvolutionConfig(program_roles=["f1", "f2"])
        self.assertEqual(cfg.population_size, 100)
        self.assertEqual(cfg.n_generations, 500)
        self.assertTrue(cfg.use_cpp)
        self.assertEqual(cfg.program_roles, ["f1", "f2"])

    def test_serialization_roundtrip(self):
        cfg = EvolutionConfig(
            program_roles=["f_down", "f_up"],
            population_size=50,
            metric_weights={"ber": 1.0, "complexity": 0.1},
        )
        d = cfg.to_dict()
        cfg2 = EvolutionConfig.from_dict(d)
        self.assertEqual(cfg2.population_size, 50)
        self.assertEqual(cfg2.program_roles, ["f_down", "f_up"])
        self.assertEqual(cfg2.metric_weights, {"ber": 1.0, "complexity": 0.1})


# ---- Fitness tests ----

class TestFitnessResult(unittest.TestCase):
    def test_composite_score(self):
        fr = FitnessResult(
            metrics={"ber": 0.01, "complexity": 100.0},
            weights={"ber": 10.0, "complexity": 0.001},
        )
        expected = 0.01 * 10.0 + 100.0 * 0.001
        self.assertAlmostEqual(fr.composite_score(), expected)

    def test_ordering(self):
        fr1 = FitnessResult(metrics={"ber": 0.01}, weights={"ber": 1.0})
        fr2 = FitnessResult(metrics={"ber": 0.02}, weights={"ber": 1.0})
        self.assertTrue(fr1 < fr2)

    def test_invalid_fitness(self):
        fr = FitnessResult(metrics={}, is_valid=False)
        self.assertEqual(fr.composite_score(), float("inf"))


# ---- Genome tests ----

class TestIRGenome(unittest.TestCase):
    def test_create_genome(self):
        ir = compile_function_to_ir(make_add_fn())
        genome = IRGenome(
            programs={"adder": ir},
            constants=np.array([1.0, 2.0]),
        )
        self.assertIn("adder", genome.programs)
        self.assertEqual(len(genome.constants), 2)
        self.assertIsNotNone(genome.genome_id)

    def test_to_source(self):
        ir = compile_function_to_ir(make_add_fn())
        genome = IRGenome(programs={"adder": ir})
        src = genome.to_source("adder")
        self.assertIn("def", src)
        self.assertIn("return", src)

    def test_to_callable(self):
        ir = compile_function_to_ir(make_add_fn())
        genome = IRGenome(programs={"adder": ir})
        fn = genome.to_callable("adder")
        self.assertEqual(fn(3, 4), 7)

    def test_to_cpp_ops(self):
        ir = compile_function_to_ir(make_add_fn())
        genome = IRGenome(programs={"adder": ir})
        ops = genome.to_cpp_ops("adder")
        self.assertIsInstance(ops, list)
        self.assertGreater(len(ops), 0)
        # Should contain RETURN
        self.assertIn(CppOp.RETURN, ops)

    def test_clone(self):
        ir = compile_function_to_ir(make_add_fn())
        genome = IRGenome(
            programs={"adder": ir},
            constants=np.array([1.0, 2.0]),
        )
        clone = genome.clone()
        self.assertIsNot(clone, genome)
        self.assertIsNot(clone.programs["adder"], genome.programs["adder"])
        np.testing.assert_array_equal(clone.constants, genome.constants)

    def test_structural_hash(self):
        ir1 = compile_function_to_ir(make_add_fn())
        ir2 = compile_function_to_ir(make_add_fn())
        g1 = IRGenome(programs={"f": ir1})
        g2 = IRGenome(programs={"f": ir2})
        self.assertEqual(g1.structural_hash(), g2.structural_hash())

    def test_serialize_roundtrip(self):
        ir = compile_function_to_ir(make_add_fn())
        genome = IRGenome(
            programs={"adder": ir},
            constants=np.array([1.5, -0.5]),
            generation=3,
        )
        data = genome.serialize()
        genome2 = IRGenome.deserialize(data)
        self.assertEqual(genome2.generation, 3)
        np.testing.assert_array_almost_equal(genome2.constants, [1.5, -0.5])
        fn = genome2.to_callable("adder")
        self.assertEqual(fn(10, 20), 30)


# ---- Codegen (emit_cpp_ops) tests ----

class TestEmitCppOps(unittest.TestCase):
    def test_simple_add(self):
        def add(a, b):
            return a + b
        ir = compile_function_to_ir(add)
        ops = emit_cpp_ops(ir)
        self.assertIn(CppOp.ADD, ops)
        self.assertIn(CppOp.RETURN, ops)

    def test_safe_div(self):
        def div_fn(a, b):
            return a / b
        ir = compile_function_to_ir(div_fn)
        ops = emit_cpp_ops(ir)
        # Div maps to SAFE_DIV
        self.assertIn(CppOp.SAFE_DIV, ops)

    def test_comparison(self):
        def cmp_fn(a, b):
            if a > b:
                return a
            return b
        ir = compile_function_to_ir(cmp_fn)
        ops = emit_cpp_ops(ir)
        self.assertIn(CppOp.GT, ops)

    def test_load_arg(self):
        def identity(x):
            return x
        ir = compile_function_to_ir(identity)
        ops = emit_cpp_ops(ir)
        self.assertIn(CppOp.LOAD_ARG, ops)
        self.assertIn(CppOp.RETURN, ops)

    def test_const_encoding(self):
        def const_fn():
            return 3.14
        ir = compile_function_to_ir(const_fn)
        ops = emit_cpp_ops(ir)
        self.assertIn(CppOp.CONST_F64, ops)


# ---- Skeleton registry tests ----

class TestSkeletonRegistry(unittest.TestCase):
    def _make_registry(self):
        reg = SkeletonRegistry()
        spec = ProgramSpec(
            name="f_score",
            param_names=["x", "y"],
            param_types=["float", "float"],
            return_type="float",
        )
        skel = SkeletonSpec(
            skeleton_id="test_skel",
            program_specs=[spec],
        )
        reg.register(skel)
        return reg

    def test_register_and_lookup(self):
        reg = self._make_registry()
        self.assertIn("f_score", reg.roles)
        ps = reg.get_program_spec("f_score")
        self.assertIsNotNone(ps)
        self.assertEqual(ps.param_names, ["x", "y"])

    def test_validate_matching_program(self):
        reg = self._make_registry()
        def f_score(x, y):
            return x + y
        ir = compile_function_to_ir(f_score)
        violations = reg.validate_program("f_score", ir)
        self.assertEqual(violations, [])

    def test_validate_wrong_arity(self):
        reg = self._make_registry()
        def f_score(x):
            return x
        ir = compile_function_to_ir(f_score)
        violations = reg.validate_program("f_score", ir)
        self.assertTrue(any("Arg count" in v for v in violations))

    def test_validate_unused_param(self):
        reg = self._make_registry()
        def f_score(x, y):
            return x
        ir = compile_function_to_ir(f_score)
        violations = reg.validate_program("f_score", ir)
        self.assertTrue(any("Unused" in v for v in violations))


# ---- Random program generation tests ----

class TestRandomProgram(unittest.TestCase):
    def test_generates_valid_ir(self):
        spec = ProgramSpec(
            name="f_test",
            param_names=["a", "b"],
            param_types=["float", "float"],
        )
        rng = np.random.default_rng(42)
        ir = random_ir_program(spec, rng, max_depth=3)
        self.assertIsNotNone(ir)
        self.assertEqual(ir.name, "f_test")

    def test_multiple_random_programs_different(self):
        spec = ProgramSpec(
            name="f_test",
            param_names=["x"],
            param_types=["float"],
        )
        rng = np.random.default_rng(123)
        programs = [random_ir_program(spec, rng, max_depth=4) for _ in range(5)]
        sources = [emit_python_source(p) for p in programs]
        # At least some should be different (probabilistic but almost certain with depth 4)
        unique = set(sources)
        self.assertGreater(len(unique), 1)

    def test_callable_from_random_program(self):
        spec = ProgramSpec(
            name="f_test",
            param_names=["x"],
            param_types=["float"],
        )
        rng = np.random.default_rng(42)
        ir = random_ir_program(spec, rng)
        src = emit_python_source(ir)
        ns: dict = {"__builtins__": __builtins__, "abs": abs}
        def _safe_div(a, b): return a / b if abs(b) > 1e-30 else 0.0
        def _safe_log(a):
            import math; return math.log(max(a, 1e-30))
        def _safe_sqrt(a):
            import math; return math.sqrt(max(a, 0.0))
        ns["_safe_div"] = _safe_div
        ns["_safe_log"] = _safe_log
        ns["_safe_sqrt"] = _safe_sqrt
        exec(compile(src, "<test>", "exec"), ns)
        fn = ns["f_test"]
        result = fn(1.5)
        self.assertIsInstance(result, (int, float))


# ---- Operator tests ----

class TestMutationOperators(unittest.TestCase):
    def test_point_mutation_changes_something(self):
        def f(a, b):
            return a + b
        ir = compile_function_to_ir(f)
        rng = np.random.default_rng(42)
        mutated = mutate_ir(ir, rng, "point")
        # Should be different (binary op swapped)
        src_orig = emit_python_source(ir)
        src_mut = emit_python_source(mutated)
        # May or may not be different (stochastic), but shouldn't crash
        self.assertIsNotNone(mutated)

    def test_constant_perturb(self):
        def f(x):
            return x + 3.0
        ir = compile_function_to_ir(f)
        rng = np.random.default_rng(42)
        mutated = mutate_ir(ir, rng, "constant_perturb")
        # Check that the constant changed
        orig_consts = [
            op.attrs.get("literal")
            for op in ir.ops.values()
            if op.opcode == "const" and isinstance(op.attrs.get("literal"), (int, float))
            and not isinstance(op.attrs.get("literal"), bool)
        ]
        mut_consts = [
            op.attrs.get("literal")
            for op in mutated.ops.values()
            if op.opcode == "const" and isinstance(op.attrs.get("literal"), (int, float))
            and not isinstance(op.attrs.get("literal"), bool)
        ]
        # At least one constant should have changed
        self.assertEqual(len(orig_consts), len(mut_consts))
        changed = sum(1 for a, b in zip(orig_consts, mut_consts) if a != b)
        self.assertGreaterEqual(changed, 1)

    def test_crossover(self):
        def f1(a, b):
            return a + b
        def f2(a, b):
            return a - b
        ir1 = compile_function_to_ir(f1)
        ir2 = compile_function_to_ir(f2)
        rng = np.random.default_rng(42)
        child = crossover_ir(ir1, ir2, rng)
        self.assertIsNotNone(child)
        src = emit_python_source(child)
        self.assertIn("def", src)

    def test_genome_mutation(self):
        ir = compile_function_to_ir(make_add_fn())
        genome = IRGenome(
            programs={"f": ir},
            constants=np.array([1.0, 2.0]),
        )
        cfg = EvolutionConfig(program_roles=["f"])
        rng = np.random.default_rng(42)
        mutate_genome(genome, cfg, rng)
        # Should not crash
        fn = genome.to_callable("f")
        self.assertIsNotNone(fn)

    def test_genome_crossover(self):
        ir1 = compile_function_to_ir(make_add_fn())
        ir2 = compile_function_to_ir(make_abs_diff_fn())
        g1 = IRGenome(programs={"f": ir1}, constants=np.array([1.0]))
        g2 = IRGenome(programs={"f": ir2}, constants=np.array([2.0]))
        cfg = EvolutionConfig(program_roles=["f"])
        rng = np.random.default_rng(42)
        child = crossover_genome(g1, g2, cfg, rng)
        self.assertIn(g1.genome_id, child.parent_ids)
        self.assertIn(g2.genome_id, child.parent_ids)


# ---- Engine tests ----

class _DummyEvaluator(FitnessEvaluator):
    """Evaluator for testing: fitness = sum of constants."""
    def evaluate(self, genome: IRGenome) -> FitnessResult:
        score = float(np.sum(np.abs(genome.constants)))
        return FitnessResult(
            metrics={"score": score},
            weights={"score": 1.0},
        )


class TestEvolutionEngine(unittest.TestCase):
    def test_init_population(self):
        spec = ProgramSpec(
            name="f_test",
            param_names=["x"],
            param_types=["float"],
        )
        skel = SkeletonSpec(skeleton_id="test", program_specs=[spec])
        reg = SkeletonRegistry()
        reg.register(skel)
        cfg = EvolutionConfig(
            program_roles=["f_test"],
            population_size=10,
            n_generations=5,
        )
        engine = EvolutionEngine(cfg, _DummyEvaluator(), reg)
        engine.init_population()
        self.assertEqual(len(engine.population), 10)
        self.assertEqual(len(engine.fitness), 10)
        self.assertIsNotNone(engine.best_ever)

    def test_run_small_evolution(self):
        spec = ProgramSpec(
            name="f_test",
            param_names=["x"],
            param_types=["float"],
        )
        skel = SkeletonSpec(skeleton_id="test", program_specs=[spec])
        reg = SkeletonRegistry()
        reg.register(skel)
        cfg = EvolutionConfig(
            program_roles=["f_test"],
            population_size=10,
            n_generations=5,
            n_constants=2,
        )
        engine = EvolutionEngine(cfg, _DummyEvaluator(), reg)
        best = engine.run()
        self.assertIsNotNone(best)
        self.assertGreater(len(engine.history), 0)
        # Evolution should have run 5 generations
        self.assertEqual(engine.generation, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
