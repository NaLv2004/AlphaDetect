"""Fitness evaluator for the MIMO BP detector evolution.

Connects the evolution engine to the C++ BP-IR decoder DLL.
"""
from __future__ import annotations

import sys
import pathlib
import time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolution.fitness import FitnessResult, FitnessEvaluator
from evolution.genome import IRGenome
from algorithm_ir.regeneration.codegen import emit_cpp_ops
from applications.mimo_bp.mimo_simulation import (
    generate_dataset, dataset_to_dict, get_constellation, MIMODataset,
)
from applications.mimo_bp.cpp_evaluator import CppBPIREvaluator


class MIMOBPFitnessEvaluator(FitnessEvaluator):
    """Evaluate evolved BP detector genomes via C++ DLL."""

    def __init__(
        self,
        Nt: int = 16,
        Nr: int = 16,
        mod_order: int = 16,
        snr_db: float = 24.0,
        n_train: int = 200,
        n_test: int = 100,
        max_nodes: int = 500,
        max_bp_iters: int = 5,
        seed: int = 42,
    ):
        self.Nt = Nt
        self.Nr = Nr
        self.mod_order = mod_order
        self.snr_db = snr_db
        self.n_train = n_train
        self.n_test = n_test
        self.max_nodes = max_nodes
        self.max_bp_iters = max_bp_iters
        self.seed = seed

        constellation = get_constellation(mod_order)
        self._cpp_eval = CppBPIREvaluator(
            Nt=Nt, Nr=Nr, mod_order=mod_order,
            constellation=constellation,
            max_nodes=max_nodes,
            max_bp_iters=max_bp_iters,
        )

        # Pre-generate datasets
        rng = np.random.default_rng(seed)
        self._train_ds = generate_dataset(n_train, Nt, Nr, mod_order, snr_db, rng)
        self._test_ds = generate_dataset(n_test, Nt, Nr, mod_order, snr_db, rng)
        self._train_dict = dataset_to_dict(self._train_ds)
        self._test_dict = dataset_to_dict(self._test_ds)

        # LMMSE baseline (computed once)
        self._lmmse_ber = self._cpp_eval.evaluate_lmmse(self._train_dict)

    def evaluate(self, genome: IRGenome) -> FitnessResult:
        """Evaluate a genome's fitness.

        Compiles all 4 programs to C++ opcodes and runs the BP decoder
        on the training dataset. Also evaluates on the test set for
        generalization gap measurement.

        Returns FitnessResult with metrics:
            - ber: bit error rate on training set
            - avg_flops: average nodes expanded per sample
            - gen_gap: |train_ber - test_ber|
            - lmmse_ber: LMMSE baseline BER
        """
        required_programs = {"f_down", "f_up", "f_belief", "h_halt"}
        if not required_programs.issubset(genome.programs.keys()):
            return FitnessResult(
                metrics={"ber": 1.0, "avg_flops": 0, "gen_gap": 0},
                is_valid=False,
                weights=self._default_weights(),
            )

        try:
            prog_down = emit_cpp_ops(genome.programs["f_down"])
            prog_up = emit_cpp_ops(genome.programs["f_up"])
            prog_belief = emit_cpp_ops(genome.programs["f_belief"])
            prog_halt = emit_cpp_ops(genome.programs["h_halt"])
        except Exception:
            return FitnessResult(
                metrics={"ber": 1.0, "avg_flops": 0, "gen_gap": 0},
                is_valid=False,
                weights=self._default_weights(),
            )

        try:
            train_ber, train_flops = self._cpp_eval.evaluate_genome(
                prog_down, prog_up, prog_belief, prog_halt,
                self._train_dict,
            )
            test_ber, _ = self._cpp_eval.evaluate_genome(
                prog_down, prog_up, prog_belief, prog_halt,
                self._test_dict,
            )
        except Exception:
            return FitnessResult(
                metrics={"ber": 1.0, "avg_flops": 0, "gen_gap": 0},
                is_valid=False,
                weights=self._default_weights(),
            )

        gen_gap = abs(train_ber - test_ber)

        return FitnessResult(
            metrics={
                "ber": train_ber,
                "avg_flops": train_flops,
                "gen_gap": gen_gap,
                "lmmse_ber": self._lmmse_ber,
            },
            is_valid=True,
            weights=self._default_weights(),
        )

    @staticmethod
    def _default_weights() -> dict[str, float]:
        return {
            "ber": 1.0,
            "avg_flops": 1e-6,
            "gen_gap": 0.3,
            "lmmse_ber": 0.0,  # informational only, not part of fitness
        }
