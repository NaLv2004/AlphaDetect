"""
Python ↔ C++ Bridge for the Push VM + Stack Decoder.

Usage:
    from cpp_bridge import CppEvaluator

    evaluator = CppEvaluator(Nt=8, Nr=16, mod_order=16,
                              max_nodes=500, flops_max=2_000_000, step_max=1500)
    ber, flops = evaluator.evaluate_program(program, dataset)
"""

import ctypes
import os
import sys
import struct
import numpy as np
from typing import List, Tuple, Optional

from vm import Instruction, CONTROL_INSTRUCTIONS

# ---- Opcode mapping (must match evaluator.h enum Op) ----
OPCODE_MAP = {
    # Stack manipulation (17)
    'Float.Pop': 0, 'Float.Dup': 1, 'Float.Swap': 2, 'Float.Rot': 3,
    'Int.Pop': 4, 'Int.Dup': 5, 'Int.Swap': 6,
    'Bool.Pop': 7, 'Bool.Dup': 8,
    'Vector.Pop': 9, 'Vector.Dup': 10, 'Vector.Swap': 11,
    'Matrix.Pop': 12, 'Matrix.Dup': 13,
    'Node.Pop': 14, 'Node.Dup': 15, 'Node.Swap': 16,
    # Float arithmetic (13)
    'Float.Add': 17, 'Float.Sub': 18, 'Float.Mul': 19, 'Float.Div': 20,
    'Float.Abs': 21, 'Float.Neg': 22, 'Float.Sqrt': 23, 'Float.Square': 24,
    'Float.Min': 25, 'Float.Max': 26, 'Float.Exp': 27, 'Float.Log': 28,
    'Float.Tanh': 29,
    # Comparisons (4)
    'Float.LT': 30, 'Float.GT': 31, 'Int.LT': 32, 'Int.GT': 33,
    # Int arithmetic (4)
    'Int.Add': 34, 'Int.Sub': 35, 'Int.Inc': 36, 'Int.Dec': 37,
    # Bool logic (3)
    'Bool.And': 38, 'Bool.Or': 39, 'Bool.Not': 40,
    # Tensor ops (11)
    'Mat.VecMul': 41, 'Vec.Add': 42, 'Vec.Sub': 43, 'Vec.Dot': 44,
    'Vec.Norm2': 45, 'Vec.Scale': 46, 'Vec.ElementAt': 47,
    'Mat.ElementAt': 48, 'Mat.Row': 49, 'Vec.Len': 50, 'Mat.Rows': 51,
    # Peek access (6)
    'Mat.PeekAt': 52, 'Vec.PeekAt': 53,
    'Mat.PeekAtIm': 54, 'Vec.PeekAtIm': 55,
    'Vec.SecondPeekAt': 56, 'Vec.SecondPeekAtIm': 57,
    # High-level (2)
    'Vec.GetResidue': 58, 'Float.GetMMSELB': 59,
    # Node memory (7)
    'Node.ReadMem': 60, 'Node.WriteMem': 61,
    'Node.GetCumDist': 62, 'Node.GetLocalDist': 63,
    'Node.GetSymRe': 64, 'Node.GetSymIm': 65, 'Node.GetLayer': 66,
    # Graph navigation (6)
    'Graph.GetRoot': 67, 'Node.GetParent': 68,
    'Node.NumChildren': 69, 'Node.ChildAt': 70,
    'Graph.NodeCount': 71, 'Graph.FrontierCount': 72,
    # BP-essential (3)
    'Node.GetScore': 73, 'Node.SetScore': 74, 'Node.IsExpanded': 75,
    # Constants (11)
    'Float.Const0': 76, 'Float.Const1': 77, 'Float.ConstHalf': 78,
    'Float.ConstNeg1': 79, 'Float.Const2': 80, 'Float.Const0_1': 81,
    'Int.Const0': 82, 'Int.Const1': 83, 'Int.Const2': 84,
    'Bool.True': 85, 'Bool.False': 86,
    # Environment (2)
    'Float.GetNoiseVar': 87, 'Int.GetNumSymbols': 88,
    # Type conversion (2)
    'Float.FromInt': 89, 'Int.FromFloat': 90,
    # Control flow (8)
    'Exec.If': 91, 'Exec.While': 92, 'Exec.DoTimes': 93,
    'Node.ForEachChild': 94, 'Node.ForEachSibling': 95,
    'Node.ForEachAncestor': 96,
    'Exec.ForEachSymbol': 97, 'Exec.MinOverSymbols': 98,
    # Block delimiters
    'BLOCK_START': 99, 'BLOCK_END': 100,
    # BP message-passing opcodes
    'Node.GetMUp': 101, 'Node.SetMUp': 102,
    'Node.GetMDown': 103, 'Node.SetMDown': 104,
    # Evolved constant: pushes an inline double encoded as 2 ints
    'Float.PushImmediate': 105,
    # Float.Inv (1/x)
    'Float.Inv': 106,
    # MinReduce over children (A* heuristic support)
    'Node.ForEachChildMin': 107,
}

OP_BLOCK_START = OPCODE_MAP['BLOCK_START']
OP_BLOCK_END = OPCODE_MAP['BLOCK_END']


def encode_program(program: List[Instruction], evolved_constants=None) -> List[int]:
    """Convert a Python Instruction list to a flat opcode array for C++.
    
    evolved_constants: array of actual constant values (10^log_const).
        When an EvoConst instruction is encountered, its value is encoded
        inline as OP_FLOAT_PUSHIMMEDIATE + 2 int32s (little-endian double).
    """
    ops = []
    for ins in program:
        # Handle evolved constants: encode as PushImmediate + inline double
        if ins.name.startswith('Float.EvoConst'):
            idx = int(ins.name[-1])
            if evolved_constants is not None and idx < len(evolved_constants):
                val = float(evolved_constants[idx])
            else:
                val = 1.0
            ops.append(OPCODE_MAP['Float.PushImmediate'])
            raw = struct.pack('<d', val)
            lo, hi = struct.unpack('<ii', raw)
            ops.append(lo)
            ops.append(hi)
            continue
        op = OPCODE_MAP.get(ins.name)
        if op is None:
            continue  # skip unknown instructions
        ops.append(op)
        if ins.name in CONTROL_INSTRUCTIONS:
            if ins.name == 'Exec.If':
                # Then block
                ops.append(OP_BLOCK_START)
                ops.extend(encode_program(ins.code_block))
                ops.append(OP_BLOCK_END)
                # Else block
                ops.append(OP_BLOCK_START)
                ops.extend(encode_program(ins.code_block2))
                ops.append(OP_BLOCK_END)
            else:
                # Single block
                ops.append(OP_BLOCK_START)
                ops.extend(encode_program(ins.code_block))
                ops.append(OP_BLOCK_END)
    return ops


def _interleave_complex(arr: np.ndarray) -> np.ndarray:
    """Interleave real and imaginary parts: [re0, im0, re1, im1, ...]"""
    flat = arr.ravel()
    out = np.empty(2 * len(flat), dtype=np.float64)
    out[0::2] = flat.real
    out[1::2] = flat.imag
    return out


class CppEvaluator:
    """High-performance evaluator using the C++ DLL."""

    def __init__(self, Nt=8, Nr=16, mod_order=16,
                 max_nodes=500, flops_max=2_000_000, step_max=1500,
                 dll_path: Optional[str] = None):
        # Load DLL
        if dll_path is None:
            dll_path = os.path.join(os.path.dirname(__file__),
                                    'cpp', 'evaluator.dll')
        if not os.path.exists(dll_path):
            raise FileNotFoundError(
                f"C++ evaluator DLL not found at {dll_path}. "
                f"Build it first with: cl.exe /EHsc /O2 /openmp /std:c++17 "
                f"evaluator.cpp /LD /Fe:evaluator.dll")

        self.lib = ctypes.CDLL(dll_path)
        self._setup_prototypes()

        # Constellation
        self.Nt, self.Nr = Nt, Nr
        if mod_order == 16:
            levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
            constellation = np.array([r + 1j * i for r in levels for i in levels])
        elif mod_order == 4:
            s = 1.0 / np.sqrt(2)
            constellation = np.array([s+1j*s, s-1j*s, -s+1j*s, -s-1j*s])
        else:
            raise ValueError(f"Unsupported mod_order={mod_order}")

        self.M = len(constellation)
        cons_il = _interleave_complex(constellation)

        self.ctx = self.lib.eval_create(
            Nt, Nr, self.M,
            cons_il.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            max_nodes, flops_max, step_max)

    def _setup_prototypes(self):
        lib = self.lib
        lib.eval_create.restype = ctypes.c_void_p
        lib.eval_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int, ctypes.c_int]

        lib.eval_destroy.argtypes = [ctypes.c_void_p]

        lib.eval_one.restype = ctypes.c_double
        lib.eval_one.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double)]

        lib.eval_dataset.restype = ctypes.c_double
        lib.eval_dataset.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)]

        lib.eval_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)]

    def __del__(self):
        if hasattr(self, 'ctx') and hasattr(self, 'lib'):
            self.lib.eval_destroy(self.ctx)

    def prepare_dataset(self, dataset):
        """Convert a list of (H, x_true, y, noise_var) to flat arrays."""
        n = len(dataset)
        H_all = np.empty(n * 2 * self.Nr * self.Nt, dtype=np.float64)
        y_all = np.empty(n * 2 * self.Nr, dtype=np.float64)
        x_all = np.empty(n * 2 * self.Nt, dtype=np.float64)
        nv_all = np.empty(n, dtype=np.float64)

        for i, (H, x_true, y, nv) in enumerate(dataset):
            H_il = _interleave_complex(H)
            y_il = _interleave_complex(y)
            x_il = _interleave_complex(x_true)
            H_all[i * len(H_il):(i+1) * len(H_il)] = H_il
            y_all[i * len(y_il):(i+1) * len(y_il)] = y_il
            x_all[i * len(x_il):(i+1) * len(x_il)] = x_il
            nv_all[i] = float(nv)

        return H_all, y_all, x_all, nv_all

    def evaluate_program(self, program: List[Instruction],
                         dataset) -> Tuple[float, float]:
        """Evaluate one program on a dataset. Returns (avg_ber, avg_flops)."""
        ops = encode_program(program)
        ops_arr = (ctypes.c_int * len(ops))(*ops)

        H_all, y_all, x_all, nv_all = self.prepare_dataset(dataset)
        n = len(dataset)

        avg_flops = ctypes.c_double(0.0)
        ber = self.lib.eval_dataset(
            self.ctx,
            ops_arr, len(ops),
            n,
            H_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nv_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(avg_flops))

        return float(ber), float(avg_flops.value)

    def evaluate_batch(self, programs: List[List[Instruction]],
                       dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of programs. Returns (ber_array, flops_array)."""
        n_progs = len(programs)
        H_all, y_all, x_all, nv_all = self.prepare_dataset(dataset)
        n = len(dataset)

        # Encode all programs
        encoded = [encode_program(p) for p in programs]
        prog_arrays = []
        prog_ptrs = (ctypes.POINTER(ctypes.c_int) * n_progs)()
        prog_lens = (ctypes.c_int * n_progs)()
        for i, ops in enumerate(encoded):
            arr = (ctypes.c_int * len(ops))(*ops)
            prog_arrays.append(arr)  # prevent GC
            prog_ptrs[i] = arr
            prog_lens[i] = len(ops)

        ber_out = np.empty(n_progs, dtype=np.float64)
        flops_out = np.empty(n_progs, dtype=np.float64)

        self.lib.eval_batch(
            self.ctx, n_progs,
            prog_ptrs, prog_lens,
            n,
            H_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nv_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ber_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            flops_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        return ber_out, flops_out


class CppBPEvaluator:
    """High-performance BP evaluator using the C++ DLL (evaluator_bp.dll)."""

    def __init__(self, Nt=8, Nr=16, mod_order=16,
                 max_nodes=500, flops_max=2_000_000, step_max=1500,
                 max_bp_iters=3, dll_path=None):
        if dll_path is None:
            dll_path = os.path.join(os.path.dirname(__file__),
                                    'cpp', 'evaluator_bp.dll')
        if not os.path.exists(dll_path):
            raise FileNotFoundError(
                f"BP evaluator DLL not found at {dll_path}. "
                f"Build with: cl.exe /EHsc /O2 /openmp /std:c++17 "
                f"evaluator_bp.cpp /LD /Fe:evaluator_bp.dll")

        self.lib = ctypes.CDLL(dll_path)
        self._setup_prototypes()

        self.Nt, self.Nr = Nt, Nr
        if mod_order == 16:
            levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
            constellation = np.array([r + 1j * i for r in levels for i in levels])
        elif mod_order == 4:
            s = 1.0 / np.sqrt(2)
            constellation = np.array([s+1j*s, s-1j*s, -s+1j*s, -s-1j*s])
        else:
            raise ValueError(f"Unsupported mod_order={mod_order}")

        self.M = len(constellation)
        cons_il = _interleave_complex(constellation)

        self.ctx = self.lib.bp_eval_create(
            Nt, Nr, self.M,
            cons_il.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            max_nodes, flops_max, step_max, max_bp_iters)

    def _setup_prototypes(self):
        lib = self.lib
        lib.bp_eval_create.restype = ctypes.c_void_p
        lib.bp_eval_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

        lib.bp_eval_destroy.argtypes = [ctypes.c_void_p]

        lib.bp_eval_one.restype = ctypes.c_double
        lib.bp_eval_one.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # prog_down
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # prog_up
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # prog_belief
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # prog_halt
            ctypes.POINTER(ctypes.c_double),  # H
            ctypes.POINTER(ctypes.c_double),  # y
            ctypes.POINTER(ctypes.c_double),  # x_true
            ctypes.c_double,                  # noise_var
            ctypes.POINTER(ctypes.c_double),  # flops_out
            ctypes.POINTER(ctypes.c_int)]     # bp_calls_out

        lib.bp_eval_dataset.restype = ctypes.c_double
        lib.bp_eval_dataset.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),  # avg_flops_out
            ctypes.POINTER(ctypes.c_int),     # total_faults_out
            ctypes.POINTER(ctypes.c_double)]  # avg_bp_calls_out

        lib.bp_eval_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,                     # n_genomes
            ctypes.POINTER(ctypes.c_int),     # prog_all (flat)
            ctypes.POINTER(ctypes.c_int),     # prog_offsets
            ctypes.POINTER(ctypes.c_int),     # prog_lengths
            ctypes.c_int,                     # n_samples
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),  # ber_out
            ctypes.POINTER(ctypes.c_double),  # flops_out
            ctypes.POINTER(ctypes.c_int),     # faults_out
            ctypes.POINTER(ctypes.c_double)]  # bp_calls_out

        lib.bp_eval_baselines.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,                     # n_samples
            ctypes.POINTER(ctypes.c_double),  # H_all
            ctypes.POINTER(ctypes.c_double),  # y_all
            ctypes.POINTER(ctypes.c_double),  # x_true_all
            ctypes.POINTER(ctypes.c_double),  # noise_vars
            ctypes.POINTER(ctypes.c_double),  # ber_lmmse_out
            ctypes.POINTER(ctypes.c_double),  # ber_kb16_out
            ctypes.POINTER(ctypes.c_double)]  # ber_kb32_out

        lib.bp_eval_mmselb_stack.restype = ctypes.c_double
        lib.bp_eval_mmselb_stack.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,                     # n_samples
            ctypes.POINTER(ctypes.c_double),  # H_all
            ctypes.POINTER(ctypes.c_double),  # y_all
            ctypes.POINTER(ctypes.c_double),  # x_true_all
            ctypes.POINTER(ctypes.c_double),  # noise_vars
            ctypes.c_int]                     # override_max_nodes

        lib.bp_eval_multi_nodes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # prog_down
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # prog_up
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # prog_belief
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # prog_halt
            ctypes.c_int,                     # n_samples
            ctypes.POINTER(ctypes.c_double),  # H_all
            ctypes.POINTER(ctypes.c_double),  # y_all
            ctypes.POINTER(ctypes.c_double),  # x_true_all
            ctypes.POINTER(ctypes.c_double),  # noise_vars
            ctypes.POINTER(ctypes.c_int),     # node_limits
            ctypes.c_int,                     # n_limits
            ctypes.POINTER(ctypes.c_double)]  # ber_out

        lib.bp_eval_mmselb_multi_nodes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,                     # n_samples
            ctypes.POINTER(ctypes.c_double),  # H_all
            ctypes.POINTER(ctypes.c_double),  # y_all
            ctypes.POINTER(ctypes.c_double),  # x_true_all
            ctypes.POINTER(ctypes.c_double),  # noise_vars
            ctypes.POINTER(ctypes.c_int),     # node_limits
            ctypes.c_int,                     # n_limits
            ctypes.POINTER(ctypes.c_double)]  # ber_out

    def __del__(self):
        if hasattr(self, 'ctx') and hasattr(self, 'lib'):
            self.lib.bp_eval_destroy(self.ctx)

    def prepare_dataset(self, dataset):
        """Convert list of (H, x_true, y, noise_var) to flat arrays."""
        n = len(dataset)
        H_all = np.empty(n * 2 * self.Nr * self.Nt, dtype=np.float64)
        y_all = np.empty(n * 2 * self.Nr, dtype=np.float64)
        x_all = np.empty(n * 2 * self.Nt, dtype=np.float64)
        nv_all = np.empty(n, dtype=np.float64)
        for i, (H, x_true, y, nv) in enumerate(dataset):
            H_il = _interleave_complex(H)
            y_il = _interleave_complex(y)
            x_il = _interleave_complex(x_true)
            H_all[i * len(H_il):(i+1) * len(H_il)] = H_il
            y_all[i * len(y_il):(i+1) * len(y_il)] = y_il
            x_all[i * len(x_il):(i+1) * len(x_il)] = x_il
            nv_all[i] = float(nv)
        return H_all, y_all, x_all, nv_all

    def _encode_genome(self, genome):
        """Encode a 4-program genome into 4 flat opcode arrays."""
        evo_c = getattr(genome, 'evo_constants', None)
        progs = [genome.prog_down, genome.prog_up,
                 genome.prog_belief, genome.prog_halt]
        encoded = [encode_program(p, evolved_constants=evo_c) for p in progs]
        return encoded

    def evaluate_genome(self, genome, dataset):
        """Evaluate a single BP genome. Returns (avg_ber, avg_flops, faults, avg_bp_calls)."""
        encoded = self._encode_genome(genome)
        arrs = []
        ptrs = []
        lens = []
        for ops in encoded:
            arr = (ctypes.c_int * max(1, len(ops)))(*ops)
            arrs.append(arr)
            ptrs.append(ctypes.cast(arr, ctypes.POINTER(ctypes.c_int)))
            lens.append(len(ops))

        H_all, y_all, x_all, nv_all = self.prepare_dataset(dataset)
        n = len(dataset)

        avg_flops = ctypes.c_double(0.0)
        total_faults = ctypes.c_int(0)
        avg_bp_calls = ctypes.c_double(0.0)

        ber = self.lib.bp_eval_dataset(
            self.ctx,
            ptrs[0], lens[0],  # down
            ptrs[1], lens[1],  # up
            ptrs[2], lens[2],  # belief
            ptrs[3], lens[3],  # halt
            n,
            H_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nv_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(avg_flops),
            ctypes.byref(total_faults),
            ctypes.byref(avg_bp_calls))

        return float(ber), float(avg_flops.value), int(total_faults.value), float(avg_bp_calls.value)

    def evaluate_batch(self, genomes, dataset):
        """Evaluate a batch of BP genomes.
        Returns (ber_array, flops_array, faults_array, bp_calls_array)."""
        n_genomes = len(genomes)
        H_all, y_all, x_all, nv_all = self.prepare_dataset(dataset)
        n = len(dataset)

        # Encode all genomes: 4 programs each, concatenate into one flat array
        all_ops = []
        offsets = []
        lengths = []
        cur_offset = 0
        for g in genomes:
            encoded = self._encode_genome(g)
            for ops in encoded:
                offsets.append(cur_offset)
                lengths.append(len(ops))
                all_ops.extend(ops)
                cur_offset += len(ops)

        if len(all_ops) == 0:
            all_ops = [0]  # prevent empty array

        prog_all_arr = (ctypes.c_int * len(all_ops))(*all_ops)
        offsets_arr = (ctypes.c_int * len(offsets))(*offsets)
        lengths_arr = (ctypes.c_int * len(lengths))(*lengths)

        ber_out = np.empty(n_genomes, dtype=np.float64)
        flops_out = np.empty(n_genomes, dtype=np.float64)
        faults_out = np.empty(n_genomes, dtype=np.int32)
        bp_calls_out = np.empty(n_genomes, dtype=np.float64)

        self.lib.bp_eval_batch(
            self.ctx,
            n_genomes,
            ctypes.cast(prog_all_arr, ctypes.POINTER(ctypes.c_int)),
            ctypes.cast(offsets_arr, ctypes.POINTER(ctypes.c_int)),
            ctypes.cast(lengths_arr, ctypes.POINTER(ctypes.c_int)),
            n,
            H_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nv_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ber_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            flops_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            faults_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            bp_calls_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        return ber_out, flops_out, faults_out, bp_calls_out

    def evaluate_baselines(self, dataset):
        """Evaluate LMMSE, K-Best-16, K-Best-32 baselines on a dataset.
        Returns (avg_ber_lmmse, avg_ber_kb16, avg_ber_kb32)."""
        H_all, y_all, x_all, nv_all = self.prepare_dataset(dataset)
        n = len(dataset)

        ber_lmmse = ctypes.c_double(0.0)
        ber_kb16 = ctypes.c_double(0.0)
        ber_kb32 = ctypes.c_double(0.0)

        self.lib.bp_eval_baselines(
            self.ctx, n,
            H_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nv_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(ber_lmmse),
            ctypes.byref(ber_kb16),
            ctypes.byref(ber_kb32))

        return float(ber_lmmse.value), float(ber_kb16.value), float(ber_kb32.value)

    def evaluate_mmselb_stack(self, dataset, max_nodes=None):
        """Evaluate pure MMSE-LB stack decoder (no BP). Returns avg_ber."""
        H_all, y_all, x_all, nv_all = self.prepare_dataset(dataset)
        n = len(dataset)
        mn = max_nodes if max_nodes is not None else 0  # 0 = use ctx default
        return float(self.lib.bp_eval_mmselb_stack(
            self.ctx, n,
            H_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nv_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            mn))

    def evaluate_genome_multi_nodes(self, genome, dataset, node_limits):
        """Evaluate a genome at multiple node count limits.
        Returns array of BERs (one per node_limit)."""
        encoded = self._encode_genome(genome)
        arrs, ptrs, lens = [], [], []
        for ops in encoded:
            arr = (ctypes.c_int * max(1, len(ops)))(*ops)
            arrs.append(arr)
            ptrs.append(ctypes.cast(arr, ctypes.POINTER(ctypes.c_int)))
            lens.append(len(ops))

        H_all, y_all, x_all, nv_all = self.prepare_dataset(dataset)
        n = len(dataset)
        n_limits = len(node_limits)
        limits_arr = (ctypes.c_int * n_limits)(*node_limits)
        ber_out = np.empty(n_limits, dtype=np.float64)

        self.lib.bp_eval_multi_nodes(
            self.ctx,
            ptrs[0], lens[0], ptrs[1], lens[1],
            ptrs[2], lens[2], ptrs[3], lens[3],
            n,
            H_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nv_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(limits_arr, ctypes.POINTER(ctypes.c_int)),
            n_limits,
            ber_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return ber_out

    def evaluate_mmselb_multi_nodes(self, dataset, node_limits):
        """Evaluate MMSE-LB stack decoder at multiple node limits.
        Returns array of BERs (one per node_limit)."""
        H_all, y_all, x_all, nv_all = self.prepare_dataset(dataset)
        n = len(dataset)
        n_limits = len(node_limits)
        limits_arr = (ctypes.c_int * n_limits)(*node_limits)
        ber_out = np.empty(n_limits, dtype=np.float64)

        self.lib.bp_eval_mmselb_multi_nodes(
            self.ctx, n,
            H_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nv_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(limits_arr, ctypes.POINTER(ctypes.c_int)),
            n_limits,
            ber_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return ber_out
