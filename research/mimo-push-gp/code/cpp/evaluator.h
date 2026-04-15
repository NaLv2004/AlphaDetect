/*
 * evaluator.h — C++ Push VM + Stack Decoder for MIMO Algorithm Discovery
 *
 * Compile (MSVC):
 *   cl.exe /EHsc /O2 /openmp /std:c++17 evaluator.cpp /LD /Fe:evaluator.dll
 *
 * Python usage via ctypes (see cpp_bridge.py).
 */
#pragma once

#include <cstdint>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

// Instruction opcodes — MUST match Python Instruction.name ordering
// Keep in sync with cpp_bridge.py OPCODE_MAP
enum Op : uint16_t {
    // Stack manipulation (17)
    OP_FLOAT_POP = 0, OP_FLOAT_DUP, OP_FLOAT_SWAP, OP_FLOAT_ROT,
    OP_INT_POP, OP_INT_DUP, OP_INT_SWAP,
    OP_BOOL_POP, OP_BOOL_DUP,
    OP_VEC_POP, OP_VEC_DUP, OP_VEC_SWAP,
    OP_MAT_POP, OP_MAT_DUP,
    OP_NODE_POP, OP_NODE_DUP, OP_NODE_SWAP,
    // Float arithmetic (13)
    OP_FLOAT_ADD, OP_FLOAT_SUB, OP_FLOAT_MUL, OP_FLOAT_DIV,
    OP_FLOAT_ABS, OP_FLOAT_NEG, OP_FLOAT_SQRT, OP_FLOAT_SQUARE,
    OP_FLOAT_MIN, OP_FLOAT_MAX, OP_FLOAT_EXP, OP_FLOAT_LOG, OP_FLOAT_TANH,
    // Comparisons (4)
    OP_FLOAT_LT, OP_FLOAT_GT, OP_INT_LT, OP_INT_GT,
    // Int arithmetic (4)
    OP_INT_ADD, OP_INT_SUB, OP_INT_INC, OP_INT_DEC,
    // Bool logic (3)
    OP_BOOL_AND, OP_BOOL_OR, OP_BOOL_NOT,
    // Tensor ops (11)
    OP_MAT_VECMUL, OP_VEC_ADD, OP_VEC_SUB, OP_VEC_DOT, OP_VEC_NORM2,
    OP_VEC_SCALE, OP_VEC_ELEMENTAT, OP_MAT_ELEMENTAT, OP_MAT_ROW,
    OP_VEC_LEN, OP_MAT_ROWS,
    // Peek-based access (6)
    OP_MAT_PEEKAT, OP_VEC_PEEKAT,
    OP_MAT_PEEKATIM, OP_VEC_PEEKATIM,
    OP_VEC_SECONDPEEKAT, OP_VEC_SECONDPEEKATIM,
    // High-level primitives (2)
    OP_VEC_GETRESIDUE, OP_FLOAT_GETMMSELB,
    // Node memory (7)
    OP_NODE_READMEM, OP_NODE_WRITEMEM,
    OP_NODE_GETCUMDIST, OP_NODE_GETLOCALDIST,
    OP_NODE_GETSYMRE, OP_NODE_GETSYMIM, OP_NODE_GETLAYER,
    // Graph navigation (6)
    OP_GRAPH_GETROOT, OP_NODE_GETPARENT,
    OP_NODE_NUMCHILDREN, OP_NODE_CHILDAT,
    OP_GRAPH_NODECOUNT, OP_GRAPH_FRONTIERCOUNT,
    // BP-essential (3)
    OP_NODE_GETSCORE, OP_NODE_SETSCORE, OP_NODE_ISEXPANDED,
    // Constants (11)
    OP_FLOAT_CONST0, OP_FLOAT_CONST1, OP_FLOAT_CONSTHALF,
    OP_FLOAT_CONSTNEG1, OP_FLOAT_CONST2, OP_FLOAT_CONST0_1,
    OP_INT_CONST0, OP_INT_CONST1, OP_INT_CONST2,
    OP_BOOL_TRUE, OP_BOOL_FALSE,
    // Environment (2)
    OP_FLOAT_GETNOISEVAR, OP_INT_GETNUMSYMBOLS,
    // Type conversion (2)
    OP_FLOAT_FROMINT, OP_INT_FROMFLOAT,

    // Control flow (8)
    OP_EXEC_IF, OP_EXEC_WHILE, OP_EXEC_DOTIMES,
    OP_NODE_FOREACHCHILD, OP_NODE_FOREACHSIBLING, OP_NODE_FOREACHANCESTOR,
    OP_EXEC_FOREACHSYMBOL, OP_EXEC_MINOVERSYMBOLS,

    // Block delimiters
    OP_BLOCK_START, OP_BLOCK_END,

    OP_COUNT  // sentinel
};

// ---- C API ----
extern "C" {

// Create evaluator context
// constellation: interleaved [re0, im0, re1, im1, ...] (2*M doubles)
EXPORT void* eval_create(int Nt, int Nr, int M,
                         const double* constellation,
                         int max_nodes, int flops_max, int step_max);

EXPORT void eval_destroy(void* ctx);

// Evaluate a single program on a single MIMO sample
// H, y: interleaved real/imag (2*Nr*Nt, 2*Nr)
// x_true: interleaved real/imag (2*Nt)
// program: flat opcode array [op0, op1, ..., BLOCK_START, ..., BLOCK_END, ...]
// Returns BER (0..1)
EXPORT double eval_one(void* ctx,
                       const int* program, int prog_len,
                       const double* H, const double* y,
                       const double* x_true,
                       double noise_var,
                       double* flops_out);

// Evaluate one program on a dataset of samples, return average BER
// H_all/y_all/x_true_all/noise_vars: contiguous arrays for n_samples
// Parallelized with OpenMP
EXPORT double eval_dataset(void* ctx,
                           const int* program, int prog_len,
                           int n_samples,
                           const double* H_all,
                           const double* y_all,
                           const double* x_true_all,
                           const double* noise_vars,
                           double* avg_flops_out);

// Evaluate multiple programs on a dataset (population batch)
// programs: array of n_programs pointers to opcode arrays
// prog_lengths: array of n_programs lengths
// ber_out, flops_out: output arrays of n_programs
EXPORT void eval_batch(void* ctx,
                       int n_programs,
                       const int** programs,
                       const int* prog_lengths,
                       int n_samples,
                       const double* H_all,
                       const double* y_all,
                       const double* x_true_all,
                       const double* noise_vars,
                       double* ber_out,
                       double* flops_out);

}  // extern "C"
