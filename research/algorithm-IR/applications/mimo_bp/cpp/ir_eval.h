/*
 * ir_eval.h — Stack-based expression evaluator for Algorithm-IR opcodes.
 *
 * Evaluates flat opcode arrays emitted by emit_cpp_ops() in codegen.py.
 * Single double-precision stack, ~33 opcodes.
 *
 * Compile (MSVC):
 *   cl.exe /EHsc /O2 /openmp /std:c++17 /DBUILD_DLL bp_ir_decoder.cpp /LD /Fe:bp_ir_eval.dll
 */
#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>

// Must match CppOp in codegen.py
enum IROp : int {
    OP_CONST_F64    = 0,
    OP_LOAD_ARG     = 1,
    OP_ADD          = 2,
    OP_SUB          = 3,
    OP_MUL          = 4,
    OP_DIV          = 5,
    OP_SQRT         = 6,
    OP_ABS          = 7,
    OP_NEG          = 8,
    OP_EXP          = 9,
    OP_LOG          = 10,
    OP_TANH         = 11,
    OP_MIN          = 12,
    OP_MAX          = 13,
    OP_LT           = 14,
    OP_GT           = 15,
    OP_LE           = 16,
    OP_GE           = 17,
    OP_EQ           = 18,
    OP_IF_START     = 19,
    OP_ELSE         = 20,
    OP_ENDIF        = 21,
    OP_WHILE_START  = 22,
    OP_WHILE_END    = 23,
    OP_RETURN       = 24,
    OP_SAFE_DIV     = 25,
    OP_SAFE_LOG     = 26,
    OP_SAFE_SQRT    = 27,
    OP_NE           = 28,
    OP_NOT          = 29,
    OP_DUP          = 30,
    OP_POP          = 31,
    OP_NOP          = 32,
};

static constexpr int IR_STACK_SIZE = 64;
static constexpr int IR_MAX_ARGS  = 16;

/* Decode a float64 from two int32 words (little-endian). */
inline double ir_decode_f64(int lo, int hi) {
    int32_t parts[2] = { (int32_t)lo, (int32_t)hi };
    double val;
    std::memcpy(&val, parts, sizeof(double));
    return val;
}

/*
 * Evaluate an IR opcode program.
 *
 * prog:     flat opcode array (from emit_cpp_ops)
 * prog_len: number of ints in the array
 * args:     argument values (accessed via OP_LOAD_ARG)
 * n_args:   number of arguments
 *
 * Returns the top-of-stack after OP_RETURN (or 0.0 if empty).
 */
inline double ir_eval(const int* prog, int prog_len,
                      const double* args, int n_args) {
    double stack[IR_STACK_SIZE];
    int sp = 0;  // stack pointer (points to next free slot)

    int pc = 0;
    int step = 0;
    constexpr int MAX_STEPS = 10000;

    while (pc < prog_len && step < MAX_STEPS) {
        step++;
        int op = prog[pc++];

        switch (op) {
        case OP_CONST_F64: {
            if (pc + 1 >= prog_len) goto done;
            int lo = prog[pc++];
            int hi = prog[pc++];
            if (sp < IR_STACK_SIZE)
                stack[sp++] = ir_decode_f64(lo, hi);
            break;
        }
        case OP_LOAD_ARG: {
            if (pc >= prog_len) goto done;
            int idx = prog[pc++];
            double val = (idx >= 0 && idx < n_args) ? args[idx] : 0.0;
            if (sp < IR_STACK_SIZE) stack[sp++] = val;
            break;
        }
        case OP_ADD: {
            if (sp >= 2) { double b = stack[--sp]; stack[sp-1] += b; }
            break;
        }
        case OP_SUB: {
            if (sp >= 2) { double b = stack[--sp]; stack[sp-1] -= b; }
            break;
        }
        case OP_MUL: {
            if (sp >= 2) { double b = stack[--sp]; stack[sp-1] *= b; }
            break;
        }
        case OP_DIV: {
            if (sp >= 2) {
                double b = stack[--sp];
                stack[sp-1] = (std::fabs(b) > 1e-30) ? stack[sp-1] / b : 0.0;
            }
            break;
        }
        case OP_SAFE_DIV: {
            if (sp >= 2) {
                double b = stack[--sp];
                stack[sp-1] = (std::fabs(b) > 1e-30) ? stack[sp-1] / b : 0.0;
            }
            break;
        }
        case OP_SQRT: {
            if (sp >= 1) stack[sp-1] = std::sqrt(std::max(stack[sp-1], 0.0));
            break;
        }
        case OP_SAFE_SQRT: {
            if (sp >= 1) stack[sp-1] = std::sqrt(std::max(stack[sp-1], 0.0));
            break;
        }
        case OP_ABS: {
            if (sp >= 1) stack[sp-1] = std::fabs(stack[sp-1]);
            break;
        }
        case OP_NEG: {
            if (sp >= 1) stack[sp-1] = -stack[sp-1];
            break;
        }
        case OP_EXP: {
            if (sp >= 1) {
                double x = std::min(stack[sp-1], 500.0);
                stack[sp-1] = std::exp(x);
            }
            break;
        }
        case OP_LOG: {
            if (sp >= 1)
                stack[sp-1] = std::log(std::max(stack[sp-1], 1e-30));
            break;
        }
        case OP_SAFE_LOG: {
            if (sp >= 1)
                stack[sp-1] = std::log(std::max(stack[sp-1], 1e-30));
            break;
        }
        case OP_TANH: {
            if (sp >= 1) stack[sp-1] = std::tanh(stack[sp-1]);
            break;
        }
        case OP_MIN: {
            if (sp >= 2) { double b = stack[--sp]; stack[sp-1] = std::min(stack[sp-1], b); }
            break;
        }
        case OP_MAX: {
            if (sp >= 2) { double b = stack[--sp]; stack[sp-1] = std::max(stack[sp-1], b); }
            break;
        }
        case OP_LT: {
            if (sp >= 2) { double b = stack[--sp]; stack[sp-1] = (stack[sp-1] < b) ? 1.0 : 0.0; }
            break;
        }
        case OP_GT: {
            if (sp >= 2) { double b = stack[--sp]; stack[sp-1] = (stack[sp-1] > b) ? 1.0 : 0.0; }
            break;
        }
        case OP_LE: {
            if (sp >= 2) { double b = stack[--sp]; stack[sp-1] = (stack[sp-1] <= b) ? 1.0 : 0.0; }
            break;
        }
        case OP_GE: {
            if (sp >= 2) { double b = stack[--sp]; stack[sp-1] = (stack[sp-1] >= b) ? 1.0 : 0.0; }
            break;
        }
        case OP_EQ: {
            if (sp >= 2) { double b = stack[--sp]; stack[sp-1] = (std::fabs(stack[sp-1] - b) < 1e-12) ? 1.0 : 0.0; }
            break;
        }
        case OP_NE: {
            if (sp >= 2) { double b = stack[--sp]; stack[sp-1] = (std::fabs(stack[sp-1] - b) >= 1e-12) ? 1.0 : 0.0; }
            break;
        }
        case OP_NOT: {
            if (sp >= 1) stack[sp-1] = (stack[sp-1] == 0.0) ? 1.0 : 0.0;
            break;
        }
        case OP_IF_START: {
            // pop condition; if false, skip to matching ELSE or ENDIF
            if (sp >= 1) {
                double cond = stack[--sp];
                if (cond == 0.0) {
                    // Skip forward to matching ELSE or ENDIF
                    int depth = 1;
                    while (pc < prog_len && depth > 0) {
                        int skip_op = prog[pc++];
                        if (skip_op == OP_IF_START) depth++;
                        else if (skip_op == OP_ELSE && depth == 1) break;
                        else if (skip_op == OP_ENDIF) { depth--; if (depth == 0) break; }
                        else if (skip_op == OP_CONST_F64) pc += 2;
                        else if (skip_op == OP_LOAD_ARG) pc += 1;
                    }
                }
            }
            break;
        }
        case OP_ELSE: {
            // True branch done, skip to ENDIF
            int depth = 1;
            while (pc < prog_len && depth > 0) {
                int skip_op = prog[pc++];
                if (skip_op == OP_IF_START) depth++;
                else if (skip_op == OP_ENDIF) { depth--; }
                else if (skip_op == OP_CONST_F64) pc += 2;
                else if (skip_op == OP_LOAD_ARG) pc += 1;
            }
            break;
        }
        case OP_ENDIF:
            // No-op marker
            break;
        case OP_WHILE_START: {
            // pop condition; if false, skip to WHILE_END
            if (sp >= 1) {
                double cond = stack[--sp];
                if (cond == 0.0) {
                    int depth = 1;
                    while (pc < prog_len && depth > 0) {
                        int skip_op = prog[pc++];
                        if (skip_op == OP_WHILE_START) depth++;
                        else if (skip_op == OP_WHILE_END) depth--;
                        else if (skip_op == OP_CONST_F64) pc += 2;
                        else if (skip_op == OP_LOAD_ARG) pc += 1;
                    }
                }
            }
            break;
        }
        case OP_WHILE_END:
            // Jump back to start of while (not implemented in simple expr eval)
            break;
        case OP_RETURN:
            goto done;
        case OP_DUP: {
            if (sp >= 1 && sp < IR_STACK_SIZE) {
                stack[sp] = stack[sp-1];
                sp++;
            }
            break;
        }
        case OP_POP: {
            if (sp >= 1) sp--;
            break;
        }
        case OP_NOP:
        default:
            break;
        }

        // Guard against NaN/Inf propagation
        if (sp > 0) {
            double& top = stack[sp-1];
            if (std::isnan(top) || std::isinf(top)) top = 0.0;
        }
    }

done:
    return (sp > 0) ? stack[sp-1] : 0.0;
}
