// rpg.hpp — Random Program Generator mirroring pushgp/random_program.py.
//
// Uses std::mt19937_64.  Distribution matches Python qualitatively
// (same weighting scheme, same recursion depth limit, same block sizes)
// but the precise byte sequence will differ from numpy PCG64 — this is
// fine because the equivalence tests don't compare programs, only
// validator pass/fail and VM output bytes for the same input program.
#pragma once

#include "common.hpp"
#include "instruction.hpp"
#include "opcodes.hpp"

#include <array>
#include <random>
#include <unordered_set>
#include <vector>

namespace pushgp_cpp {

// Categorise each opcode for sampling-weight purposes.
inline bool is_stack_op_(Op op) {
    // Names ending with .Pop .Dup .Swap .Rot .Yank .Shove
    switch (op) {
    case Op::Float_Pop: case Op::Float_Dup: case Op::Float_Swap:
    case Op::Float_Rot: case Op::Float_Yank: case Op::Float_Shove:
    case Op::Int_Pop:   case Op::Int_Dup:   case Op::Int_Swap: case Op::Int_Rot:
    case Op::Bool_Pop:  case Op::Bool_Dup:  case Op::Bool_Swap:
    case Op::FVec_Pop:  case Op::FVec_Dup:  case Op::FVec_Swap: case Op::FVec_Rot:
    case Op::BVec_Pop:  case Op::BVec_Dup:  case Op::BVec_Swap:
    case Op::IVec_Pop:  case Op::IVec_Dup:  case Op::IVec_Swap:
        return true;
    default: return false;
    }
}

class RandomProgramGenerator {
   public:
    explicit RandomProgramGenerator(uint64_t seed, int max_recur_depth = 2)
        : rng_(seed), max_recur_depth_(max_recur_depth) {}

    // Generate a flat top-level program with length in [min_size, max_size].
    Program random_program(const std::vector<Op>& instr_set,
                           int min_size, int max_size) {
        std::uniform_int_distribution<int> usz(min_size, max_size);
        int n = usz(rng_);
        return gen_(instr_set, n, /*depth=*/0);
    }

    // V2C: full set.  C2V: drop Env_GetChannelLLR.
    static std::vector<Op> v2c_set() {
        std::vector<Op> v;
        for (int i = 0; i < OP_COUNT; ++i) v.push_back(static_cast<Op>(i));
        return v;
    }
    static std::vector<Op> c2v_set() {
        std::vector<Op> v;
        for (int i = 0; i < OP_COUNT; ++i) {
            Op op = static_cast<Op>(i);
            if (op == Op::Env_GetChannelLLR) continue;
            v.push_back(op);
        }
        return v;
    }

    // 8-element log-domain constants in [LOG_CONST_MIN, LOG_CONST_MAX].
    std::array<double, N_EVO_CONSTS> random_log_constants() {
        std::uniform_real_distribution<double> u(LOG_CONST_MIN, LOG_CONST_MAX);
        std::array<double, N_EVO_CONSTS> out{};
        for (auto& x : out) x = u(rng_);
        return out;
    }

   private:
    std::mt19937_64 rng_;
    int max_recur_depth_;

    // Compute weights for the candidate name list (mirrors _instr_weights in Python).
    static std::vector<double> weights_(const std::vector<Op>& names) {
        int n_ctrl = 0;
        for (Op op : names) if (op_is_control(op)) ++n_ctrl;
        std::vector<double> w(names.size(), 1.0);
        for (size_t i = 0; i < names.size(); ++i) {
            Op op = names[i];
            if (op_is_control(op)) {
                w[i] = 0.10 * static_cast<double>(names.size()) / std::max(1, n_ctrl);
            } else if (is_stack_op_(op)) {
                w[i] = 0.5;
            }
        }
        // Normalize.
        double s = 0.0;
        for (double v : w) s += v;
        if (s > 0) for (double& v : w) v /= s;
        return w;
    }

    Op sample_(const std::vector<Op>& names, const std::vector<double>& w) {
        std::discrete_distribution<int> d(w.begin(), w.end());
        return names[static_cast<size_t>(d(rng_))];
    }

    Program gen_(const std::vector<Op>& instr_set, int n, int depth) {
        // Filter to non-control if at max depth.
        const std::vector<Op>* names_ptr = &instr_set;
        std::vector<Op> filtered;
        if (depth >= max_recur_depth_) {
            for (Op op : instr_set) if (!op_is_control(op)) filtered.push_back(op);
            if (filtered.empty()) names_ptr = &instr_set;  // fallback
            else names_ptr = &filtered;
        }
        std::vector<double> w = weights_(*names_ptr);

        Program out;
        out.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            Op op = sample_(*names_ptr, w);
            Instruction ins(op);
            if (op_is_control(op)) {
                std::uniform_int_distribution<int> usz(2, 6);  // Python integers(2,7) → [2,6]
                int sz = usz(rng_);
                ins.code_block = std::make_unique<Program>(gen_(instr_set, sz, depth + 1));
                if (op_has_two_blocks(op)) {
                    int sz2 = usz(rng_);
                    ins.code_block2 = std::make_unique<Program>(gen_(instr_set, sz2, depth + 1));
                }
            }
            out.push_back(std::move(ins));
        }
        return out;
    }
};

}  // namespace pushgp_cpp
