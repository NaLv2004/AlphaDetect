// rpg.hpp — Stack-aware random program generator.
//
// At every position we sample only from the subset of `instr_set` whose
// stack preconditions (encoded in stack_effects.hpp) are satisfied by
// the current virtual stack state.  Existing weight scheme (control:
// 0.10*N/n_ctrl, stack-shuffle: 0.5, others: 1.0) is preserved but
// re-normalised over the feasible subset each step.
//
// Top-level c2v programs must end with at least one float on the stack
// (that float is the C2V message).  If the natural draw leaves the
// float stack empty, the generator appends a single feasible
// float-producing op as a tail repair.  If no such op exists in the
// instr_set, generation throws — this signals an inadequate filter,
// not a runtime fallback.
//
// Sub-block recursion uses VStack::child_for to seed the inner virtual
// stack (DoTimes/DoRange push the loop counter).  Sub-block effects on
// the parent virtual stack are treated as neutral (a deliberate
// approximation: the body's net pop/push depends on its random content).
#pragma once

#include "common.hpp"
#include "instruction.hpp"
#include "opcodes.hpp"
#include "stack_effects.hpp"

#include <array>
#include <random>
#include <stdexcept>
#include <vector>

namespace pushgp_cpp {

inline bool is_stack_op_(Op op) {
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

enum class OutputKind { None, V2C, C2V };

class RandomProgramGenerator {
   public:
    explicit RandomProgramGenerator(uint64_t seed, int max_recur_depth = 2)
        : rng_(seed), max_recur_depth_(max_recur_depth) {}

    // Generate a flat top-level program with length in [min_size, max_size]
    // using the V2C entry-point stack state.
    Program random_program_v2c(const std::vector<Op>& instr_set,
                               int min_size, int max_size) {
        std::uniform_int_distribution<int> usz(min_size, max_size);
        int n = usz(rng_);
        VStack vs = VStack::v2c_entry();
        return gen_(instr_set, n, /*depth=*/0, vs, OutputKind::V2C);
    }

    // Generate a top-level C2V program.  Tail-repaired to ensure ≥1
    // float remains on the stack at end (C2V output requirement).
    Program random_program_c2v(const std::vector<Op>& instr_set,
                               int min_size, int max_size) {
        std::uniform_int_distribution<int> usz(min_size, max_size);
        int n = usz(rng_);
        VStack vs = VStack::c2v_entry();
        return gen_(instr_set, n, /*depth=*/0, vs, OutputKind::C2V);
    }

    // Legacy entry point (used by tests/benchmarks).  Uses V2C entry stack.
    Program random_program(const std::vector<Op>& instr_set,
                           int min_size, int max_size) {
        return random_program_v2c(instr_set, min_size, max_size);
    }

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

    std::array<double, N_EVO_CONSTS> random_log_constants() {
        std::uniform_real_distribution<double> u(LOG_CONST_MIN, LOG_CONST_MAX);
        std::array<double, N_EVO_CONSTS> out{};
        for (auto& x : out) x = u(rng_);
        return out;
    }

   private:
    std::mt19937_64 rng_;
    int max_recur_depth_;

    // Per-op weight (un-normalised).  Same scheme as Python.
    static double op_weight_(Op op, int total_n, int n_ctrl) {
        if (op_is_control(op)) {
            int denom = std::max(1, n_ctrl);
            return 0.10 * static_cast<double>(total_n) / denom;
        }
        if (is_stack_op_(op)) return 0.5;
        return 1.0;
    }

    void feasible_(const std::vector<Op>& instr_set,
                   const VStack& vs, int depth,
                   std::vector<Op>& out_ops,
                   std::vector<double>& out_w) const {
        out_ops.clear();
        out_w.clear();
        const bool allow_ctrl = depth < max_recur_depth_;
        int n_ctrl = 0;
        for (Op op : instr_set) {
            if (!allow_ctrl && op_is_control(op)) continue;
            if (!vs.can_apply(op)) continue;
            out_ops.push_back(op);
            if (op_is_control(op)) ++n_ctrl;
        }
        const int n_total = static_cast<int>(out_ops.size());
        out_w.reserve(out_ops.size());
        double s = 0.0;
        for (Op op : out_ops) {
            double w = op_weight_(op, n_total, n_ctrl);
            out_w.push_back(w);
            s += w;
        }
        if (s > 0.0) for (double& v : out_w) v /= s;
    }

    Op sample_(const std::vector<Op>& names, const std::vector<double>& w) {
        std::discrete_distribution<int> d(w.begin(), w.end());
        return names[static_cast<size_t>(d(rng_))];
    }

    Program gen_(const std::vector<Op>& instr_set, int n, int depth,
                 VStack& vs, OutputKind out_kind) {
        Program out;
        out.reserve(static_cast<size_t>(n));
        std::vector<Op> cand_ops;
        std::vector<double> cand_w;

        for (int i = 0; i < n; ++i) {
            feasible_(instr_set, vs, depth, cand_ops, cand_w);
            if (cand_ops.empty()) {
                throw std::runtime_error(
                    "rpg: no feasible op for current virtual stack "
                    "(instr_set inadequate or recursion-depth filter too strict)");
            }
            Op op = sample_(cand_ops, cand_w);
            Instruction ins(op);
            if (op_is_control(op)) {
                std::uniform_int_distribution<int> usz(2, 6);
                int sz = usz(rng_);
                VStack child_vs = VStack::child_for(vs, op);
                ins.code_block = std::make_unique<Program>(
                    gen_(instr_set, sz, depth + 1, child_vs, OutputKind::None));
                if (op_has_two_blocks(op)) {
                    int sz2 = usz(rng_);
                    VStack child_vs2 = VStack::child_for(vs, op);
                    ins.code_block2 = std::make_unique<Program>(
                        gen_(instr_set, sz2, depth + 1, child_vs2, OutputKind::None));
                }
            }
            vs.apply(op);
            out.push_back(std::move(ins));
        }

        // C2V output constraint: ≥1 float remaining on top-level stack.
        if (out_kind == OutputKind::C2V && vs.f < 1) {
            std::vector<Op> producers;
            std::vector<double> pw;
            for (Op op : instr_set) {
                if (!vs.can_apply(op)) continue;
                if (op_is_control(op)) continue;
                const auto& e = effect_of(op);
                if (e.f_push - e.f_pop > 0) {
                    producers.push_back(op);
                    pw.push_back(1.0);
                }
            }
            if (producers.empty()) {
                throw std::runtime_error(
                    "rpg: c2v program ended with empty float stack and "
                    "instr_set has no feasible float-producing op for repair");
            }
            Op repair = sample_(producers, pw);
            Instruction ins(repair);
            vs.apply(repair);
            out.push_back(std::move(ins));
        }
        return out;
    }
};

}  // namespace pushgp_cpp
