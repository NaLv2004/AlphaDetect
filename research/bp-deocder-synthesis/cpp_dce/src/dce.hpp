// dce.hpp — C++ port of pushgp.dce.behavioral_reduce_bp.
//
// Strict design contract: structurally identical reduction trajectory
// to the Python implementation in pushgp/dce.py:behavioral_reduce_bp.
// _walk_positions order, _remove_at semantics, and the
// reversed-list / first-match-wins / restart-pass loop must match
// byte-for-byte; only the BP inner loop is replaced by decode_bp_cpp
// (which Step-2 has proven bit-equal to ldpc_5g.decode_bp).
//
// We keep the Python position type structurally:
//   PosStep { idx, desc } where desc = 0 (terminal), 1 ("cb"), 2 ("cb2")
//   Position = std::vector<PosStep>; the LAST step in a position has
//   desc = 0 and identifies the instruction inside its parent
//   container that we want to delete.
#pragma once

#include "../../cpp_seeder/src/common.hpp"
#include "../../cpp_seeder/src/instruction.hpp"

#include "bp_decoder.hpp"
#include "parity_struct.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pushgp_cpp_dce {

using pushgp_cpp::Instruction;
using pushgp_cpp::N_EVO_CONSTS;
using pushgp_cpp::Program;

struct PosStep {
    int idx;
    int desc;   // 0 = terminal, 1 = code_block, 2 = code_block2
};
using Position = std::vector<PosStep>;

// Depth-first list of every instruction position, mirroring
// pushgp.dce._walk_positions exactly: top-level, then code_block of
// that instruction, then code_block2, then the next top-level.
inline void walk_positions_inplace(const Program& prog,
                                   Position& prefix,
                                   std::vector<Position>& out) {
    const int n = static_cast<int>(prog.size());
    for (int i = 0; i < n; ++i) {
        // emit prefix + (i, desc=0)
        prefix.push_back({i, 0});
        out.push_back(prefix);
        prefix.pop_back();

        const auto& ins = prog[i];
        if (ins.code_block) {
            prefix.push_back({i, 1});
            walk_positions_inplace(*ins.code_block, prefix, out);
            prefix.pop_back();
        }
        if (ins.code_block2) {
            prefix.push_back({i, 2});
            walk_positions_inplace(*ins.code_block2, prefix, out);
            prefix.pop_back();
        }
    }
}

inline std::vector<Position> walk_positions(const Program& prog) {
    std::vector<Position> out;
    Position prefix;
    walk_positions_inplace(prog, prefix, out);
    return out;
}

// Deep-copy then remove at position.  Throws on malformed position.
inline Program remove_at(const Program& prog, const Position& pos) {
    if (pos.empty()) throw std::runtime_error("empty position");
    Program out = prog;   // Instruction has deep copy ctor
    Program* container = &out;
    // Walk all but the last step.
    for (size_t i = 0; i + 1 < pos.size(); ++i) {
        const PosStep& s = pos[i];
        if (s.idx < 0 || s.idx >= static_cast<int>(container->size())) {
            throw std::runtime_error("remove_at: idx out of range");
        }
        Instruction& ins = (*container)[s.idx];
        if (s.desc == 1) {
            if (!ins.code_block) throw std::runtime_error("cb on instr w/o code_block");
            container = ins.code_block.get();
        } else if (s.desc == 2) {
            if (!ins.code_block2) throw std::runtime_error("cb2 on instr w/o code_block2");
            container = ins.code_block2.get();
        } else {
            throw std::runtime_error("remove_at: non-terminal step has desc=0");
        }
    }
    const PosStep& last = pos.back();
    if (last.desc != 0) throw std::runtime_error("remove_at: last step desc != 0");
    if (last.idx < 0 || last.idx >= static_cast<int>(container->size())) {
        throw std::runtime_error("remove_at: final idx out of range");
    }
    container->erase(container->begin() + last.idx);
    return out;
}

// 6-decimal equality used by behavioral_reduce_bp.
// Matches np.array_equal(np.round(x, d), np.round(y, d)) — IEEE round
// half-to-even via std::nearbyint with default FE_TONEAREST.
inline bool eq_at_decimals(const std::vector<double>& a,
                           const std::vector<double>& b,
                           int decimals) {
    if (a.size() != b.size()) return false;
    const double scale = std::pow(10.0, decimals);
    for (size_t i = 0; i < a.size(); ++i) {
        // np.round uses banker's rounding; std::nearbyint with the
        // default rounding mode (FE_TONEAREST) is also banker's
        // rounding, so the two produce identical results.
        double ra = std::nearbyint(a[i] * scale) / scale;
        double rb = std::nearbyint(b[i] * scale) / scale;
        // Compare bit-for-bit (np.array_equal: True only when bits equal,
        // also True for NaN==NaN per the docs — but post_llr should be
        // finite here; we still tolerate NaN-equality for safety).
        if (ra != rb) {
            // np.array_equal treats NaN==NaN as False unless equal_nan=True.
            // _eq_all in pushgp.dce uses the default array_equal, so NaN
            // never equals anything — return false (matches python).
            return false;
        }
    }
    return true;
}

// Run BP on every frame using (prog_v2c, prog_c2v).  Returns the list
// of post-LLR vectors.  If any frame raises in C++ we instead set the
// "ok" flag false (mirrors python's `try: ...; except: return None`).
struct DecodeAllOut {
    bool ok = false;
    std::vector<std::vector<double>> per_frame;
};
inline DecodeAllOut decode_all(
    const Program& prog_v2c, const Program& prog_c2v,
    const LiftedParityC& par,
    const std::vector<std::vector<double>>& rx_llrs,
    const std::array<double, N_EVO_CONSTS>& evo,
    int max_iter)
{
    DecodeAllOut out;
    out.per_frame.reserve(rx_llrs.size());
    for (const auto& rx : rx_llrs) {
        try {
            out.per_frame.push_back(
                decode_bp_cpp(rx, par, prog_v2c, prog_c2v, evo, max_iter));
        } catch (...) {
            out.ok = false;
            out.per_frame.clear();
            return out;
        }
    }
    out.ok = true;
    return out;
}

struct ReduceStats {
    int passes = 0;
    int fp_evals = 0;
    int size_before = 0;
    int size_after = 0;
    std::vector<Position> removed_positions;
};

// One-side behavioral reduce (mirrors pushgp.dce.behavioral_reduce_bp).
// `prog` is the program being reduced; `peer_prog` is the fixed other
// side.  `side_is_v2c=true` means prog is the v2c program.
inline Program behavioral_reduce_bp_cpp(
    const Program& prog,
    bool side_is_v2c,
    const Program& peer_prog,
    const std::array<double, N_EVO_CONSTS>& evo,
    const LiftedParityC& par,
    const std::vector<std::vector<double>>& rx_llrs,
    int max_iter = 8,
    int max_passes = 800,
    int max_decode_evals = -1,    // -1 = unlimited (Python None)
    int decimals = 6,
    ReduceStats* stats_out = nullptr)
{
    if (rx_llrs.empty()) throw std::runtime_error("rx_llrs empty");

    auto side_decode = [&](const Program& p) -> DecodeAllOut {
        const Program& v = side_is_v2c ? p : peer_prog;
        const Program& c = side_is_v2c ? peer_prog : p;
        return decode_all(v, c, par, rx_llrs, evo, max_iter);
    };

    ReduceStats stats;
    stats.size_before = pushgp_cpp::program_length(prog);
    stats.size_after  = stats.size_before;

    Program cur = prog;   // deep copy via Instruction ctor

    DecodeAllOut base = side_decode(cur);
    stats.fp_evals += 1;
    if (!base.ok) {
        if (stats_out) *stats_out = stats;
        return cur;
    }

    auto eq_all = [&](const DecodeAllOut& a, const DecodeAllOut& b) -> bool {
        if (!a.ok || !b.ok) return false;
        if (a.per_frame.size() != b.per_frame.size()) return false;
        for (size_t i = 0; i < a.per_frame.size(); ++i) {
            if (!eq_at_decimals(a.per_frame[i], b.per_frame[i], decimals)) {
                return false;
            }
        }
        return true;
    };

    for (int pass_idx = 0; pass_idx < max_passes; ++pass_idx) {
        stats.passes = pass_idx + 1;
        bool removed_in_pass = false;
        std::vector<Position> positions = walk_positions(cur);
        // reversed iteration
        for (auto it = positions.rbegin(); it != positions.rend(); ++it) {
            if (max_decode_evals >= 0 && stats.fp_evals >= max_decode_evals) break;
            Program cand;
            try {
                cand = remove_at(cur, *it);
            } catch (...) {
                continue;
            }
            DecodeAllOut cand_out = side_decode(cand);
            stats.fp_evals += 1;
            if (eq_all(cand_out, base)) {
                cur = std::move(cand);
                stats.removed_positions.push_back(*it);
                stats.size_after = pushgp_cpp::program_length(cur);
                removed_in_pass = true;
                break;
            }
        }
        if (!removed_in_pass) break;
        if (max_decode_evals >= 0 && stats.fp_evals >= max_decode_evals) break;
    }

    if (stats_out) *stats_out = std::move(stats);
    return cur;
}

}  // namespace pushgp_cpp_dce
