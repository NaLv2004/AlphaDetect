// validator.hpp — V2C / C2V validators that mirror pushgp/validators.py.
//
// Two flavours per side:
//   * validate_*_with_panel: fully deterministic — caller supplies the
//     L_v values, incoming vectors, perturb indices and permutations.
//     Used by the equivalence tests so Python + C++ can be driven by
//     the SAME panel and must agree on pass/fail.
//   * validate_*_random:     internal RNG (std::mt19937_64) for
//     production seeding.  Statistical distribution matches Python's
//     PCG64 closely enough that pass-rate is unchanged in expectation.
#pragma once

#include "common.hpp"
#include "instruction.hpp"
#include "vm.hpp"
#include "vm_state.hpp"

#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace pushgp_cpp {

// One validator panel: per-config draws (L_v, incoming, perturb idxs, perms).
struct ConfigPanel {
    double L_v = 0.0;            // ignored for c2v
    FVec   incoming;
    // Indices to try (up to 4) for the single-element perturbation step.
    // Mirrors:   idx = (cfg*7 + tries*3 + rng.integers(0,n)) % n
    std::vector<int> perturb_indices;
    // Pre-computed permutations of `incoming` (num_permutations of them).
    std::vector<FVec> permutations;
};

// Helper to seed VM stacks (mirrors validators.py:_seed_v2c_stacks/_seed_c2v_stacks).
inline void seed_v2c_stacks(VM& vm) {
    vm.state.floats.push(vm.state.ctx_channel_llr);
    vm.state.ints.push(vm.state.ctx_edge_index);
    vm.state.ints.push(vm.state.ctx_deg);
    vm.state.ints.push(vm.state.ctx_iter);
    vm.state.ints.push(vm.state.ctx_max_iter);
    vm.state.fvecs.push(vm.state.ctx_incoming);
}
inline void seed_c2v_stacks(VM& vm) {
    vm.state.ctx_has_channel_llr = false;
    vm.state.ints.push(vm.state.ctx_edge_index);
    vm.state.ints.push(vm.state.ctx_deg);
    vm.state.ints.push(vm.state.ctx_iter);
    vm.state.ints.push(vm.state.ctx_max_iter);
    vm.state.fvecs.push(vm.state.ctx_incoming);
}

// Per-call VM constructor.  Caller fills ctx_* fields.
inline void configure_vm_v2c(VM& vm, const FVec& incoming, double L_v, int deg,
                             int iter_idx, const std::array<double, N_EVO_CONSTS>& evo) {
    vm.state.reset_stacks();
    vm.state.ctx_has_channel_llr = true;
    vm.state.ctx_channel_llr = L_v;
    vm.state.ctx_incoming = incoming;
    vm.state.ctx_deg = deg;
    vm.state.ctx_iter = iter_idx;
    vm.state.ctx_max_iter = 25;
    vm.state.ctx_noise_var = 1.0;
    vm.state.ctx_edge_index = 0;
    vm.state.ctx_code_rate = 0.5;
    vm.state.ctx_evo_constants = evo;
}
inline void configure_vm_c2v(VM& vm, const FVec& incoming, int deg,
                             int iter_idx, const std::array<double, N_EVO_CONSTS>& evo) {
    vm.state.reset_stacks();
    vm.state.ctx_has_channel_llr = false;
    vm.state.ctx_channel_llr = 0.0;
    vm.state.ctx_incoming = incoming;
    vm.state.ctx_deg = deg;
    vm.state.ctx_iter = iter_idx;
    vm.state.ctx_max_iter = 25;
    vm.state.ctx_noise_var = 1.0;
    vm.state.ctx_edge_index = 0;
    vm.state.ctx_code_rate = 0.5;
    vm.state.ctx_evo_constants = evo;
}

inline bool run_v2c(VM& vm, const Program& prog, double& out) {
    seed_v2c_stacks(vm);
    vm.execute_block(prog);
    if (vm.state.fault) return false;
    const double* top = vm.state.floats.peek();
    if (!top || !std::isfinite(*top)) return false;
    out = *top;
    return true;
}
inline bool run_c2v(VM& vm, const Program& prog, double& out) {
    seed_c2v_stacks(vm);
    vm.execute_block(prog);
    if (vm.state.fault) return false;
    const double* top = vm.state.floats.peek();
    if (!top || !std::isfinite(*top)) return false;
    out = *top;
    return true;
}

struct ValidationResult {
    bool ok = false;
    std::string reason;
};

// =================================================================== V2C
inline ValidationResult validate_v2c_with_panels(
    const Program& prog,
    const std::vector<ConfigPanel>& panels,
    const std::array<double, N_EVO_CONSTS>& evo,
    int deg = DEFAULT_DEG)
{
    VM vm;
    for (size_t cfg = 0; cfg < panels.size(); ++cfg) {
        const auto& P = panels[cfg];
        // baseline
        configure_vm_v2c(vm, P.incoming, P.L_v, deg, static_cast<int>(cfg), evo);
        double base;
        if (!run_v2c(vm, prog, base))
            return {false, "v2c cfg" + std::to_string(cfg) + ": faulty / non-finite baseline"};

        // 1. dependence on L_v
        configure_vm_v2c(vm, P.incoming, P.L_v + DEFAULT_PERTURB_DELTA, deg, static_cast<int>(cfg), evo);
        double out2;
        if (!run_v2c(vm, prog, out2))
            return {false, "v2c cfg" + std::to_string(cfg) + ": faulty after L_v perturbation"};
        if (std::fabs(out2 - base) < EPS_DEPENDENCY)
            return {false, "v2c cfg" + std::to_string(cfg) + ": output independent of L_v"};

        // 2. dependence on incoming
        bool changed = false;
        size_t tries_max = std::min<size_t>(P.incoming.size(), 4);
        for (size_t tries = 0; tries < tries_max; ++tries) {
            int idx = P.perturb_indices[tries];
            FVec inc2 = P.incoming;
            inc2[static_cast<size_t>(idx)] += DEFAULT_PERTURB_DELTA;
            configure_vm_v2c(vm, inc2, P.L_v, deg, static_cast<int>(cfg), evo);
            double out3;
            if (!run_v2c(vm, prog, out3))
                return {false, "v2c cfg" + std::to_string(cfg) + ": faulty after incoming perturbation"};
            if (std::fabs(out3 - base) >= EPS_DEPENDENCY) { changed = true; break; }
        }
        if (!changed) {
            FVec inc3 = P.incoming;
            for (auto& x : inc3) x += DEFAULT_PERTURB_DELTA;
            configure_vm_v2c(vm, inc3, P.L_v, deg, static_cast<int>(cfg), evo);
            double out5;
            bool ok = run_v2c(vm, prog, out5);
            if (!ok || std::fabs(out5 - base) < EPS_DEPENDENCY)
                return {false, "v2c cfg" + std::to_string(cfg) + ": output independent of incoming"};
        }

        // 3. permutation invariance
        for (const auto& perm : P.permutations) {
            configure_vm_v2c(vm, perm, P.L_v, deg, static_cast<int>(cfg), evo);
            double out4;
            if (!run_v2c(vm, prog, out4))
                return {false, "v2c cfg" + std::to_string(cfg) + ": faulty under permutation"};
            if (std::fabs(out4 - base) > EPS_INVARIANCE)
                return {false, "v2c cfg" + std::to_string(cfg) + ": not permutation-invariant"};
        }
    }
    return {true, "ok"};
}

// =================================================================== C2V
inline ValidationResult validate_c2v_with_panels(
    const Program& prog,
    const std::vector<ConfigPanel>& panels,
    const std::array<double, N_EVO_CONSTS>& evo,
    int deg = DEFAULT_DEG)
{
    VM vm;
    for (size_t cfg = 0; cfg < panels.size(); ++cfg) {
        const auto& P = panels[cfg];
        configure_vm_c2v(vm, P.incoming, deg, static_cast<int>(cfg), evo);
        double base;
        if (!run_c2v(vm, prog, base))
            return {false, "c2v cfg" + std::to_string(cfg) + ": faulty / non-finite baseline"};

        bool changed = false;
        size_t tries_max = std::min<size_t>(P.incoming.size(), 4);
        for (size_t tries = 0; tries < tries_max; ++tries) {
            int idx = P.perturb_indices[tries];
            FVec inc2 = P.incoming;
            inc2[static_cast<size_t>(idx)] += DEFAULT_PERTURB_DELTA;
            configure_vm_c2v(vm, inc2, deg, static_cast<int>(cfg), evo);
            double out2;
            if (!run_c2v(vm, prog, out2))
                return {false, "c2v cfg" + std::to_string(cfg) + ": faulty after incoming perturbation"};
            if (std::fabs(out2 - base) >= EPS_DEPENDENCY) { changed = true; break; }
        }
        if (!changed) {
            FVec inc3 = P.incoming;
            for (auto& x : inc3) x += DEFAULT_PERTURB_DELTA;
            configure_vm_c2v(vm, inc3, deg, static_cast<int>(cfg), evo);
            double out4;
            bool ok = run_c2v(vm, prog, out4);
            if (!ok || std::fabs(out4 - base) < EPS_DEPENDENCY)
                return {false, "c2v cfg" + std::to_string(cfg) + ": output independent of incoming"};
        }

        for (const auto& perm : P.permutations) {
            configure_vm_c2v(vm, perm, deg, static_cast<int>(cfg), evo);
            double out3;
            if (!run_c2v(vm, prog, out3))
                return {false, "c2v cfg" + std::to_string(cfg) + ": faulty under permutation"};
            if (std::fabs(out3 - base) > EPS_INVARIANCE)
                return {false, "c2v cfg" + std::to_string(cfg) + ": not permutation-invariant"};
        }
    }
    return {true, "ok"};
}

// ---- Internal-RNG variant for production seeding ----
//
// Generates panels with std::mt19937_64; semantic equivalence is maintained
// (same checks, same thresholds) but actual panel values differ from numpy's
// PCG64 — that's fine because validity is a statistical question.
inline std::vector<ConfigPanel> make_panels_internal(
    std::mt19937_64& rng, int deg, int num_configs, int num_permutations,
    bool /*v2c*/)
{
    std::uniform_real_distribution<double> uL(-2.0, 2.0);
    std::uniform_real_distribution<double> uI(-3.0, 3.0);
    std::uniform_int_distribution<int> uIdx(0, deg - 2);

    std::vector<ConfigPanel> panels;
    panels.reserve(num_configs);
    for (int cfg = 0; cfg < num_configs; ++cfg) {
        ConfigPanel P;
        P.L_v = uL(rng);
        P.incoming.resize(deg - 1);
        for (auto& x : P.incoming) x = uI(rng);
        // Mirror python: idx = (cfg*7 + tries*3 + rng.integers(0,n)) % n
        int n = deg - 1;
        for (int tries = 0; tries < 4 && tries < n; ++tries) {
            int r = uIdx(rng);
            P.perturb_indices.push_back(((cfg * 7 + tries * 3 + r) % n + n) % n);
        }
        for (int p = 0; p < num_permutations; ++p) {
            FVec perm = P.incoming;
            std::shuffle(perm.begin(), perm.end(), rng);
            P.permutations.push_back(std::move(perm));
        }
        panels.push_back(std::move(P));
    }
    return panels;
}

inline ValidationResult validate_v2c_random(
    const Program& prog,
    const std::array<double, N_EVO_CONSTS>& evo,
    int deg, int num_configs, int num_permutations, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    auto panels = make_panels_internal(rng, deg, num_configs, num_permutations, /*v2c=*/true);
    return validate_v2c_with_panels(prog, panels, evo, deg);
}
inline ValidationResult validate_c2v_random(
    const Program& prog,
    const std::array<double, N_EVO_CONSTS>& evo,
    int deg, int num_configs, int num_permutations, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    auto panels = make_panels_internal(rng, deg, num_configs, num_permutations, /*v2c=*/false);
    return validate_c2v_with_panels(prog, panels, evo, deg);
}

}  // namespace pushgp_cpp
