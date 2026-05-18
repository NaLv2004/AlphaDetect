// validator.hpp — V2C / C2V validators mirroring pushgp/validators.py (rev 2).
//
// Rev 2 design (post peer-#13/#18 / OMS-cfg2 incident):
//   * 6*deg + 6 input probes per cfg: per-position +/-d, +/-5d, sign-flip,
//     zero, plus global shifts and scales.
//   * L_v dependence is probed at both magnitudes / both signs (v2c only).
//   * 5 structured permutations: slot 0 = cyclic shift by 1 (always moves
//     every position), slot 1 = reverse, remainder = random shuffles.
//   * DEFAULT_NUM_EVO_PANELS = 0 — random evo sampling falsely rejects
//     legitimate programs whose constants are tuned (e.g. OMS uses
//     EvoConst1=1e6 as a min sentinel).
//   * Independent rngs for V2C / C2V (no coupling) when invoked from
//     validate_genome_with_rng.
//
// VM run policy:
//   * baseline fault / non-finite top -> reject.
//   * dependence probes: a faulted probe is *not* a rejection — it just
//     doesn't count toward "changed".  At least one probe must show
//     |out - base| >= EPS_DEPENDENCY.
//   * permutation probes: a faulted probe rejects ("faulty under perm").
//
// Two flavours per side are exposed:
//   * validate_*_with_panels   — fully deterministic; caller supplies the
//     ConfigPanels (used by the cross-language parity tests so Python and
//     C++ can be driven by the SAME panel and must agree).
//   * validate_*_random        — internal-RNG variant for production
//     seeding (uses std::mt19937_64; distribution differs from numpy's
//     PCG64, which is fine because validity is a statistical question).
#pragma once

#include "common.hpp"
#include "instruction.hpp"
#include "vm.hpp"
#include "vm_state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace pushgp_cpp {

// -------------------------------------------------- panel data ------------

// Sequence of permutations of one incoming vector.
using PermList = std::vector<FVec>;
// Labelled input-probe: (label, perturbed_incoming).
using InputProbe  = std::pair<std::string, FVec>;
using InputProbes = std::vector<InputProbe>;

struct ConfigPanel {
    double     L_v = 0.0;        // ignored for c2v
    FVec       incoming;
    InputProbes input_probes;   // 6*deg + 6 entries
    PermList   permutations;    // num_permutations entries
};

// -------------------------------------------------- stack seeding ---------

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

inline void configure_vm(VM& vm, const FVec& incoming, double L_v, bool has_llr,
                         int deg, int iter_idx,
                         const std::array<double, N_EVO_CONSTS>& evo) {
    vm.state.reset_stacks();
    vm.state.ctx_has_channel_llr = has_llr;
    vm.state.ctx_channel_llr     = L_v;
    vm.state.ctx_incoming        = incoming;
    vm.state.ctx_deg             = deg;
    vm.state.ctx_iter            = iter_idx;
    vm.state.ctx_max_iter        = 25;
    vm.state.ctx_noise_var       = 1.0;
    vm.state.ctx_edge_index      = 0;
    vm.state.ctx_code_rate       = 0.5;
    vm.state.ctx_evo_constants   = evo;
}

// Run `prog`; returns true & writes finite top to `out`, else false
// (fault, empty stack, or non-finite top).
inline bool run_side(VM& vm, const Program& prog, bool v2c, double& out) {
    if (v2c) seed_v2c_stacks(vm);
    else     seed_c2v_stacks(vm);
    vm.execute_block(prog);
    if (vm.state.fault) return false;
    const double* top = vm.state.floats.peek();
    if (!top || !std::isfinite(*top)) return false;
    out = *top;
    return true;
}

// ---- Backwards-compat shims (used by cpp_dce/bp_decoder.hpp and tests) ----
inline void configure_vm_v2c(VM& vm, const FVec& incoming, double L_v, int deg,
                             int iter_idx,
                             const std::array<double, N_EVO_CONSTS>& evo) {
    configure_vm(vm, incoming, L_v, /*has_llr=*/true, deg, iter_idx, evo);
}
inline void configure_vm_c2v(VM& vm, const FVec& incoming, int deg, int iter_idx,
                             const std::array<double, N_EVO_CONSTS>& evo) {
    configure_vm(vm, incoming, 0.0, /*has_llr=*/false, deg, iter_idx, evo);
}
inline bool run_v2c(VM& vm, const Program& prog, double& out) {
    return run_side(vm, prog, /*v2c=*/true, out);
}
inline bool run_c2v(VM& vm, const Program& prog, double& out) {
    return run_side(vm, prog, /*v2c=*/false, out);
}

// -------------------------------------------------- probe construction ---

// Mirror pushgp.validators._build_input_probes — 6*deg + 6 probes.
//
// Layout (in this order):
//   2*n per-position +/-d           (s_label=d,  signs +/-)
//   2*n per-position +/-5d          (s_label=5d, signs +/-)
//   n   per-position sign-flip
//   n   per-position zero
//   4   global shifts (+d, +5d, -d, -5d)  — order matches Python loop
//   2   global scales (*0.5, *2.0)
inline InputProbes build_input_probes(const FVec& incoming, double delta) {
    const int n = static_cast<int>(incoming.size());
    InputProbes probes;
    probes.reserve(static_cast<size_t>(6 * n + 6));
    const double scales[2]   = {1.0, 5.0};
    const char*  scale_lbl[] = {"d", "5d"};
    for (int si = 0; si < 2; ++si) {
        const double s = scales[si];
        for (int sg = 0; sg < 2; ++sg) {
            const double sign = sg == 0 ? +1.0 : -1.0;
            const char*  sig_lbl = sg == 0 ? "+" : "-";
            for (int i = 0; i < n; ++i) {
                FVec p = incoming;
                p[static_cast<size_t>(i)] += sign * s * delta;
                std::string lbl = "pos" + std::to_string(i) + sig_lbl + scale_lbl[si];
                probes.emplace_back(std::move(lbl), std::move(p));
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        FVec p = incoming;
        p[static_cast<size_t>(i)] = -p[static_cast<size_t>(i)];
        probes.emplace_back("pos" + std::to_string(i) + "_flip", std::move(p));
    }
    for (int i = 0; i < n; ++i) {
        FVec p = incoming;
        p[static_cast<size_t>(i)] = 0.0;
        probes.emplace_back("pos" + std::to_string(i) + "_zero", std::move(p));
    }
    // global shifts:  +d, +5d, -d, -5d  (Python: for sign in (+,-): for s in (d,5d))
    for (int sg = 0; sg < 2; ++sg) {
        const double sign = sg == 0 ? +1.0 : -1.0;
        const char*  sig_lbl = sg == 0 ? "+" : "-";
        for (int si = 0; si < 2; ++si) {
            const double s = scales[si];
            FVec p = incoming;
            for (auto& x : p) x += sign * s * delta;
            std::string lbl = std::string("glob") + sig_lbl + scale_lbl[si];
            probes.emplace_back(std::move(lbl), std::move(p));
        }
    }
    {
        FVec p = incoming; for (auto& x : p) x *= 0.5;
        probes.emplace_back("scale0.5", std::move(p));
    }
    {
        FVec p = incoming; for (auto& x : p) x *= 2.0;
        probes.emplace_back("scale2.0", std::move(p));
    }
    return probes;
}

// Mirror pushgp.validators._structured_perms.
inline PermList build_structured_perms(const FVec& incoming, int num_permutations,
                                       std::mt19937_64& rng) {
    PermList perms;
    const int n = static_cast<int>(incoming.size());
    if (num_permutations > 0 && n >= 2) {
        FVec shifted(static_cast<size_t>(n));
        for (int i = 0; i < n - 1; ++i) shifted[static_cast<size_t>(i)] = incoming[static_cast<size_t>(i + 1)];
        shifted[static_cast<size_t>(n - 1)] = incoming[0];
        perms.push_back(std::move(shifted));
    }
    if (num_permutations > 1 && n >= 2) {
        perms.emplace_back(incoming.rbegin(), incoming.rend());
    }
    while (static_cast<int>(perms.size()) < num_permutations) {
        FVec p = incoming;
        std::shuffle(p.begin(), p.end(), rng);
        perms.push_back(std::move(p));
    }
    return perms;
}

// -------------------------------------------------- result type ----------

struct ValidationResult {
    bool ok = false;
    std::string reason;
};

// -------------------------------------------------- core validators ------

// Both sides take an evo-panels list (mirroring Python's evo_panels);
// pass {evo} for the default num_evo_panels=0 behaviour.
inline ValidationResult validate_v2c_with_panels(
    const Program& prog,
    const std::vector<ConfigPanel>& panels,
    const std::vector<std::array<double, N_EVO_CONSTS>>& evo_panels,
    int deg = DEFAULT_DEG,
    double delta = DEFAULT_PERTURB_DELTA)
{
    VM vm;
    const double scales[2] = {1.0, 5.0};
    for (size_t cfg = 0; cfg < panels.size(); ++cfg) {
        const ConfigPanel& P = panels[cfg];
        for (size_t evo_idx = 0; evo_idx < evo_panels.size(); ++evo_idx) {
            const auto& evo = evo_panels[evo_idx];
            const std::string tag = "cfg" + std::to_string(cfg)
                                  + "/evo" + std::to_string(evo_idx);

            // baseline
            configure_vm(vm, P.incoming, P.L_v, /*has_llr=*/true,
                         deg, static_cast<int>(cfg), evo);
            double base;
            if (!run_side(vm, prog, /*v2c=*/true, base))
                return {false, "v2c " + tag + ": faulty / non-finite baseline"};

            // 1. dependence on L_v (2 magnitudes, both signs)
            bool l_changed = false;
            for (int sg = 0; sg < 2 && !l_changed; ++sg) {
                const double sign = sg == 0 ? +1.0 : -1.0;
                for (int si = 0; si < 2 && !l_changed; ++si) {
                    configure_vm(vm, P.incoming, P.L_v + sign * scales[si] * delta,
                                 /*has_llr=*/true, deg, static_cast<int>(cfg), evo);
                    double out2;
                    if (!run_side(vm, prog, /*v2c=*/true, out2)) continue;
                    if (std::fabs(out2 - base) >= EPS_DEPENDENCY) l_changed = true;
                }
            }
            if (!l_changed)
                return {false, "v2c " + tag + ": output independent of L_v"};

            // 2. dependence on incoming — ANY probe must change output
            bool inc_changed = false;
            for (const auto& pr : P.input_probes) {
                configure_vm(vm, pr.second, P.L_v, /*has_llr=*/true,
                             deg, static_cast<int>(cfg), evo);
                double out3;
                if (!run_side(vm, prog, /*v2c=*/true, out3)) continue;
                if (std::fabs(out3 - base) >= EPS_DEPENDENCY) { inc_changed = true; break; }
            }
            if (!inc_changed)
                return {false, "v2c " + tag + ": output independent of incoming"};

            // 3. permutation invariance — faulted perm => reject
            for (const auto& perm : P.permutations) {
                configure_vm(vm, perm, P.L_v, /*has_llr=*/true,
                             deg, static_cast<int>(cfg), evo);
                double out4;
                if (!run_side(vm, prog, /*v2c=*/true, out4))
                    return {false, "v2c " + tag + ": faulty under permutation"};
                if (std::fabs(out4 - base) > EPS_INVARIANCE)
                    return {false, "v2c " + tag + ": not permutation-invariant"};
            }
        }
    }
    return {true, "ok"};
}

inline ValidationResult validate_c2v_with_panels(
    const Program& prog,
    const std::vector<ConfigPanel>& panels,
    const std::vector<std::array<double, N_EVO_CONSTS>>& evo_panels,
    int deg = DEFAULT_DEG,
    double delta = DEFAULT_PERTURB_DELTA)
{
    (void)delta;
    VM vm;
    for (size_t cfg = 0; cfg < panels.size(); ++cfg) {
        const ConfigPanel& P = panels[cfg];
        for (size_t evo_idx = 0; evo_idx < evo_panels.size(); ++evo_idx) {
            const auto& evo = evo_panels[evo_idx];
            const std::string tag = "cfg" + std::to_string(cfg)
                                  + "/evo" + std::to_string(evo_idx);

            configure_vm(vm, P.incoming, 0.0, /*has_llr=*/false,
                         deg, static_cast<int>(cfg), evo);
            double base;
            if (!run_side(vm, prog, /*v2c=*/false, base))
                return {false, "c2v " + tag + ": faulty / non-finite baseline"};

            bool inc_changed = false;
            for (const auto& pr : P.input_probes) {
                configure_vm(vm, pr.second, 0.0, /*has_llr=*/false,
                             deg, static_cast<int>(cfg), evo);
                double out2;
                if (!run_side(vm, prog, /*v2c=*/false, out2)) continue;
                if (std::fabs(out2 - base) >= EPS_DEPENDENCY) { inc_changed = true; break; }
            }
            if (!inc_changed)
                return {false, "c2v " + tag + ": output independent of incoming"};

            for (const auto& perm : P.permutations) {
                configure_vm(vm, perm, 0.0, /*has_llr=*/false,
                             deg, static_cast<int>(cfg), evo);
                double out3;
                if (!run_side(vm, prog, /*v2c=*/false, out3))
                    return {false, "c2v " + tag + ": faulty under permutation"};
                if (std::fabs(out3 - base) > EPS_INVARIANCE)
                    return {false, "c2v " + tag + ": not permutation-invariant"};
            }
        }
    }
    return {true, "ok"};
}

// -------------------------------------------------- random / internal-RNG -

// Build panels with std::mt19937_64.  L_v ~ U(-2,2), incoming ~ U(-3,3),
// then derive deterministic input probes and structured permutations.
inline std::vector<ConfigPanel> make_panels_internal(
    std::mt19937_64& rng, int deg, int num_configs, int num_permutations,
    bool v2c, double delta)
{
    std::uniform_real_distribution<double> uL(L_V_SAMPLE_LO, L_V_SAMPLE_HI);
    std::uniform_real_distribution<double> uI(INCOMING_SAMPLE_LO, INCOMING_SAMPLE_HI);
    std::vector<ConfigPanel> panels;
    panels.reserve(static_cast<size_t>(num_configs));
    for (int cfg = 0; cfg < num_configs; ++cfg) {
        ConfigPanel P;
        P.L_v = v2c ? uL(rng) : 0.0;
        P.incoming.resize(static_cast<size_t>(deg - 1));
        for (auto& x : P.incoming) x = uI(rng);
        P.input_probes = build_input_probes(P.incoming, delta);
        P.permutations = build_structured_perms(P.incoming, num_permutations, rng);
        panels.push_back(std::move(P));
    }
    return panels;
}

// Sample evo panels: genome's own + `num_extra` random panels with
// log10(c) ~ U(VAL_EVO_LOG_LO, VAL_EVO_LOG_HI).
inline std::vector<std::array<double, N_EVO_CONSTS>> sample_evo_panels(
    const std::array<double, N_EVO_CONSTS>& genome_evo,
    int num_extra, std::mt19937_64& rng)
{
    std::vector<std::array<double, N_EVO_CONSTS>> out;
    out.reserve(static_cast<size_t>(1 + std::max(0, num_extra)));
    out.push_back(genome_evo);
    std::uniform_real_distribution<double> uL(VAL_EVO_LOG_LO, VAL_EVO_LOG_HI);
    for (int p = 0; p < num_extra; ++p) {
        std::array<double, N_EVO_CONSTS> panel{};
        for (int i = 0; i < N_EVO_CONSTS; ++i) panel[static_cast<size_t>(i)] = std::pow(10.0, uL(rng));
        out.push_back(panel);
    }
    return out;
}

inline ValidationResult validate_v2c_random(
    const Program& prog,
    const std::array<double, N_EVO_CONSTS>& evo,
    int deg, int num_configs, int num_permutations, int num_evo_panels,
    uint64_t seed, double delta = DEFAULT_PERTURB_DELTA)
{
    std::mt19937_64 rng(seed);
    auto panels     = make_panels_internal(rng, deg, num_configs, num_permutations,
                                           /*v2c=*/true, delta);
    auto evo_panels = sample_evo_panels(evo, num_evo_panels, rng);
    return validate_v2c_with_panels(prog, panels, evo_panels, deg, delta);
}

inline ValidationResult validate_c2v_random(
    const Program& prog,
    const std::array<double, N_EVO_CONSTS>& evo,
    int deg, int num_configs, int num_permutations, int num_evo_panels,
    uint64_t seed, double delta = DEFAULT_PERTURB_DELTA)
{
    std::mt19937_64 rng(seed);
    auto panels     = make_panels_internal(rng, deg, num_configs, num_permutations,
                                           /*v2c=*/false, delta);
    auto evo_panels = sample_evo_panels(evo, num_evo_panels, rng);
    return validate_c2v_with_panels(prog, panels, evo_panels, deg, delta);
}

// ---- Single-evo overloads (back-compat for existing bindings) ----
inline ValidationResult validate_v2c_with_panels(
    const Program& prog,
    const std::vector<ConfigPanel>& panels,
    const std::array<double, N_EVO_CONSTS>& evo,
    int deg)
{
    std::vector<std::array<double, N_EVO_CONSTS>> evo_panels{evo};
    return validate_v2c_with_panels(prog, panels, evo_panels, deg, DEFAULT_PERTURB_DELTA);
}
inline ValidationResult validate_c2v_with_panels(
    const Program& prog,
    const std::vector<ConfigPanel>& panels,
    const std::array<double, N_EVO_CONSTS>& evo,
    int deg)
{
    std::vector<std::array<double, N_EVO_CONSTS>> evo_panels{evo};
    return validate_c2v_with_panels(prog, panels, evo_panels, deg, DEFAULT_PERTURB_DELTA);
}
inline ValidationResult validate_v2c_random(
    const Program& prog,
    const std::array<double, N_EVO_CONSTS>& evo,
    int deg, int num_configs, int num_permutations, uint64_t seed)
{
    return validate_v2c_random(prog, evo, deg, num_configs, num_permutations,
                               DEFAULT_NUM_EVO_PANELS, seed);
}
inline ValidationResult validate_c2v_random(
    const Program& prog,
    const std::array<double, N_EVO_CONSTS>& evo,
    int deg, int num_configs, int num_permutations, uint64_t seed)
{
    return validate_c2v_random(prog, evo, deg, num_configs, num_permutations,
                               DEFAULT_NUM_EVO_PANELS, seed);
}

}  // namespace pushgp_cpp
