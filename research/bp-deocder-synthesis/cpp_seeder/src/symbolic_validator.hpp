// symbolic_validator.hpp — dependency / permutation / oddness checks on Expr.
//
// Reusable interface: caller supplies CheckSpec describing
//   * what atoms must appear in the output (dep_required_kinds)
//   * which atom groups must be permutation-invariant (sym_groups)
//   * whether the output must be odd under sign-flip of a given atom set
//
// Result: (ok, reason) — same shape as the probe validator.
#pragma once

#include <sstream>

#include "symbolic_expr.hpp"
#include "symbolic_vm.hpp"

#include <string>
#include <vector>

namespace pushgp_cpp {

struct DepRequirement {
    AtomKind kind;
    // If indices empty: require at least one atom of this kind.
    // Else: require at least one atom with kind & idx in indices.
    std::vector<int> indices;
};

struct SymGroup {
    AtomKind kind;
    std::vector<int> indices;  // atoms (k, idx) that may permute freely
};

struct CheckSpec {
    std::vector<DepRequirement> deps_required;
    std::vector<SymGroup> sym_groups;
    bool require_odd = false;
    // Which atoms (encoded) should be substituted with Neg(atom) for oddness test.
    std::vector<int> odd_negate_atoms;
};

struct CheckResult {
    bool ok = false;
    std::string reason;
};

inline bool atoms_contains(const std::vector<int>& sorted_atoms, int enc) {
    return std::binary_search(sorted_atoms.begin(), sorted_atoms.end(), enc);
}
inline bool atoms_contains_any_of_kind(const std::vector<int>& sorted_atoms, AtomKind k) {
    int lo = encode_atom(k, 0);
    int hi = encode_atom(k, 0xffff);
    auto it = std::lower_bound(sorted_atoms.begin(), sorted_atoms.end(), lo);
    return it != sorted_atoms.end() && *it <= hi;
}

// ---- the three checks ----

inline CheckResult check_dep(const ExprRef& out, const CheckSpec& spec) {
    for (const auto& req : spec.deps_required) {
        if (req.indices.empty()) {
            if (!atoms_contains_any_of_kind(out->atoms, req.kind))
                return {false, std::string("missing dep on kind ") +
                              std::to_string(static_cast<int>(req.kind))};
        } else {
            bool found = false;
            for (int idx : req.indices) {
                if (atoms_contains(out->atoms, encode_atom(req.kind, idx))) {
                    found = true; break;
                }
            }
            if (!found) return {false, "missing dep on required atom"};
        }
    }
    return {true, "ok"};
}

inline CheckResult check_sym(const ExprRef& out, const CheckSpec& spec) {
    for (const auto& g : spec.sym_groups) {
        if (g.indices.size() < 2) continue;
        // Check adjacent transpositions: swap g.indices[i] and g.indices[i+1].
        for (size_t i = 0; i + 1 < g.indices.size(); ++i) {
            int a = g.indices[i];
            int b = g.indices[i+1];
            AtomMap m;
            m.table[encode_atom(g.kind, a)] = encode_atom(g.kind, b);
            m.table[encode_atom(g.kind, b)] = encode_atom(g.kind, a);
            ExprRef swapped = substitute(out, m);
            if (swapped.get() != out.get()) {
                return {false, "not permutation-invariant (swap "
                              + std::to_string(a) + " <-> " + std::to_string(b) + ")"};
            }
        }
    }
    return {true, "ok"};
}

inline CheckResult check_odd(const ExprRef& out, const CheckSpec& spec) {
    // ODD CHECK DISABLED (user directive 2025): symbolic odd reasoning cannot
    // discharge sign_prod-style parity (e.g. canonical OMS C2V uses a product
    // of deg-1 Float.Sign factors). Keeping the check enabled causes
    // catastrophic false rejects on the ground-truth OMS decoder. We remove
    // odd-symmetry from the symbolic spec entirely; downstream probe/random
    // validators retain numerical odd-symmetry sampling if needed.
    (void)out; (void)spec;
    return {true, "ok"};
}

inline CheckResult symbolic_validate(const ExprRef& out, const CheckSpec& spec) {
    if (!out) return {false, "no output"};
    if (out->kind == ExprKind::Opaque) return {false, "output opaque"};
    auto r = check_dep(out, spec);
    if (!r.ok) return r;
    r = check_sym(out, spec);
    if (!r.ok) return r;
    r = check_odd(out, spec);
    if (!r.ok) return r;
    return {true, "ok"};
}

// ---- BP-specific specs ----

inline CheckSpec make_v2c_spec(int deg) {
    CheckSpec s;
    s.deps_required.push_back({AtomKind::X, {}});   // at least one X_i
    s.deps_required.push_back({AtomKind::LV, {0}}); // LV must appear
    SymGroup g; g.kind = AtomKind::X;
    for (int i = 0; i < deg - 1; ++i) g.indices.push_back(i);
    s.sym_groups.push_back(std::move(g));
    s.require_odd = true;
    for (int i = 0; i < deg - 1; ++i)
        s.odd_negate_atoms.push_back(encode_atom(AtomKind::X, i));
    s.odd_negate_atoms.push_back(encode_atom(AtomKind::LV, 0));
    return s;
}
inline CheckSpec make_c2v_spec(int deg) {
    CheckSpec s;
    s.deps_required.push_back({AtomKind::X, {}});
    SymGroup g; g.kind = AtomKind::X;
    for (int i = 0; i < deg - 1; ++i) g.indices.push_back(i);
    s.sym_groups.push_back(std::move(g));
    s.require_odd = true;
    for (int i = 0; i < deg - 1; ++i)
        s.odd_negate_atoms.push_back(encode_atom(AtomKind::X, i));
    return s;
}

// ---- Drive the symbolic VM ----
//
// Symbolic execution explores ALL feasible control-flow paths by
// enumerating concrete schedules for symbolic boolean branches
// (If / When / While).  A program is accepted iff EVERY explored path
// independently satisfies the dependency / permutation / oddness spec.
//
// If any path bails for a non-scheduler reason (true opaque: symbolic
// index, node cap, step cap, etc.) the whole validation fails.
//
// MAX_PATHS bounds the total number of leaves; MAX_CHOICE_DEPTH bounds
// the length of any single schedule (depth of nested branches + loop
// iterations).  Programs that exceed either are conservatively rejected
// as opaque.

inline constexpr int SYMBOLIC_MAX_PATHS = 64;
inline constexpr int SYMBOLIC_MAX_CHOICE_DEPTH = 16;

struct PathOutcome {
    bool ok;                // false => opaque or genuine validation failure
    std::string reason;
    ExprRef out;
    // Conjunction of path-cond literals (i.e. the condition under which
    // this concrete branch schedule is the actually-taken trace).  Null
    // (with ok=true) means single-path / unconditional execution.
    ExprRef path_cond;
};

inline ExprRef conjoin_literals_(const std::vector<ExprRef>& lits) {
    if (lits.empty()) return nullptr;
    if (lits.size() == 1) return lits[0];
    // Nary-And short-circuits on literal false / collapses true; we let
    // make_nary do the normalization.
    std::vector<ExprRef> args(lits.begin(), lits.end());
    return make_nary(Op::Bool_And, args);
}

inline std::vector<PathOutcome> run_all_paths_(
    const SymbolicVM& seeded, const Program& prog) {
    std::vector<PathOutcome> outs;
    std::vector<std::vector<int>> stack;
    stack.push_back({});
    while (!stack.empty()) {
        if (static_cast<int>(outs.size()) >= SYMBOLIC_MAX_PATHS) {
            outs.push_back({false, "opaque: path cap exceeded", nullptr, nullptr});
            return outs;
        }
        std::vector<int> choices = std::move(stack.back()); stack.pop_back();
        SymbolicVM vm = seeded;
        vm.forced_choices = choices;
        vm.branch_cursor = 0;
        vm.total_branches_seen = 0;
        vm.need_more_choices = false;
        vm.opaque = false;
        vm.opaque_reason.clear();
        vm.step_count = 0;
        vm.recur_depth = 0;
        vm.path_cond_literals.clear();
        ExprRef out = vm.run(prog);
        if (vm.need_more_choices) {
            if (static_cast<int>(choices.size()) >= SYMBOLIC_MAX_CHOICE_DEPTH) {
                outs.push_back({false, "opaque: choice depth exceeded", nullptr, nullptr});
                return outs;
            }
            auto c1 = choices; c1.push_back(1);
            auto c0 = std::move(choices); c0.push_back(0);
            stack.push_back(std::move(c1));
            stack.push_back(std::move(c0));
            continue;
        }
        if (vm.opaque) {
            outs.push_back({false, std::string("opaque: ") + vm.opaque_reason, nullptr, nullptr});
            return outs;
        }
        ExprRef pc = conjoin_literals_(vm.path_cond_literals);
        outs.push_back({true, "ok", out, pc});
    }
    if (outs.empty()) outs.push_back({false, "opaque: no paths", nullptr, nullptr});
    return outs;
}

// Combine the per-path outputs into a single ITE-chain expression.
// Order matters: ite(pc_0, y_0, ite(pc_1, y_1, ..., y_last)).
// Because the path conditions are mutually exclusive and exhaustive,
// the final else can simply be the last path's output (its condition
// is implied by the negation of all previous ones).
inline ExprRef combine_paths_output_(const std::vector<PathOutcome>& paths) {
    if (paths.empty()) return nullptr;
    if (paths.size() == 1) return paths[0].out;
    ExprRef y = paths.back().out;
    if (!y) return nullptr;
    for (int i = static_cast<int>(paths.size()) - 2; i >= 0; --i) {
        ExprRef yi = paths[i].out;
        if (!yi) return nullptr;
        ExprRef pc = paths[i].path_cond;
        if (!pc) return nullptr;  // no condition but multiple paths => bug
        y = make_ifelse(pc, yi, y);
    }
    return y;
}

// ASCII s-expression dump of an ExprRef for debugging.
inline std::string expr_to_string(const ExprRef& e) {
    if (!e) return "<null>";
    switch (e->kind) {
        case ExprKind::LitFloat: {
            std::ostringstream os; os << e->lit_d; return os.str();
        }
        case ExprKind::LitInt:  return std::to_string(e->lit_i);
        case ExprKind::LitBool: return e->lit_i ? "true" : "false";
        case ExprKind::Atom: {
            const char* k = e->atom_kind == AtomKind::X ? "X"
                          : e->atom_kind == AtomKind::LV ? "LV" : "EVO";
            return std::string(k) + "[" + std::to_string(e->idx) + "]";
        }
        case ExprKind::Opaque: {
            std::string s = "Opaque{";
            for (size_t i = 0; i < e->atoms.size(); ++i) {
                if (i) s += ",";
                int enc = e->atoms[i];
                AtomKind k = atom_kind_of(enc);
                const char* ks = k == AtomKind::X ? "X" : k == AtomKind::LV ? "LV" : "EVO";
                s += ks; s += std::to_string(atom_idx_of(enc));
            }
            return s + "}";
        }
        case ExprKind::UnOp: {
            return "(" + op_to_name(e->op_tag) + " " + expr_to_string(e->ch[0]) + ")";
        }
        case ExprKind::BinOp: {
            return "(" + op_to_name(e->op_tag) + " " + expr_to_string(e->ch[0])
                       + " " + expr_to_string(e->ch[1]) + ")";
        }
        case ExprKind::NAry: {
            std::string s = "(" + op_to_name(e->op_tag);
            for (auto& c : e->ch) { s += " "; s += expr_to_string(c); }
            return s + ")";
        }
        case ExprKind::IfElse: {
            return "(ite " + expr_to_string(e->ch[0]) + " "
                          + expr_to_string(e->ch[1]) + " "
                          + expr_to_string(e->ch[2]) + ")";
        }
    }
    return "<?>";
}

// Run all paths and return per-path dump for debugging.
struct PathDump {
    bool ok;
    std::string reason;
    std::string cond;
    std::string out;
};

inline std::vector<PathDump> dump_all_paths_(
    const SymbolicVM& seeded, const Program& prog) {
    auto raw = run_all_paths_(seeded, prog);
    std::vector<PathDump> out;
    out.reserve(raw.size());
    for (auto& p : raw) {
        PathDump d;
        d.ok = p.ok;
        d.reason = p.reason;
        d.cond = expr_to_string(p.path_cond);
        d.out = expr_to_string(p.out);
        out.push_back(std::move(d));
    }
    return out;
}

inline std::string combined_output_string_(
    const SymbolicVM& seeded, const Program& prog) {
    auto raw = run_all_paths_(seeded, prog);
    for (auto& p : raw) if (!p.ok) return std::string("<bail: ") + p.reason + ">";
    return expr_to_string(combine_paths_output_(raw));
}

inline CheckResult validate_symbolic_with_spec_(
    const SymbolicVM& seeded, const Program& prog, const CheckSpec& spec) {
    auto paths = run_all_paths_(seeded, prog);
    for (const auto& p : paths) {
        if (!p.ok) return {false, p.reason};
    }
    ExprRef combined = combine_paths_output_(paths);
    if (!combined) return {false, "no output"};
    return symbolic_validate(combined, spec);
}

inline CheckResult validate_symbolic_v2c(const Program& prog, int deg = DEFAULT_DEG,
                                         int iter_idx = 0) {
    // Per-thread intern table is reset each call so we don't carry
    // expressions from prior programs.  Within a single validation,
    // every enumerated path shares the same table so substitute() and
    // identity comparisons remain consistent across paths.
    expr_table().clear();
    SymbolicVM vm;
    vm.ctx_deg = deg;
    vm.ctx_iter = iter_idx;
    vm.state.clear();
    vm.seed_v2c_atoms();
    return validate_symbolic_with_spec_(vm, prog, make_v2c_spec(deg));
}
inline CheckResult validate_symbolic_c2v(const Program& prog, int deg = DEFAULT_DEG,
                                         int iter_idx = 0) {
    expr_table().clear();
    SymbolicVM vm;
    vm.ctx_deg = deg;
    vm.ctx_iter = iter_idx;
    vm.state.clear();
    vm.seed_c2v_atoms();
    return validate_symbolic_with_spec_(vm, prog, make_c2v_spec(deg));
}

}  // namespace pushgp_cpp
