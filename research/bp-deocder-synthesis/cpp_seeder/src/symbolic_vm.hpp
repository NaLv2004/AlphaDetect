// symbolic_vm.hpp — shadow VM that executes Push-GP programs over Expr nodes
// instead of concrete values.  Mirrors vm.hpp's dispatch structure 1:1.
//
// Semantics (mirroring concrete VM):
//   * Each stack holds ExprRef (scalars) or vector<ExprRef> (vectors).
//   * Constant folding is done by the make_* factories — concrete values
//     stay concrete unless mixed with symbolic atoms.
//   * Branch with symbolic condition: execute both branches over a stack
//     snapshot, then merge element-wise via make_ifelse.  If post-branch
//     stack depths differ across branches, mark the VM OPAQUE.
//   * DoTimes / DoRange / While with symbolic bound or condition →
//     mark OPAQUE immediately.
//   * Memory: array<ExprRef, 16>, also merged via ifelse on branch.
//   * Any FVec_At with a symbolic index pushes Opaque(union of element atoms).
//
// `opaque` means the VM cannot prove any structural property of the result
// → the validator rejects.
#pragma once

#include "common.hpp"
#include "instruction.hpp"
#include "opcodes.hpp"
#include "symbolic_expr.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace pushgp_cpp {

using FVecSym = std::vector<ExprRef>;
using BVecSym = std::vector<ExprRef>;
using IVecSym = std::vector<ExprRef>;

struct SymStacks {
    std::vector<ExprRef> floats, ints, bools;
    std::vector<FVecSym> fvecs;
    std::vector<BVecSym> bvecs;
    std::vector<IVecSym> ivecs;
    std::array<ExprRef, MEMORY_SIZE> memory;

    void clear() {
        floats.clear(); ints.clear(); bools.clear();
        fvecs.clear(); bvecs.clear(); ivecs.clear();
        for (auto& m : memory) m = make_lit_float(0.0);
    }
};

class SymbolicVM {
   public:
    SymStacks state;
    bool opaque = false;
    std::string opaque_reason;
    int step_count = 0;
    int recur_depth = 0;
    int step_max = VM_STEP_MAX;
    int recur_max = VM_RECUR_MAX;

    // Optional per-instruction trace hook. Called *after* dispatch_ of each
    // instruction inside execute_block, including nested invocations from
    // control-flow handlers. Used by symbolic_trace_v2c binding for
    // step-by-step debugging.
    std::function<void(const Instruction&)> trace_hook;

    // ---------------- multi-path execution ----------------
    //
    // When the VM hits a symbolic boolean condition (If / When / While)
    // it consumes one entry from `forced_choices` to decide which branch
    // to take.  If the cursor reaches the end of the choice vector while
    // a symbolic branch is still pending, the VM sets `need_more_choices`
    // and bails (opaque).  The outer enumerator (`run_all_paths_`) is
    // responsible for re-running the program with longer choice vectors
    // until every leaf path either terminates cleanly or the global cap
    // is hit.  This lets us symbolically execute arbitrary control flow
    // by enumerating finitely many concrete branch schedules instead of
    // bailing the way the single-state merge approach does.
    std::vector<int> forced_choices;
    int  branch_cursor = 0;
    int  total_branches_seen = 0;
    bool need_more_choices = false;

    // For every consumed symbolic branch we record the literal that was
    // assumed (the popped condition if the choice was "true", otherwise
    // its boolean negation).  The conjunction of these literals is the
    // path condition under which the current execution is valid; the
    // outer enumerator combines per-path outputs into a single
    // ITE-chain expression and validates that combined expression.
    std::vector<ExprRef> path_cond_literals;

    // Validator-provided env scalars (concrete, all become LitFloat/LitInt).
    int ctx_deg = 8;
    int ctx_iter = 0;
    int ctx_max_iter = 25;
    int ctx_edge_index = 0;
    double ctx_noise_var = 1.0;
    double ctx_code_rate = 0.5;
    bool ctx_has_channel_llr = true;

    void mark_opaque(const char* why) {
        if (!opaque) { opaque = true; opaque_reason = why; }
    }

    // Consume one schedule bit; on cursor exhaustion, request more choices
    // and mark the VM opaque so the outer enumerator can re-run.  We also
    // record the path-cond literal (the popped cond if chosen true, else
    // Not(cond)) so that the outer enumerator can stitch paths back into
    // a single ITE-combined output expression.
    bool consume_branch_(const ExprRef& cond_expr, bool& out) {
        ++total_branches_seen;
        if (branch_cursor < static_cast<int>(forced_choices.size())) {
            bool taken = forced_choices[branch_cursor++] != 0;
            out = taken;
            ExprRef lit = taken ? cond_expr
                                : make_unop(Op::Bool_Not, cond_expr);
            path_cond_literals.push_back(lit);
            return true;
        }
        need_more_choices = true;
        mark_opaque("need more branch choices");
        return false;
    }

    void seed_v2c_atoms() {
        // Float stack: Lv  (channel LLR atom)
        push_f(make_atom(AtomKind::LV, 0));
        // Int stack: edge_idx, deg, iter, max_iter (concrete)
        push_i(make_lit_int(ctx_edge_index));
        push_i(make_lit_int(ctx_deg));
        push_i(make_lit_int(ctx_iter));
        push_i(make_lit_int(ctx_max_iter));
        // FVec stack: incoming = [X0, X1, ..., X_{deg-2}]
        FVecSym v;
        int n_in = ctx_deg - 1;
        for (int i = 0; i < n_in; ++i) v.push_back(make_atom(AtomKind::X, i));
        state.fvecs.push_back(std::move(v));
    }
    void seed_c2v_atoms() {
        ctx_has_channel_llr = false;
        push_i(make_lit_int(ctx_edge_index));
        push_i(make_lit_int(ctx_deg));
        push_i(make_lit_int(ctx_iter));
        push_i(make_lit_int(ctx_max_iter));
        FVecSym v;
        int n_in = ctx_deg - 1;
        for (int i = 0; i < n_in; ++i) v.push_back(make_atom(AtomKind::X, i));
        state.fvecs.push_back(std::move(v));
    }

    // Stack helpers
    void push_f(ExprRef e) { state.floats.push_back(std::move(e)); cap_(); }
    void push_i(ExprRef e) { state.ints.push_back(std::move(e)); cap_(); }
    void push_b(ExprRef e) { state.bools.push_back(std::move(e)); cap_(); }

    bool pop_f(ExprRef& out) {
        if (state.floats.empty()) return false;
        out = state.floats.back(); state.floats.pop_back(); return true;
    }
    bool pop_i(ExprRef& out) {
        if (state.ints.empty()) return false;
        out = state.ints.back(); state.ints.pop_back(); return true;
    }
    bool pop_b(ExprRef& out) {
        if (state.bools.empty()) return false;
        out = state.bools.back(); state.bools.pop_back(); return true;
    }

    // Run program; returns top of floats or null if empty.
    ExprRef run(const Program& prog) {
        execute_block(prog);
        if (opaque) return nullptr;
        if (state.floats.empty()) return nullptr;
        return state.floats.back();
    }

    void execute_block(const Program& block) {
        if (recur_depth >= recur_max) { mark_opaque("recur_max exceeded"); return; }
        ++recur_depth;
        for (const auto& ins : block) {
            if (opaque) break;
            ++step_count;
            if (step_count > step_max) { mark_opaque("step_max exceeded"); break; }
            dispatch_(ins);
            if (trace_hook) trace_hook(ins);
        }
        --recur_depth;
    }

   private:
    void cap_() {
        if (expr_table().size() > static_cast<size_t>(SYMBOLIC_NODE_CAP()))
            mark_opaque("node cap exceeded");
    }

    static int64_t mod_pos_(int64_t i, int64_t n) {
        if (n <= 0) return 0;
        int64_t r = i % n;
        if (r < 0) r += n;
        return r;
    }
    static bool concrete_int_(const ExprRef& e, int64_t& out) {
        if (e && e->kind == ExprKind::LitInt) { out = e->lit_i; return true; }
        return false;
    }
    static bool concrete_bool_(const ExprRef& e, bool& out) {
        if (e && e->kind == ExprKind::LitBool) { out = e->lit_i != 0; return true; }
        return false;
    }

    void f_unop_(Op op) {
        ExprRef a; if (!pop_f(a)) return;
        push_f(make_unop(op, a));
    }
    void f_binop_(Op op) {
        ExprRef b, a;
        if (!pop_f(b)) return;
        if (!pop_f(a)) { push_f(b); return; }
        push_f(make_binop(op, a, b));
    }
    void f_binop_nary_(Op op) {
        ExprRef b, a;
        if (!pop_f(b)) return;
        if (!pop_f(a)) { push_f(b); return; }
        push_f(make_nary(op, {a, b}));
    }
    void i_unop_(Op op) {
        ExprRef a; if (!pop_i(a)) return;
        push_i(make_unop(op, a));
    }
    void i_binop_(Op op) {
        ExprRef b, a;
        if (!pop_i(b)) return;
        if (!pop_i(a)) { push_i(b); return; }
        push_i(make_binop(op, a, b));
    }
    void i_binop_nary_(Op op) {
        ExprRef b, a;
        if (!pop_i(b)) return;
        if (!pop_i(a)) { push_i(b); return; }
        push_i(make_nary(op, {a, b}));
    }
    void b_binop_nary_(Op op) {
        ExprRef b, a;
        if (!pop_b(b)) return;
        if (!pop_b(a)) { push_b(b); return; }
        push_b(make_nary(op, {a, b}));
    }
    void cmp_f_(Op op, bool nary) {
        ExprRef b, a;
        if (!pop_f(b)) return;
        if (!pop_f(a)) { push_f(b); return; }
        push_b(nary ? make_nary(op, {a, b}) : make_binop(op, a, b));
    }
    void cmp_i_(Op op, bool nary) {
        ExprRef b, a;
        if (!pop_i(b)) return;
        if (!pop_i(a)) { push_i(b); return; }
        push_b(nary ? make_nary(op, {a, b}) : make_binop(op, a, b));
    }

    // Branch merge: snapshot/restore stacks, run two paths, merge.
    struct StackSnapshot {
        std::vector<ExprRef> f, i, b;
        std::vector<FVecSym> fv;
        std::vector<BVecSym> bv;
        std::vector<IVecSym> iv;
        std::array<ExprRef, MEMORY_SIZE> mem;
    };
    StackSnapshot snapshot_() const {
        return {state.floats, state.ints, state.bools,
                state.fvecs, state.bvecs, state.ivecs, state.memory};
    }
    void restore_(StackSnapshot& s) {
        state.floats = std::move(s.f);
        state.ints   = std::move(s.i);
        state.bools  = std::move(s.b);
        state.fvecs  = std::move(s.fv);
        state.bvecs  = std::move(s.bv);
        state.ivecs  = std::move(s.iv);
        state.memory = std::move(s.mem);
    }
    bool merge_(const StackSnapshot& th, const StackSnapshot& el, const ExprRef& cond) {
        if (th.f.size() != el.f.size() || th.i.size() != el.i.size() ||
            th.b.size() != el.b.size() ||
            th.fv.size() != el.fv.size() || th.bv.size() != el.bv.size() ||
            th.iv.size() != el.iv.size()) {
            return false;
        }
        state.floats.resize(th.f.size());
        for (size_t k = 0; k < th.f.size(); ++k)
            state.floats[k] = make_ifelse(cond, th.f[k], el.f[k]);
        state.ints.resize(th.i.size());
        for (size_t k = 0; k < th.i.size(); ++k)
            state.ints[k] = make_ifelse(cond, th.i[k], el.i[k]);
        state.bools.resize(th.b.size());
        for (size_t k = 0; k < th.b.size(); ++k)
            state.bools[k] = make_ifelse(cond, th.b[k], el.b[k]);
        // Vectors: must have matching depths AND matching element counts per slot.
        state.fvecs.resize(th.fv.size());
        for (size_t k = 0; k < th.fv.size(); ++k) {
            if (th.fv[k].size() != el.fv[k].size()) return false;
            state.fvecs[k].resize(th.fv[k].size());
            for (size_t e = 0; e < th.fv[k].size(); ++e)
                state.fvecs[k][e] = make_ifelse(cond, th.fv[k][e], el.fv[k][e]);
        }
        state.bvecs.resize(th.bv.size());
        for (size_t k = 0; k < th.bv.size(); ++k) {
            if (th.bv[k].size() != el.bv[k].size()) return false;
            state.bvecs[k].resize(th.bv[k].size());
            for (size_t e = 0; e < th.bv[k].size(); ++e)
                state.bvecs[k][e] = make_ifelse(cond, th.bv[k][e], el.bv[k][e]);
        }
        state.ivecs.resize(th.iv.size());
        for (size_t k = 0; k < th.iv.size(); ++k) {
            if (th.iv[k].size() != el.iv[k].size()) return false;
            state.ivecs[k].resize(th.iv[k].size());
            for (size_t e = 0; e < th.iv[k].size(); ++e)
                state.ivecs[k][e] = make_ifelse(cond, th.iv[k][e], el.iv[k][e]);
        }
        for (size_t k = 0; k < MEMORY_SIZE; ++k)
            state.memory[k] = make_ifelse(cond, th.mem[k], el.mem[k]);
        return true;
    }

    // ----------------- main dispatch -----------------
    void dispatch_(const Instruction& ins) {
        using O = Op;
        switch (ins.op) {
        // ============== Float arith ==============
        case O::Float_Add: f_binop_nary_(O::Float_Add); break;
        case O::Float_Sub: f_binop_(O::Float_Sub); break;
        case O::Float_Mul: f_binop_nary_(O::Float_Mul); break;
        case O::Float_Div: f_binop_(O::Float_Div); break;
        case O::Float_Mod: f_binop_(O::Float_Mod); break;
        case O::Float_Min: f_binop_nary_(O::Float_Min); break;
        case O::Float_Max: f_binop_nary_(O::Float_Max); break;
        case O::Float_Abs:    f_unop_(O::Float_Abs); break;
        case O::Float_Neg:    f_unop_(O::Float_Neg); break;
        case O::Float_Inv:    f_unop_(O::Float_Inv); break;
        case O::Float_Sqrt:   f_unop_(O::Float_Sqrt); break;
        case O::Float_Square: f_unop_(O::Float_Square); break;
        case O::Float_Exp:    f_unop_(O::Float_Exp); break;
        case O::Float_Log:    f_unop_(O::Float_Log); break;
        case O::Float_Tanh:   f_unop_(O::Float_Tanh); break;
        case O::Float_Atanh:  f_unop_(O::Float_Atanh); break;
        case O::Float_Sign:   f_unop_(O::Float_Sign); break;
        case O::Float_Floor:  f_unop_(O::Float_Floor); break;
        case O::Float_Ceil:   f_unop_(O::Float_Ceil); break;

        // ============== Float consts ==============
        case O::Float_Const0:    push_f(make_lit_float(0.0)); break;
        case O::Float_Const1:    push_f(make_lit_float(1.0)); break;
        case O::Float_ConstNeg1: push_f(make_lit_float(-1.0)); break;
        case O::Float_ConstHalf: push_f(make_lit_float(0.5)); break;
        case O::Float_Const2:    push_f(make_lit_float(2.0)); break;
        case O::Float_Const0_1:  push_f(make_lit_float(0.1)); break;
        case O::Float_ConstPi:   push_f(make_lit_float(3.14159265358979323846)); break;
        case O::Float_Const1eNeg6: push_f(make_lit_float(1e-6)); break;
        case O::Float_EvoConst0: push_f(make_atom(AtomKind::EVO, 0)); break;
        case O::Float_EvoConst1: push_f(make_atom(AtomKind::EVO, 1)); break;
        case O::Float_EvoConst2: push_f(make_atom(AtomKind::EVO, 2)); break;
        case O::Float_EvoConst3: push_f(make_atom(AtomKind::EVO, 3)); break;
        case O::Float_EvoConst4: push_f(make_atom(AtomKind::EVO, 4)); break;
        case O::Float_EvoConst5: push_f(make_atom(AtomKind::EVO, 5)); break;
        case O::Float_EvoConst6: push_f(make_atom(AtomKind::EVO, 6)); break;
        case O::Float_EvoConst7: push_f(make_atom(AtomKind::EVO, 7)); break;

        // ============== Float stack ==============
        case O::Float_Pop:   if (!state.floats.empty()) state.floats.pop_back(); break;
        case O::Float_Dup:   if (!state.floats.empty()) state.floats.push_back(state.floats.back()); break;
        case O::Float_Swap:  if (state.floats.size() >= 2) std::swap(state.floats[state.floats.size()-1], state.floats[state.floats.size()-2]); break;
        case O::Float_Rot:   if (state.floats.size() >= 3) {
            auto top = state.floats.back(); state.floats.pop_back();
            auto mid = state.floats.back(); state.floats.pop_back();
            auto bot = state.floats.back(); state.floats.pop_back();
            state.floats.push_back(mid);
            state.floats.push_back(top);
            state.floats.push_back(bot);
        } break;
        case O::Float_Yank: {
            ExprRef ne; if (!pop_i(ne)) break;
            int64_t n; if (!concrete_int_(ne, n)) { mark_opaque("symbolic yank index"); break; }
            int sz = static_cast<int>(state.floats.size());
            if (sz == 0) break;
            int k = static_cast<int>(mod_pos_(n, sz));
            int idx = sz - 1 - k;
            if (idx < 0 || idx >= sz) break;
            ExprRef v = state.floats[idx];
            state.floats.erase(state.floats.begin() + idx);
            state.floats.push_back(v);
        } break;
        case O::Float_Shove: {
            ExprRef ne; if (!pop_i(ne)) break;
            int64_t n; if (!concrete_int_(ne, n)) { mark_opaque("symbolic shove index"); break; }
            int sz = static_cast<int>(state.floats.size());
            if (sz == 0) break;
            ExprRef v = state.floats.back(); state.floats.pop_back();
            sz -= 1;
            int k = static_cast<int>(mod_pos_(n, sz + 1));
            int idx = sz - k;
            if (idx < 0) idx = 0;
            if (idx > sz) idx = sz;
            state.floats.insert(state.floats.begin() + idx, v);
        } break;

        // ============== Int arith ==============
        case O::Int_Add: i_binop_nary_(O::Int_Add); break;
        case O::Int_Sub: i_binop_(O::Int_Sub); break;
        case O::Int_Mul: i_binop_nary_(O::Int_Mul); break;
        case O::Int_Div: i_binop_(O::Int_Div); break;
        case O::Int_Mod: i_binop_(O::Int_Mod); break;
        case O::Int_Min: i_binop_nary_(O::Int_Min); break;
        case O::Int_Max: i_binop_nary_(O::Int_Max); break;
        case O::Int_Inc: i_unop_(O::Int_Inc); break;
        case O::Int_Dec: i_unop_(O::Int_Dec); break;
        case O::Int_Neg: i_unop_(O::Int_Neg); break;

        case O::Int_Const0:    push_i(make_lit_int(0)); break;
        case O::Int_Const1:    push_i(make_lit_int(1)); break;
        case O::Int_Const2:    push_i(make_lit_int(2)); break;
        case O::Int_ConstNeg1: push_i(make_lit_int(-1)); break;

        case O::Int_Pop: if (!state.ints.empty()) state.ints.pop_back(); break;
        case O::Int_Dup: if (!state.ints.empty()) state.ints.push_back(state.ints.back()); break;
        case O::Int_Swap: if (state.ints.size() >= 2) std::swap(state.ints[state.ints.size()-1], state.ints[state.ints.size()-2]); break;
        case O::Int_Rot: if (state.ints.size() >= 3) {
            auto top = state.ints.back(); state.ints.pop_back();
            auto mid = state.ints.back(); state.ints.pop_back();
            auto bot = state.ints.back(); state.ints.pop_back();
            state.ints.push_back(mid);
            state.ints.push_back(top);
            state.ints.push_back(bot);
        } break;

        // ============== Bool ==============
        case O::Bool_And: b_binop_nary_(O::Bool_And); break;
        case O::Bool_Or:  b_binop_nary_(O::Bool_Or); break;
        case O::Bool_Xor: b_binop_nary_(O::Bool_Xor); break;
        case O::Bool_Not: { ExprRef a; if (pop_b(a)) push_b(make_unop(O::Bool_Not, a)); } break;
        case O::Bool_True:  push_b(make_lit_bool(true)); break;
        case O::Bool_False: push_b(make_lit_bool(false)); break;
        case O::Bool_Pop:  if (!state.bools.empty()) state.bools.pop_back(); break;
        case O::Bool_Dup:  if (!state.bools.empty()) state.bools.push_back(state.bools.back()); break;
        case O::Bool_Swap: if (state.bools.size() >= 2) std::swap(state.bools[state.bools.size()-1], state.bools[state.bools.size()-2]); break;

        // ============== Compare ==============
        case O::Float_LT: cmp_f_(O::Float_LT, false); break;
        case O::Float_GT: cmp_f_(O::Float_GT, false); break;
        case O::Float_EQ: cmp_f_(O::Float_EQ, true); break;
        case O::Int_LT: cmp_i_(O::Int_LT, false); break;
        case O::Int_GT: cmp_i_(O::Int_GT, false); break;
        case O::Int_EQ: cmp_i_(O::Int_EQ, true); break;

        // ============== Conversions ==============
        case O::Float_FromInt: { ExprRef a; if (pop_i(a)) push_f(make_unop(O::Float_FromInt, a)); } break;
        case O::Int_FromFloat: { ExprRef a; if (pop_f(a)) push_i(make_unop(O::Int_FromFloat, a)); } break;
        case O::Int_FromBool:  { ExprRef a; if (pop_b(a)) push_i(make_unop(O::Int_FromBool, a)); } break;
        case O::Bool_FromFloat:{ ExprRef a; if (pop_f(a)) push_b(make_unop(O::Bool_FromFloat, a)); } break;
        case O::Bool_FromInt:  { ExprRef a; if (pop_i(a)) push_b(make_unop(O::Bool_FromInt, a)); } break;

        // ============== FVec ==============
        case O::FVec_Len: {
            if (state.fvecs.empty()) break;
            push_i(make_lit_int(static_cast<int64_t>(state.fvecs.back().size())));
        } break;
        case O::FVec_At: {
            if (state.fvecs.empty() || state.fvecs.back().empty() || state.ints.empty()) break;
            ExprRef ie; pop_i(ie);
            const auto& v = state.fvecs.back();
            int64_t i;
            if (concrete_int_(ie, i)) {
                int k = static_cast<int>(mod_pos_(i, static_cast<int64_t>(v.size())));
                push_f(v[k]);
            } else {
                // symbolic index → OR over all positions
                std::vector<int> atoms = ie->atoms;
                for (const auto& e : v) atoms = merge_atoms(atoms, e->atoms);
                push_f(make_opaque(std::move(atoms)));
            }
        } break;
        case O::FVec_Set: {
            if (state.fvecs.empty() || state.fvecs.back().empty()
                || state.ints.empty() || state.floats.empty()) break;
            ExprRef ie; pop_i(ie);
            ExprRef val; pop_f(val);
            auto& v = state.fvecs.back();
            int64_t i;
            if (concrete_int_(ie, i)) {
                int k = static_cast<int>(mod_pos_(i, static_cast<int64_t>(v.size())));
                v[k] = val;
            } else {
                // symbolic index → all positions become OR(val, old_pos_e)
                std::vector<int> atoms = ie->atoms;
                atoms = merge_atoms(atoms, val->atoms);
                for (auto& e : v) atoms = merge_atoms(atoms, e->atoms);
                ExprRef op = make_opaque(atoms);
                for (auto& e : v) e = op;
            }
        } break;
        case O::FVec_Push: {
            if (state.fvecs.empty() || state.floats.empty()) break;
            ExprRef val; pop_f(val);
            auto& v = state.fvecs.back();
            v.push_back(val);
            if (static_cast<int>(v.size()) > MAX_VEC_LEN) v.resize(MAX_VEC_LEN);
        } break;
        case O::FVec_PopBack: {
            if (state.fvecs.empty() || state.fvecs.back().empty()) break;
            ExprRef last = state.fvecs.back().back();
            state.fvecs.back().pop_back();
            push_f(last);
        } break;
        case O::FVec_New: {
            ExprRef ne; if (!pop_i(ne)) break;
            int64_t n;
            if (!concrete_int_(ne, n)) { mark_opaque("symbolic FVec_New len"); break; }
            int nn = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(n, MAX_VEC_LEN)));
            FVecSym v(nn, make_lit_float(0.0));
            state.fvecs.push_back(std::move(v));
        } break;
        case O::FVec_FromFloat: {
            ExprRef v; if (!pop_f(v)) break;
            state.fvecs.push_back(FVecSym{v});
        } break;
        case O::FVec_Concat: {
            if (state.fvecs.size() < 2) break;
            FVecSym b = state.fvecs.back(); state.fvecs.pop_back();
            FVecSym a = state.fvecs.back(); state.fvecs.pop_back();
            FVecSym out; out.reserve(a.size() + b.size());
            out.insert(out.end(), a.begin(), a.end());
            out.insert(out.end(), b.begin(), b.end());
            if (static_cast<int>(out.size()) > MAX_VEC_LEN) out.resize(MAX_VEC_LEN);
            state.fvecs.push_back(std::move(out));
        } break;
        case O::FVec_Slice: {
            if (state.fvecs.empty() || state.ints.size() < 2) break;
            ExprRef ee; pop_i(ee);
            ExprRef se; pop_i(se);
            int64_t e_, s_;
            if (!concrete_int_(ee, e_) || !concrete_int_(se, s_)) {
                mark_opaque("symbolic FVec_Slice bounds"); break;
            }
            const auto& v = state.fvecs.back();
            int64_t n = static_cast<int64_t>(v.size());
            int64_t s = std::max<int64_t>(0, std::min<int64_t>(s_, n));
            int64_t en = std::max<int64_t>(s, std::min<int64_t>(e_, n));
            FVecSym sliced(v.begin() + s, v.begin() + en);
            state.fvecs.pop_back();
            state.fvecs.push_back(std::move(sliced));
        } break;
        case O::FVec_Reverse: {
            if (state.fvecs.empty()) break;
            auto& v = state.fvecs.back();
            std::reverse(v.begin(), v.end());
        } break;
        case O::FVec_Roll: {
            if (state.fvecs.empty() || state.ints.empty()) break;
            ExprRef ke; pop_i(ke);
            int64_t k;
            if (!concrete_int_(ke, k)) { mark_opaque("symbolic FVec_Roll"); break; }
            auto& v = state.fvecs.back();
            if (v.empty()) break;
            int64_t n = static_cast<int64_t>(v.size());
            int64_t shift = mod_pos_(k, n);
            if (shift == 0) break;
            FVecSym r(v.size());
            for (int64_t i = 0; i < n; ++i) r[(i + shift) % n] = v[static_cast<size_t>(i)];
            v = std::move(r);
        } break;
        case O::FVec_Pop:  if (!state.fvecs.empty()) state.fvecs.pop_back(); break;
        case O::FVec_Dup:  if (!state.fvecs.empty()) state.fvecs.push_back(state.fvecs.back()); break;
        case O::FVec_Swap: if (state.fvecs.size() >= 2) std::swap(state.fvecs[state.fvecs.size()-1], state.fvecs[state.fvecs.size()-2]); break;
        case O::FVec_Rot:  if (state.fvecs.size() >= 3) {
            auto t = state.fvecs.back(); state.fvecs.pop_back();
            auto m = state.fvecs.back(); state.fvecs.pop_back();
            auto b = state.fvecs.back(); state.fvecs.pop_back();
            state.fvecs.push_back(std::move(m));
            state.fvecs.push_back(std::move(t));
            state.fvecs.push_back(std::move(b));
        } break;

        // ============== BVec ==============
        case O::BVec_Len: {
            if (state.bvecs.empty()) break;
            push_i(make_lit_int(static_cast<int64_t>(state.bvecs.back().size())));
        } break;
        case O::BVec_At: {
            if (state.bvecs.empty() || state.bvecs.back().empty() || state.ints.empty()) break;
            ExprRef ie; pop_i(ie);
            const auto& v = state.bvecs.back();
            int64_t i;
            if (concrete_int_(ie, i)) {
                int k = static_cast<int>(mod_pos_(i, static_cast<int64_t>(v.size())));
                push_b(v[k]);
            } else {
                std::vector<int> atoms = ie->atoms;
                for (const auto& e : v) atoms = merge_atoms(atoms, e->atoms);
                push_b(make_opaque(std::move(atoms)));
            }
        } break;
        case O::BVec_Set: {
            if (state.bvecs.empty() || state.bvecs.back().empty()
                || state.ints.empty() || state.bools.empty()) break;
            ExprRef ie; pop_i(ie);
            ExprRef val; pop_b(val);
            auto& v = state.bvecs.back();
            int64_t i;
            if (concrete_int_(ie, i)) {
                int k = static_cast<int>(mod_pos_(i, static_cast<int64_t>(v.size())));
                v[k] = val;
            } else {
                std::vector<int> atoms = ie->atoms;
                atoms = merge_atoms(atoms, val->atoms);
                for (auto& e : v) atoms = merge_atoms(atoms, e->atoms);
                ExprRef op = make_opaque(atoms);
                for (auto& e : v) e = op;
            }
        } break;
        case O::BVec_Push: {
            if (state.bvecs.empty() || state.bools.empty()) break;
            ExprRef val; pop_b(val);
            auto& v = state.bvecs.back();
            v.push_back(val);
            if (static_cast<int>(v.size()) > MAX_VEC_LEN) v.resize(MAX_VEC_LEN);
        } break;
        case O::BVec_PopBack: {
            if (state.bvecs.empty() || state.bvecs.back().empty()) break;
            ExprRef last = state.bvecs.back().back();
            state.bvecs.back().pop_back();
            push_b(last);
        } break;
        case O::BVec_New: {
            ExprRef ne; if (!pop_i(ne)) break;
            int64_t n;
            if (!concrete_int_(ne, n)) { mark_opaque("symbolic BVec_New"); break; }
            int nn = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(n, MAX_VEC_LEN)));
            state.bvecs.push_back(BVecSym(nn, make_lit_bool(false)));
        } break;
        case O::BVec_Concat: {
            if (state.bvecs.size() < 2) break;
            BVecSym b = state.bvecs.back(); state.bvecs.pop_back();
            BVecSym a = state.bvecs.back(); state.bvecs.pop_back();
            BVecSym out; out.reserve(a.size() + b.size());
            out.insert(out.end(), a.begin(), a.end());
            out.insert(out.end(), b.begin(), b.end());
            if (static_cast<int>(out.size()) > MAX_VEC_LEN) out.resize(MAX_VEC_LEN);
            state.bvecs.push_back(std::move(out));
        } break;
        case O::BVec_Slice: {
            if (state.bvecs.empty() || state.ints.size() < 2) break;
            ExprRef ee; pop_i(ee);
            ExprRef se; pop_i(se);
            int64_t e_, s_;
            if (!concrete_int_(ee, e_) || !concrete_int_(se, s_)) {
                mark_opaque("symbolic BVec_Slice"); break;
            }
            const auto& v = state.bvecs.back();
            int64_t n = static_cast<int64_t>(v.size());
            int64_t s = std::max<int64_t>(0, std::min<int64_t>(s_, n));
            int64_t en = std::max<int64_t>(s, std::min<int64_t>(e_, n));
            BVecSym sliced(v.begin() + s, v.begin() + en);
            state.bvecs.pop_back();
            state.bvecs.push_back(std::move(sliced));
        } break;
        case O::BVec_Reverse: {
            if (state.bvecs.empty()) break;
            auto& v = state.bvecs.back();
            std::reverse(v.begin(), v.end());
        } break;
        case O::BVec_Pop:  if (!state.bvecs.empty()) state.bvecs.pop_back(); break;
        case O::BVec_Dup:  if (!state.bvecs.empty()) state.bvecs.push_back(state.bvecs.back()); break;
        case O::BVec_Swap: if (state.bvecs.size() >= 2) std::swap(state.bvecs[state.bvecs.size()-1], state.bvecs[state.bvecs.size()-2]); break;

        // ============== IVec ==============
        case O::IVec_Len: {
            if (state.ivecs.empty()) break;
            push_i(make_lit_int(static_cast<int64_t>(state.ivecs.back().size())));
        } break;
        case O::IVec_At: {
            if (state.ivecs.empty() || state.ivecs.back().empty() || state.ints.empty()) break;
            ExprRef ie; pop_i(ie);
            const auto& v = state.ivecs.back();
            int64_t i;
            if (concrete_int_(ie, i)) {
                int k = static_cast<int>(mod_pos_(i, static_cast<int64_t>(v.size())));
                push_i(v[k]);
            } else {
                std::vector<int> atoms = ie->atoms;
                for (const auto& e : v) atoms = merge_atoms(atoms, e->atoms);
                push_i(make_opaque(std::move(atoms)));
            }
        } break;
        case O::IVec_Set: {
            if (state.ivecs.empty() || state.ivecs.back().empty() || state.ints.size() < 2) break;
            ExprRef val; pop_i(val);  // Python pops val first, then idx
            ExprRef ie; pop_i(ie);
            auto& v = state.ivecs.back();
            int64_t i;
            if (concrete_int_(ie, i)) {
                int k = static_cast<int>(mod_pos_(i, static_cast<int64_t>(v.size())));
                v[k] = val;
            } else {
                std::vector<int> atoms = ie->atoms;
                atoms = merge_atoms(atoms, val->atoms);
                for (auto& e : v) atoms = merge_atoms(atoms, e->atoms);
                ExprRef op = make_opaque(atoms);
                for (auto& e : v) e = op;
            }
        } break;
        case O::IVec_Push: {
            if (state.ivecs.empty() || state.ints.empty()) break;
            ExprRef val; pop_i(val);
            auto& v = state.ivecs.back();
            v.push_back(val);
            if (static_cast<int>(v.size()) > MAX_VEC_LEN) v.resize(MAX_VEC_LEN);
        } break;
        case O::IVec_PopBack: {
            if (state.ivecs.empty() || state.ivecs.back().empty()) break;
            ExprRef last = state.ivecs.back().back();
            state.ivecs.back().pop_back();
            push_i(last);
        } break;
        case O::IVec_New: {
            ExprRef ne; if (!pop_i(ne)) break;
            int64_t n;
            if (!concrete_int_(ne, n)) { mark_opaque("symbolic IVec_New"); break; }
            int nn = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(n, MAX_VEC_LEN)));
            state.ivecs.push_back(IVecSym(nn, make_lit_int(0)));
        } break;
        case O::IVec_Concat: {
            if (state.ivecs.size() < 2) break;
            IVecSym b = state.ivecs.back(); state.ivecs.pop_back();
            IVecSym a = state.ivecs.back(); state.ivecs.pop_back();
            IVecSym out; out.reserve(a.size() + b.size());
            out.insert(out.end(), a.begin(), a.end());
            out.insert(out.end(), b.begin(), b.end());
            if (static_cast<int>(out.size()) > MAX_VEC_LEN) out.resize(MAX_VEC_LEN);
            state.ivecs.push_back(std::move(out));
        } break;
        case O::IVec_Slice: {
            if (state.ivecs.empty() || state.ints.size() < 2) break;
            ExprRef ee; pop_i(ee);
            ExprRef se; pop_i(se);
            int64_t e_, s_;
            if (!concrete_int_(ee, e_) || !concrete_int_(se, s_)) {
                mark_opaque("symbolic IVec_Slice"); break;
            }
            const auto& v = state.ivecs.back();
            int64_t n = static_cast<int64_t>(v.size());
            int64_t s = std::max<int64_t>(0, std::min<int64_t>(s_, n));
            int64_t en = std::max<int64_t>(s, std::min<int64_t>(e_, n));
            IVecSym sliced(v.begin() + s, v.begin() + en);
            state.ivecs.pop_back();
            state.ivecs.push_back(std::move(sliced));
        } break;
        case O::IVec_Reverse: {
            if (state.ivecs.empty()) break;
            auto& v = state.ivecs.back();
            std::reverse(v.begin(), v.end());
        } break;
        case O::IVec_Pop:  if (!state.ivecs.empty()) state.ivecs.pop_back(); break;
        case O::IVec_Dup:  if (!state.ivecs.empty()) state.ivecs.push_back(state.ivecs.back()); break;
        case O::IVec_Swap: if (state.ivecs.size() >= 2) std::swap(state.ivecs[state.ivecs.size()-1], state.ivecs[state.ivecs.size()-2]); break;

        // ============== Memory ==============
        case O::Mem_Read: {
            ExprRef ie; if (!pop_i(ie)) break;
            int64_t i;
            if (!concrete_int_(ie, i)) { mark_opaque("symbolic Mem_Read"); break; }
            int k = static_cast<int>(mod_pos_(i, MEMORY_SIZE));
            push_f(state.memory[k]);
        } break;
        case O::Mem_Write: {
            if (state.ints.empty() || state.floats.empty()) break;
            ExprRef ie; pop_i(ie);
            ExprRef val; pop_f(val);
            int64_t i;
            if (!concrete_int_(ie, i)) { mark_opaque("symbolic Mem_Write"); break; }
            int k = static_cast<int>(mod_pos_(i, MEMORY_SIZE));
            state.memory[k] = val;
        } break;
        case O::Mem_ReadVec: {
            if (state.ints.size() < 2) break;
            ExprRef ee; pop_i(ee);
            ExprRef se; pop_i(se);
            int64_t e_, s_;
            if (!concrete_int_(ee, e_) || !concrete_int_(se, s_)) {
                mark_opaque("symbolic Mem_ReadVec"); break;
            }
            int64_t n = MEMORY_SIZE;
            int64_t s = std::max<int64_t>(0, std::min<int64_t>(s_, n));
            int64_t en = std::max<int64_t>(s, std::min<int64_t>(e_, n));
            FVecSym slice;
            for (int64_t i = s; i < en; ++i) slice.push_back(state.memory[static_cast<size_t>(i)]);
            state.fvecs.push_back(std::move(slice));
        } break;
        case O::Mem_WriteVec: {
            if (state.ints.empty() || state.fvecs.empty()) break;
            ExprRef se; pop_i(se);
            int64_t s_;
            if (!concrete_int_(se, s_)) { mark_opaque("symbolic Mem_WriteVec"); break; }
            FVecSym v = state.fvecs.back(); state.fvecs.pop_back();
            int64_t n = MEMORY_SIZE;
            int64_t s = std::max<int64_t>(0, std::min<int64_t>(s_, n));
            int64_t en = std::min<int64_t>(n, s + static_cast<int64_t>(v.size()));
            for (int64_t i = s; i < en; ++i)
                state.memory[static_cast<size_t>(i)] = v[static_cast<size_t>(i - s)];
        } break;

        // ============== Env (concrete) ==============
        case O::Env_GetChannelLLR: {
            if (ctx_has_channel_llr) push_f(make_atom(AtomKind::LV, 0));
        } break;
        case O::Env_GetIncomingVec: {
            FVecSym v;
            for (int i = 0; i < ctx_deg - 1; ++i) v.push_back(make_atom(AtomKind::X, i));
            state.fvecs.push_back(std::move(v));
        } break;
        case O::Env_GetNoiseVar:   push_f(make_lit_float(ctx_noise_var)); break;
        case O::Env_GetIter:       push_i(make_lit_int(ctx_iter)); break;
        case O::Env_GetMaxIter:    push_i(make_lit_int(ctx_max_iter)); break;
        case O::Env_GetDeg:        push_i(make_lit_int(ctx_deg)); break;
        case O::Env_GetEdgeIndex:  push_i(make_lit_int(ctx_edge_index)); break;
        case O::Env_GetCodeRate:   push_f(make_lit_float(ctx_code_rate)); break;

        // ============== Control flow ==============
        case O::Exec_If: {
            ExprRef cond; if (!pop_b(cond)) break;
            bool cv;
            if (concrete_bool_(cond, cv)) {
                const auto* block = cv ? ins.code_block.get() : ins.code_block2.get();
                if (block && !block->empty()) execute_block(*block);
            } else {
                // Symbolic condition: try element-wise merge first (no path
                // explosion).  If branches end up with mismatched stack
                // shapes, fall back to the forced-choice scheduler.
                auto snap = snapshot_();
                if (ins.code_block && !ins.code_block->empty())
                    execute_block(*ins.code_block);
                if (opaque) return;
                auto th = snapshot_();
                restore_(snap);
                if (ins.code_block2 && !ins.code_block2->empty())
                    execute_block(*ins.code_block2);
                if (opaque) return;
                auto el = snapshot_();
                if (merge_(th, el, cond)) break;
                // Diverged in shape — take a concrete branch from the scheduler.
                bool take;
                restore_(snap);
                if (!consume_branch_(cond, take)) return;
                if (take) {
                    if (ins.code_block && !ins.code_block->empty())
                        execute_block(*ins.code_block);
                } else {
                    if (ins.code_block2 && !ins.code_block2->empty())
                        execute_block(*ins.code_block2);
                }
            }
        } break;
        case O::Exec_When: {
            ExprRef cond; if (!pop_b(cond)) break;
            bool cv;
            if (concrete_bool_(cond, cv)) {
                if (cv && ins.code_block && !ins.code_block->empty())
                    execute_block(*ins.code_block);
            } else {
                if (!ins.code_block || ins.code_block->empty()) break;
                auto snap = snapshot_();
                execute_block(*ins.code_block);
                if (opaque) return;
                auto th = snapshot_();
                auto el = snap;
                restore_(snap);
                if (merge_(th, el, cond)) break;
                // Diverged — consult the scheduler for a concrete decision.
                bool take;
                if (!consume_branch_(cond, take)) return;
                if (take) execute_block(*ins.code_block);
                // else: state already restored to snap, nothing to do
            }
        } break;
        case O::Exec_DoTimes: {
            if (state.ints.empty() || !ins.code_block || ins.code_block->empty()) break;
            ExprRef ne; pop_i(ne);
            int64_t n;
            if (!concrete_int_(ne, n)) { mark_opaque("symbolic DoTimes bound"); break; }
            int nn = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(n, N_MAX_LOOP)));
            for (int i = 0; i < nn; ++i) {
                if (opaque) return;
                push_i(make_lit_int(i));
                execute_block(*ins.code_block);
            }
        } break;
        case O::Exec_DoRange: {
            if (state.ints.size() < 2 || !ins.code_block || ins.code_block->empty()) break;
            ExprRef ee; pop_i(ee);
            ExprRef se; pop_i(se);
            int64_t e_, s_;
            if (!concrete_int_(ee, e_) || !concrete_int_(se, s_)) {
                mark_opaque("symbolic DoRange bounds"); break;
            }
            if (e_ <= s_) break;
            int64_t span = std::min<int64_t>(N_MAX_LOOP, e_ - s_);
            for (int64_t i = s_; i < s_ + span; ++i) {
                if (opaque) return;
                push_i(make_lit_int(i));
                execute_block(*ins.code_block);
            }
        } break;
        case O::Exec_While: {
            if (!ins.code_block || ins.code_block->empty()) break;
            for (int it = 0; it < N_MAX_WHILE; ++it) {
                if (opaque) return;
                if (state.bools.empty()) return;
                ExprRef cond; pop_b(cond);
                bool cv;
                if (concrete_bool_(cond, cv)) {
                    if (!cv) return;
                } else {
                    // Symbolic loop condition: one scheduler bit per iteration
                    // tells us whether the loop continues.  Bounded by
                    // N_MAX_WHILE so the choice vector remains finite.
                    bool take;
                    if (!consume_branch_(cond, take)) return;
                    if (!take) return;
                }
                execute_block(*ins.code_block);
            }
        } break;

        default: break;
        }
    }
};

}  // namespace pushgp_cpp
