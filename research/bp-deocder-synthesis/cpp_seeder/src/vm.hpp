// vm.hpp — Push-GP VM matching pushgp/vm.py + instructions.py semantics.
// All 127 instruction handlers are inlined here so the dispatch is a single switch.
#pragma once

#include "common.hpp"
#include "instruction.hpp"
#include "opcodes.hpp"
#include "vm_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pushgp_cpp {

// ---------- numeric guards ----------
inline double safe_float(double x) {
    if (std::isnan(x) || std::isinf(x)) return NAN_INF_REPLACEMENT;
    if (x >  FLOAT_ABS_MAX) return  FLOAT_ABS_MAX;
    if (x < -FLOAT_ABS_MAX) return -FLOAT_ABS_MAX;
    return x;
}
inline int64_t clamp_int(int64_t x) {
    if (x >  INT_ABS_MAX) return  INT_ABS_MAX;
    if (x < -INT_ABS_MAX) return -INT_ABS_MAX;
    return x;
}
inline void sanitize_fvec(FVec& v) {
    for (auto& x : v) x = safe_float(x);
}

class VM {
   public:
    VMState state;

    int step_max  = VM_STEP_MAX;
    int flop_max  = VM_FLOP_MAX;
    int recur_max = VM_RECUR_MAX;

    // run a top-level program; returns top of float stack or NaN on fault.
    // Returns true on success and writes to `out`.  False on fault / non-finite top.
    bool run(const Program& prog, double& out) {
        state.reset_stacks();
        recur_depth_ = 0;
        execute_block(prog);
        if (state.fault) return false;
        const double* top = state.floats.peek();
        if (!top) return false;
        if (!std::isfinite(*top)) return false;
        out = *top;
        return true;
    }

    bool aborted() const { return state.fault; }
    void charge_flops(int n) {
        state.flop_count += n;
        if (state.flop_count > flop_max) abort_("flop_max exceeded");
    }

    void execute_block(const Program& block) {
        if (recur_depth_ >= recur_max) {
            abort_("recur_max exceeded");
            return;
        }
        ++recur_depth_;
        for (const auto& ins : block) {
            if (state.fault) break;
            step_(ins);
        }
        --recur_depth_;
    }

   private:
    int recur_depth_ = 0;

    void abort_(const char* reason) {
        state.fault = true;
        if (state.fault_reason.empty()) state.fault_reason = reason;
    }

    void step_(const Instruction& ins) {
        ++state.step_count;
        if (state.step_count > step_max) { abort_("step_max exceeded"); return; }
        // Wrap dispatch in try-catch to mirror Python's "never crash on handler exception"
        try {
            dispatch_(ins);
        } catch (...) {
            abort_("handler exception");
        }
    }

    // ---- helpers ----
    bool pop_float_(double& v) { return state.floats.pop(v); }
    bool pop_int_(int64_t& v)  { return state.ints.pop(v); }
    bool pop_bool_(uint8_t& v) { return state.bools.pop(v); }

    // ----------------------------------------------------------------
    // The big dispatch.
    // ----------------------------------------------------------------
    void dispatch_(const Instruction& ins) {
        using O = Op;
        switch (ins.op) {
        // ============== Float arith ==============
        case O::Float_Add: f_binop_([](double a, double b){ return a + b; }); break;
        case O::Float_Sub: f_binop_([](double a, double b){ return a - b; }); break;
        case O::Float_Mul: f_binop_([](double a, double b){ return a * b; }); break;
        case O::Float_Div: f_binop_([](double a, double b){
            return b != 0.0 ? a / b : NAN_INF_REPLACEMENT; }); break;
        case O::Float_Mod: f_binop_([](double a, double b){
            return b != 0.0 ? std::fmod(a, b) : NAN_INF_REPLACEMENT; }); break;
        case O::Float_Min: f_binop_([](double a, double b){ return std::min(a, b); }); break;
        case O::Float_Max: f_binop_([](double a, double b){ return std::max(a, b); }); break;

        case O::Float_Abs:    f_unop_([](double a){ return std::fabs(a); }); break;
        case O::Float_Neg:    f_unop_([](double a){ return -a; }); break;
        case O::Float_Inv:    f_unop_([](double a){
            return a != 0.0 ? 1.0 / a : NAN_INF_REPLACEMENT; }); break;
        case O::Float_Sqrt:   f_unop_([](double a){
            return a >= 0.0 ? std::sqrt(a) : NAN_INF_REPLACEMENT; }); break;
        case O::Float_Square: f_unop_([](double a){ return a * a; }); break;
        case O::Float_Exp:    f_unop_([](double a){
            return a < EXP_INPUT_CAP ? std::exp(a) : NAN_INF_REPLACEMENT; }); break;
        case O::Float_Log:    f_unop_([](double a){
            return a > 0.0 ? std::log(a) : NAN_INF_REPLACEMENT; }); break;
        case O::Float_Tanh:   f_unop_([](double a){ return std::tanh(a); }); break;
        case O::Float_Atanh:  f_unop_([](double a){
            return (a > -1.0 && a < 1.0) ? std::atanh(a) : NAN_INF_REPLACEMENT; }); break;
        case O::Float_Sign:   f_unop_([](double a){
            return a > 0 ? 1.0 : (a < 0 ? -1.0 : 0.0); }); break;
        case O::Float_Floor:  f_unop_([](double a){
            // Python uses math.floor which returns int, collapsing -0.0 to 0.
            double r = std::floor(a);
            if (r >= -1e18 && r <= 1e18) r = static_cast<double>(static_cast<int64_t>(r));
            return r;
        }); break;
        case O::Float_Ceil:   f_unop_([](double a){
            double r = std::ceil(a);
            if (r >= -1e18 && r <= 1e18) r = static_cast<double>(static_cast<int64_t>(r));
            return r;
        }); break;

        // ============== Float consts ==============
        case O::Float_Const0:    push_f_(0.0); break;
        case O::Float_Const1:    push_f_(1.0); break;
        case O::Float_ConstNeg1: push_f_(-1.0); break;
        case O::Float_ConstHalf: push_f_(0.5); break;
        case O::Float_Const2:    push_f_(2.0); break;
        case O::Float_Const0_1:  push_f_(0.1); break;
        case O::Float_ConstPi:   push_f_(3.14159265358979323846); break;
        case O::Float_Const1eNeg6: push_f_(1e-6); break;
        case O::Float_EvoConst0: push_f_(state.ctx_evo_constants[0]); break;
        case O::Float_EvoConst1: push_f_(state.ctx_evo_constants[1]); break;
        case O::Float_EvoConst2: push_f_(state.ctx_evo_constants[2]); break;
        case O::Float_EvoConst3: push_f_(state.ctx_evo_constants[3]); break;
        case O::Float_EvoConst4: push_f_(state.ctx_evo_constants[4]); break;
        case O::Float_EvoConst5: push_f_(state.ctx_evo_constants[5]); break;
        case O::Float_EvoConst6: push_f_(state.ctx_evo_constants[6]); break;
        case O::Float_EvoConst7: push_f_(state.ctx_evo_constants[7]); break;

        // ============== Float stack ==============
        case O::Float_Pop:   state.floats.drop_top(); break;
        case O::Float_Dup:   state.floats.dup(); break;
        case O::Float_Swap:  state.floats.swap(); break;
        case O::Float_Rot:   state.floats.rot(); break;
        case O::Float_Yank:  { int64_t n; if (pop_int_(n)) state.floats.yank(static_cast<int>(n)); } break;
        case O::Float_Shove: { int64_t n; if (pop_int_(n)) state.floats.shove(static_cast<int>(n)); } break;

        // ============== Int arith ==============
        case O::Int_Add: i_binop_([](int64_t a, int64_t b){ return a + b; }); break;
        case O::Int_Sub: i_binop_([](int64_t a, int64_t b){ return a - b; }); break;
        case O::Int_Mul: i_binop_([](int64_t a, int64_t b){ return a * b; }); break;
        case O::Int_Div: i_binop_([](int64_t a, int64_t b){
            // Python's // (floor div). For 0, return 0.
            if (b == 0) return int64_t{0};
            int64_t q = a / b;
            int64_t r = a % b;
            if ((r != 0) && ((r < 0) != (b < 0))) --q;  // floor semantics
            return q;
        }); break;
        case O::Int_Mod: i_binop_([](int64_t a, int64_t b){
            if (b == 0) return int64_t{0};
            int64_t r = a % b;
            if ((r != 0) && ((r < 0) != (b < 0))) r += b;  // python %
            return r;
        }); break;
        case O::Int_Min: i_binop_([](int64_t a, int64_t b){ return std::min(a, b); }); break;
        case O::Int_Max: i_binop_([](int64_t a, int64_t b){ return std::max(a, b); }); break;
        case O::Int_Inc: i_unop_([](int64_t a){ return a + 1; }); break;
        case O::Int_Dec: i_unop_([](int64_t a){ return a - 1; }); break;
        case O::Int_Neg: i_unop_([](int64_t a){ return -a; }); break;

        // Int consts
        case O::Int_Const0:    state.ints.push(0); break;
        case O::Int_Const1:    state.ints.push(1); break;
        case O::Int_Const2:    state.ints.push(2); break;
        case O::Int_ConstNeg1: state.ints.push(-1); break;

        // Int stack
        case O::Int_Pop:  state.ints.drop_top(); break;
        case O::Int_Dup:  state.ints.dup(); break;
        case O::Int_Swap: state.ints.swap(); break;
        case O::Int_Rot:  state.ints.rot(); break;

        // ============== Bool ==============
        case O::Bool_And: b_binop_([](bool a, bool b){ return a && b; }); break;
        case O::Bool_Or:  b_binop_([](bool a, bool b){ return a || b; }); break;
        case O::Bool_Xor: b_binop_([](bool a, bool b){ return a != b; }); break;
        case O::Bool_Not: { uint8_t v; if (pop_bool_(v)) state.bools.push(v ? 0 : 1); } break;
        case O::Bool_True:  state.bools.push(1); break;
        case O::Bool_False: state.bools.push(0); break;
        case O::Bool_Pop:   state.bools.drop_top(); break;
        case O::Bool_Dup:   state.bools.dup(); break;
        case O::Bool_Swap:  state.bools.swap(); break;

        // ============== Compare ==============
        case O::Float_LT: cmp_f_([](double a, double b){ return a <  b; }); break;
        case O::Float_GT: cmp_f_([](double a, double b){ return a >  b; }); break;
        case O::Float_EQ: cmp_f_([](double a, double b){
            return std::fabs(a - b) < FLOAT_EQ_EPSILON; }); break;
        case O::Int_LT: cmp_i_([](int64_t a, int64_t b){ return a <  b; }); break;
        case O::Int_GT: cmp_i_([](int64_t a, int64_t b){ return a >  b; }); break;
        case O::Int_EQ: cmp_i_([](int64_t a, int64_t b){ return a == b; }); break;

        // ============== Conversions ==============
        case O::Float_FromInt: { int64_t v; if (pop_int_(v))  push_f_(static_cast<double>(v)); } break;
        case O::Int_FromFloat: { double v; if (pop_float_(v)) {
            // Python int(v) truncates toward zero; raises on NaN/Inf → 0.
            if (!std::isfinite(v)) state.ints.push(0);
            else state.ints.push(clamp_int(static_cast<int64_t>(std::trunc(v))));
        } } break;
        case O::Int_FromBool: { uint8_t v; if (pop_bool_(v)) state.ints.push(v ? 1 : 0); } break;
        case O::Bool_FromFloat: { double v; if (pop_float_(v)) state.bools.push(v > 0.0 ? 1 : 0); } break;
        case O::Bool_FromInt:   { int64_t v; if (pop_int_(v))  state.bools.push(v != 0 ? 1 : 0); } break;

        // ============== FVec ==============
        case O::FVec_Len: {
            const FVec* v = state.fvecs.peek();
            if (v) state.ints.push(static_cast<int64_t>(v->size()));
        } break;
        case O::FVec_At: {
            const FVec* v = state.fvecs.peek();
            if (!v || v->empty()) break;
            int64_t idx; if (!pop_int_(idx)) break;
            int64_t i = mod_pos_(idx, static_cast<int64_t>(v->size()));
            push_f_((*v)[static_cast<size_t>(i)]);
        } break;
        case O::FVec_Set: {
            FVec* v = state.fvecs.peek_mut();
            if (!v || v->empty()) break;
            if (state.ints.empty() || state.floats.empty()) break;
            int64_t idx; pop_int_(idx);
            double  val; pop_float_(val);
            int64_t i = mod_pos_(idx, static_cast<int64_t>(v->size()));
            // Python: copy then replace top.  We replicate by mutating in place
            // (semantically equivalent because Python immediately replaces top).
            (*v)[static_cast<size_t>(i)] = safe_float(val);
        } break;
        case O::FVec_Push: {
            FVec* v = state.fvecs.peek_mut();
            if (!v) break;
            if (state.floats.empty()) break;
            double val; pop_float_(val);
            v->push_back(safe_float(val));
            if (static_cast<int>(v->size()) > MAX_VEC_LEN) v->resize(MAX_VEC_LEN);
        } break;
        case O::FVec_PopBack: {
            const FVec* v = state.fvecs.peek();
            if (!v || v->empty()) break;
            double val = v->back();
            FVec new_v(v->begin(), v->end() - 1);
            state.fvecs.drop_top();
            state.fvecs.push(std::move(new_v));
            push_f_(val);
        } break;
        case O::FVec_New: {
            int64_t n; if (!pop_int_(n)) break;
            int nn = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(n, MAX_VEC_LEN)));
            state.fvecs.push(FVec(nn, 0.0));
        } break;
        case O::FVec_FromFloat: {
            double v; if (!pop_float_(v)) break;
            state.fvecs.push(FVec{safe_float(v)});
        } break;
        case O::FVec_Concat: {
            if (state.fvecs.depth() < 2) break;
            FVec b; state.fvecs.pop(b);
            FVec a; state.fvecs.pop(a);
            FVec out;
            out.reserve(std::min<size_t>(a.size() + b.size(), MAX_VEC_LEN));
            out.insert(out.end(), a.begin(), a.end());
            out.insert(out.end(), b.begin(), b.end());
            if (static_cast<int>(out.size()) > MAX_VEC_LEN) out.resize(MAX_VEC_LEN);
            state.fvecs.push(std::move(out));
        } break;
        case O::FVec_Slice: {
            const FVec* v = state.fvecs.peek();
            if (!v || state.ints.depth() < 2) break;
            int64_t end_; pop_int_(end_);
            int64_t start; pop_int_(start);
            int64_t n = static_cast<int64_t>(v->size());
            int64_t s = std::max<int64_t>(0, std::min<int64_t>(start, n));
            int64_t e = std::max<int64_t>(s, std::min<int64_t>(end_, n));
            FVec sliced(v->begin() + s, v->begin() + e);
            state.fvecs.drop_top();
            state.fvecs.push(std::move(sliced));
        } break;
        case O::FVec_Reverse: {
            FVec* v = state.fvecs.peek_mut();
            if (!v) break;
            std::reverse(v->begin(), v->end());
        } break;
        case O::FVec_Roll: {
            // Match Python order: bail if peek is None or ints depth < 1.
            // Otherwise pop k FIRST, then bail if v.size == 0.
            const FVec* vp = state.fvecs.peek();
            if (!vp || state.ints.empty()) break;
            int64_t k; pop_int_(k);
            if (vp->empty()) break;
            FVec* v = state.fvecs.peek_mut();
            int64_t n = static_cast<int64_t>(v->size());
            int64_t shift = mod_pos_(k, n);
            if (shift == 0) break;
            FVec rolled(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                rolled[(i + shift) % n] = (*v)[static_cast<size_t>(i)];
            }
            *v = std::move(rolled);
        } break;
        case O::FVec_Pop:  state.fvecs.drop_top(); break;
        case O::FVec_Dup:  state.fvecs.dup(); break;
        case O::FVec_Swap: state.fvecs.swap(); break;
        case O::FVec_Rot:  state.fvecs.rot(); break;

        // ============== BVec ==============
        case O::BVec_Len: {
            const BVec* v = state.bvecs.peek();
            if (v) state.ints.push(static_cast<int64_t>(v->size()));
        } break;
        case O::BVec_At: {
            const BVec* v = state.bvecs.peek();
            if (!v || v->empty() || state.ints.empty()) break;
            int64_t idx; pop_int_(idx);
            int64_t i = mod_pos_(idx, static_cast<int64_t>(v->size()));
            state.bools.push((*v)[static_cast<size_t>(i)] ? 1 : 0);
        } break;
        case O::BVec_Set: {
            BVec* v = state.bvecs.peek_mut();
            if (!v || v->empty()) break;
            if (state.ints.empty() || state.bools.empty()) break;
            int64_t idx; pop_int_(idx);
            uint8_t val; pop_bool_(val);
            int64_t i = mod_pos_(idx, static_cast<int64_t>(v->size()));
            (*v)[static_cast<size_t>(i)] = val ? 1 : 0;
        } break;
        case O::BVec_Push: {
            BVec* v = state.bvecs.peek_mut();
            if (!v || state.bools.empty()) break;
            uint8_t val; pop_bool_(val);
            v->push_back(val ? 1 : 0);
            if (static_cast<int>(v->size()) > MAX_VEC_LEN) v->resize(MAX_VEC_LEN);
        } break;
        case O::BVec_PopBack: {
            const BVec* v = state.bvecs.peek();
            if (!v || v->empty()) break;
            uint8_t val = v->back();
            BVec new_v(v->begin(), v->end() - 1);
            state.bvecs.drop_top();
            state.bvecs.push(std::move(new_v));
            state.bools.push(val ? 1 : 0);
        } break;
        case O::BVec_New: {
            int64_t n; if (!pop_int_(n)) break;
            int nn = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(n, MAX_VEC_LEN)));
            state.bvecs.push(BVec(nn, 0));
        } break;
        case O::BVec_Concat: {
            if (state.bvecs.depth() < 2) break;
            BVec b; state.bvecs.pop(b);
            BVec a; state.bvecs.pop(a);
            BVec out;
            out.reserve(std::min<size_t>(a.size() + b.size(), MAX_VEC_LEN));
            out.insert(out.end(), a.begin(), a.end());
            out.insert(out.end(), b.begin(), b.end());
            if (static_cast<int>(out.size()) > MAX_VEC_LEN) out.resize(MAX_VEC_LEN);
            state.bvecs.push(std::move(out));
        } break;
        case O::BVec_Slice: {
            const BVec* v = state.bvecs.peek();
            if (!v || state.ints.depth() < 2) break;
            int64_t end_; pop_int_(end_);
            int64_t start; pop_int_(start);
            int64_t n = static_cast<int64_t>(v->size());
            int64_t s = std::max<int64_t>(0, std::min<int64_t>(start, n));
            int64_t e = std::max<int64_t>(s, std::min<int64_t>(end_, n));
            BVec sliced(v->begin() + s, v->begin() + e);
            state.bvecs.drop_top();
            state.bvecs.push(std::move(sliced));
        } break;
        case O::BVec_Reverse: {
            BVec* v = state.bvecs.peek_mut();
            if (!v) break;
            std::reverse(v->begin(), v->end());
        } break;
        case O::BVec_Pop:  state.bvecs.drop_top(); break;
        case O::BVec_Dup:  state.bvecs.dup(); break;
        case O::BVec_Swap: state.bvecs.swap(); break;

        // ============== IVec ==============
        case O::IVec_Len: {
            const IVec* v = state.ivecs.peek();
            if (v) state.ints.push(static_cast<int64_t>(v->size()));
        } break;
        case O::IVec_At: {
            const IVec* v = state.ivecs.peek();
            if (!v || v->empty() || state.ints.empty()) break;
            int64_t idx; pop_int_(idx);
            int64_t i = mod_pos_(idx, static_cast<int64_t>(v->size()));
            state.ints.push((*v)[static_cast<size_t>(i)]);
        } break;
        case O::IVec_Set: {
            IVec* v = state.ivecs.peek_mut();
            if (!v || v->empty()) break;
            if (state.ints.depth() < 2) break;
            // Python pops val first, then idx.
            int64_t val; pop_int_(val);
            int64_t idx; pop_int_(idx);
            int64_t i = mod_pos_(idx, static_cast<int64_t>(v->size()));
            (*v)[static_cast<size_t>(i)] = val;
        } break;
        case O::IVec_Push: {
            IVec* v = state.ivecs.peek_mut();
            if (!v || state.ints.empty()) break;
            int64_t val; pop_int_(val);
            v->push_back(val);
            if (static_cast<int>(v->size()) > MAX_VEC_LEN) v->resize(MAX_VEC_LEN);
        } break;
        case O::IVec_PopBack: {
            const IVec* v = state.ivecs.peek();
            if (!v || v->empty()) break;
            int64_t val = v->back();
            IVec new_v(v->begin(), v->end() - 1);
            state.ivecs.drop_top();
            state.ivecs.push(std::move(new_v));
            state.ints.push(val);
        } break;
        case O::IVec_New: {
            int64_t n; if (!pop_int_(n)) break;
            int nn = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(n, MAX_VEC_LEN)));
            state.ivecs.push(IVec(nn, 0));
        } break;
        case O::IVec_Concat: {
            if (state.ivecs.depth() < 2) break;
            IVec b; state.ivecs.pop(b);
            IVec a; state.ivecs.pop(a);
            IVec out;
            out.reserve(std::min<size_t>(a.size() + b.size(), MAX_VEC_LEN));
            out.insert(out.end(), a.begin(), a.end());
            out.insert(out.end(), b.begin(), b.end());
            if (static_cast<int>(out.size()) > MAX_VEC_LEN) out.resize(MAX_VEC_LEN);
            state.ivecs.push(std::move(out));
        } break;
        case O::IVec_Slice: {
            const IVec* v = state.ivecs.peek();
            if (!v || state.ints.depth() < 2) break;
            int64_t end_; pop_int_(end_);
            int64_t start; pop_int_(start);
            int64_t n = static_cast<int64_t>(v->size());
            int64_t s = std::max<int64_t>(0, std::min<int64_t>(start, n));
            int64_t e = std::max<int64_t>(s, std::min<int64_t>(end_, n));
            IVec sliced(v->begin() + s, v->begin() + e);
            state.ivecs.drop_top();
            state.ivecs.push(std::move(sliced));
        } break;
        case O::IVec_Reverse: {
            IVec* v = state.ivecs.peek_mut();
            if (!v) break;
            std::reverse(v->begin(), v->end());
        } break;
        case O::IVec_Pop:  state.ivecs.drop_top(); break;
        case O::IVec_Dup:  state.ivecs.dup(); break;
        case O::IVec_Swap: state.ivecs.swap(); break;

        // ============== Memory ==============
        case O::Mem_Read: {
            int64_t idx; if (!pop_int_(idx)) break;
            int64_t i = mod_pos_(idx, MEMORY_SIZE);
            push_f_(state.memory[static_cast<size_t>(i)]);
        } break;
        case O::Mem_Write: {
            if (state.ints.empty() || state.floats.empty()) break;
            int64_t idx; pop_int_(idx);
            double  val; pop_float_(val);
            int64_t i = mod_pos_(idx, MEMORY_SIZE);
            state.memory[static_cast<size_t>(i)] = safe_float(val);
        } break;
        case O::Mem_ReadVec: {
            if (state.ints.depth() < 2) break;
            int64_t end_; pop_int_(end_);
            int64_t start; pop_int_(start);
            int64_t n = MEMORY_SIZE;
            int64_t s = std::max<int64_t>(0, std::min<int64_t>(start, n));
            int64_t e = std::max<int64_t>(s, std::min<int64_t>(end_, n));
            FVec slice(state.memory.begin() + s, state.memory.begin() + e);
            state.fvecs.push(std::move(slice));
        } break;
        case O::Mem_WriteVec: {
            if (state.ints.empty() || state.fvecs.empty()) break;
            int64_t start; pop_int_(start);
            FVec v; state.fvecs.pop(v);
            int64_t n = MEMORY_SIZE;
            int64_t s = std::max<int64_t>(0, std::min<int64_t>(start, n));
            int64_t e = std::min<int64_t>(n, s + static_cast<int64_t>(v.size()));
            for (int64_t i = s; i < e; ++i) {
                state.memory[static_cast<size_t>(i)] = safe_float(v[static_cast<size_t>(i - s)]);
            }
        } break;

        // ============== Env ==============
        case O::Env_GetChannelLLR: {
            if (state.ctx_has_channel_llr) push_f_(state.ctx_channel_llr);
        } break;
        case O::Env_GetIncomingVec: {
            FVec copy = state.ctx_incoming;
            sanitize_fvec(copy);
            state.fvecs.push(std::move(copy));
        } break;
        case O::Env_GetNoiseVar:   push_f_(state.ctx_noise_var); break;
        case O::Env_GetIter:       state.ints.push(state.ctx_iter); break;
        case O::Env_GetMaxIter:    state.ints.push(state.ctx_max_iter); break;
        case O::Env_GetDeg:        state.ints.push(state.ctx_deg); break;
        case O::Env_GetEdgeIndex:  state.ints.push(state.ctx_edge_index); break;
        case O::Env_GetCodeRate:   push_f_(state.ctx_code_rate); break;

        // ============== Control flow ==============
        case O::Exec_If: {
            uint8_t cond; if (!pop_bool_(cond)) break;
            const auto* block = cond ? ins.code_block.get() : ins.code_block2.get();
            if (block && !block->empty()) execute_block(*block);
        } break;
        case O::Exec_When: {
            uint8_t cond; if (!pop_bool_(cond)) break;
            if (cond && ins.code_block && !ins.code_block->empty()) {
                execute_block(*ins.code_block);
            }
        } break;
        case O::Exec_DoTimes: {
            if (state.ints.empty() || !ins.code_block) break;
            int64_t n; pop_int_(n);
            int nn = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(n, N_MAX_LOOP)));
            for (int i = 0; i < nn; ++i) {
                if (state.fault) return;
                state.ints.push(i);
                execute_block(*ins.code_block);
            }
        } break;
        case O::Exec_DoRange: {
            if (state.ints.depth() < 2 || !ins.code_block) break;
            int64_t end_; pop_int_(end_);
            int64_t start; pop_int_(start);
            if (end_ <= start) break;
            int64_t span = std::min<int64_t>(N_MAX_LOOP, end_ - start);
            for (int64_t i = start; i < start + span; ++i) {
                if (state.fault) return;
                state.ints.push(i);
                execute_block(*ins.code_block);
            }
        } break;
        case O::Exec_While: {
            if (!ins.code_block) break;
            for (int it = 0; it < N_MAX_WHILE; ++it) {
                if (state.fault) return;
                if (state.bools.empty()) return;
                uint8_t cond; pop_bool_(cond);
                if (!cond) return;
                execute_block(*ins.code_block);
            }
        } break;

        case O::_COUNT: break;  // unreachable
        }
    }

    // Generic helpers used by handlers
    void push_f_(double v) { state.floats.push(safe_float(v)); }

    template <class F>
    void f_binop_(F op) {
        if (state.floats.depth() < 2) return;
        double b; pop_float_(b);
        double a; pop_float_(a);
        double r;
        try { r = op(a, b); } catch (...) { r = NAN_INF_REPLACEMENT; }
        push_f_(r);
        charge_flops(1);
    }
    template <class F>
    void f_unop_(F op) {
        if (state.floats.empty()) return;
        double a; pop_float_(a);
        double r;
        try { r = op(a); } catch (...) { r = NAN_INF_REPLACEMENT; }
        push_f_(r);
        charge_flops(1);
    }
    template <class F>
    void i_binop_(F op) {
        if (state.ints.depth() < 2) return;
        int64_t b; pop_int_(b);
        int64_t a; pop_int_(a);
        int64_t r = op(a, b);
        state.ints.push(clamp_int(r));
        charge_flops(1);
    }
    template <class F>
    void i_unop_(F op) {
        if (state.ints.empty()) return;
        int64_t a; pop_int_(a);
        int64_t r = op(a);
        state.ints.push(clamp_int(r));
        charge_flops(1);
    }
    template <class F>
    void b_binop_(F op) {
        if (state.bools.depth() < 2) return;
        uint8_t b; pop_bool_(b);
        uint8_t a; pop_bool_(a);
        state.bools.push(op(a != 0, b != 0) ? 1 : 0);
    }
    template <class F>
    void cmp_f_(F op) {
        if (state.floats.depth() < 2) return;
        double b; pop_float_(b);
        double a; pop_float_(a);
        state.bools.push(op(a, b) ? 1 : 0);
    }
    template <class F>
    void cmp_i_(F op) {
        if (state.ints.depth() < 2) return;
        int64_t b; pop_int_(b);
        int64_t a; pop_int_(a);
        state.bools.push(op(a, b) ? 1 : 0);
    }

    // Python-style modulo: result has same sign as divisor.
    static int64_t mod_pos_(int64_t x, int64_t n) {
        if (n == 0) return 0;
        int64_t r = x % n;
        if ((r != 0) && ((r < 0) != (n < 0))) r += n;
        return r;
    }
};

}  // namespace pushgp_cpp
