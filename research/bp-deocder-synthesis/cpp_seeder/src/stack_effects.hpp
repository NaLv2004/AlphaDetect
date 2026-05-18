// stack_effects.hpp — per-opcode stack pop/push requirements.
//
// Used by stack-aware random program generation in rpg.hpp:
//   * pops are MINIMUM stack depths needed to do meaningful work
//     (matches the early-bail checks in vm.hpp dispatch).
//   * pushes are the net counts pushed by a SUCCESSFUL execution.
//
// For control-flow ops the listed pops are what the op itself consumes
// BEFORE entering the body; the body's contribution is generated
// recursively and is treated as stack-neutral at the parent scope
// (a deliberate simplification — the inner generator still tracks its
// own virtual stacks).
//
// Some ops have value-level prerequisites (e.g. FVec.At needs len(v)>0
// in addition to the FVec being on the stack).  Those residual failures
// are accepted; the validator still catches them.
#pragma once

#include "opcodes.hpp"

#include <array>
#include <cstdint>

namespace pushgp_cpp {

struct StackEffect {
    int8_t f_pop  = 0, i_pop  = 0, b_pop  = 0;
    int8_t fv_pop = 0, bv_pop = 0, iv_pop = 0;
    int8_t f_push = 0, i_push = 0, b_push = 0;
    int8_t fv_push= 0, bv_push= 0, iv_push= 0;
    bool needs_channel_llr = false;   // Env.GetChannelLLR no-ops in c2v
    bool is_control        = false;   // has a code_block to recurse into
};

namespace detail {

inline constexpr StackEffect make_eff_(
    int f_pop, int i_pop, int b_pop, int fv_pop, int bv_pop, int iv_pop,
    int f_push, int i_push, int b_push, int fv_push, int bv_push, int iv_push,
    bool needs_llr = false, bool is_ctrl = false)
{
    StackEffect e;
    e.f_pop  = (int8_t)f_pop;  e.i_pop  = (int8_t)i_pop;  e.b_pop  = (int8_t)b_pop;
    e.fv_pop = (int8_t)fv_pop; e.bv_pop = (int8_t)bv_pop; e.iv_pop = (int8_t)iv_pop;
    e.f_push = (int8_t)f_push; e.i_push = (int8_t)i_push; e.b_push = (int8_t)b_push;
    e.fv_push= (int8_t)fv_push;e.bv_push= (int8_t)bv_push;e.iv_push= (int8_t)iv_push;
    e.needs_channel_llr = needs_llr;
    e.is_control = is_ctrl;
    return e;
}

// Macros to keep the table readable.  Columns:  f i b fv bv iv  pops, then pushes.
#define EFF(fp,ip,bp,fvp,bvp,ivp,  fpu,ipu,bpu,fvpu,bvpu,ivpu) \
    make_eff_(fp,ip,bp,fvp,bvp,ivp, fpu,ipu,bpu,fvpu,bvpu,ivpu, false, false)
#define EFF_LLR(fp,ip,bp,fvp,bvp,ivp,  fpu,ipu,bpu,fvpu,bvpu,ivpu) \
    make_eff_(fp,ip,bp,fvp,bvp,ivp, fpu,ipu,bpu,fvpu,bvpu,ivpu, true,  false)
#define EFF_CTRL(fp,ip,bp,fvp,bvp,ivp,  fpu,ipu,bpu,fvpu,bvpu,ivpu) \
    make_eff_(fp,ip,bp,fvp,bvp,ivp, fpu,ipu,bpu,fvpu,bvpu,ivpu, false, true)

inline const std::array<StackEffect, OP_COUNT>& build_table() {
    static const std::array<StackEffect, OP_COUNT> T = []() {
        std::array<StackEffect, OP_COUNT> t{};
        // ---- Float arith (binops pop 2 push 1; unops pop 1 push 1) ----
        t[(int)Op::Float_Add] = EFF(2,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Sub] = EFF(2,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Mul] = EFF(2,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Div] = EFF(2,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Mod] = EFF(2,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Min] = EFF(2,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Max] = EFF(2,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Abs] = EFF(1,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Neg] = EFF(1,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Inv] = EFF(1,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Sqrt]= EFF(1,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Square]=EFF(1,0,0,0,0,0,1,0,0,0,0,0);
        t[(int)Op::Float_Exp] = EFF(1,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Log] = EFF(1,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Tanh]= EFF(1,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Atanh]=EFF(1,0,0,0,0,0,1,0,0,0,0,0);
        t[(int)Op::Float_Sign]= EFF(1,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Floor]=EFF(1,0,0,0,0,0,1,0,0,0,0,0);
        t[(int)Op::Float_Ceil]= EFF(1,0,0,0,0,0, 1,0,0,0,0,0);

        // ---- Float constants (no pops, push 1 float) ----
        t[(int)Op::Float_Const0]    = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Const1]    = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_ConstNeg1] = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_ConstHalf] = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Const2]    = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Const0_1]  = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_ConstPi]   = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_Const1eNeg6]=EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_EvoConst0] = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_EvoConst1] = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_EvoConst2] = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_EvoConst3] = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_EvoConst4] = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_EvoConst5] = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_EvoConst6] = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Float_EvoConst7] = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);

        // ---- Float stack ops ----
        t[(int)Op::Float_Pop]  = EFF(1,0,0,0,0,0, 0,0,0,0,0,0);
        t[(int)Op::Float_Dup]  = EFF(1,0,0,0,0,0, 2,0,0,0,0,0);  // peeks then push
        t[(int)Op::Float_Swap] = EFF(2,0,0,0,0,0, 2,0,0,0,0,0);
        t[(int)Op::Float_Rot]  = EFF(3,0,0,0,0,0, 3,0,0,0,0,0);
        t[(int)Op::Float_Yank] = EFF(0,1,0,0,0,0, 0,0,0,0,0,0);  // pops int; rearranges floats
        t[(int)Op::Float_Shove]= EFF(0,1,0,0,0,0, 0,0,0,0,0,0);

        // ---- Int arith ----
        t[(int)Op::Int_Add] = EFF(0,2,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_Sub] = EFF(0,2,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_Mul] = EFF(0,2,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_Div] = EFF(0,2,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_Mod] = EFF(0,2,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_Min] = EFF(0,2,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_Max] = EFF(0,2,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_Inc] = EFF(0,1,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_Dec] = EFF(0,1,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_Neg] = EFF(0,1,0,0,0,0, 0,1,0,0,0,0);

        // ---- Int constants ----
        t[(int)Op::Int_Const0]    = EFF(0,0,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_Const1]    = EFF(0,0,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_Const2]    = EFF(0,0,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_ConstNeg1] = EFF(0,0,0,0,0,0, 0,1,0,0,0,0);

        // ---- Int stack ----
        t[(int)Op::Int_Pop]  = EFF(0,1,0,0,0,0, 0,0,0,0,0,0);
        t[(int)Op::Int_Dup]  = EFF(0,1,0,0,0,0, 0,2,0,0,0,0);
        t[(int)Op::Int_Swap] = EFF(0,2,0,0,0,0, 0,2,0,0,0,0);
        t[(int)Op::Int_Rot]  = EFF(0,3,0,0,0,0, 0,3,0,0,0,0);

        // ---- Bool ----
        t[(int)Op::Bool_And] = EFF(0,0,2,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Bool_Or]  = EFF(0,0,2,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Bool_Xor] = EFF(0,0,2,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Bool_Not] = EFF(0,0,1,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Bool_True] = EFF(0,0,0,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Bool_False]= EFF(0,0,0,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Bool_Pop] = EFF(0,0,1,0,0,0, 0,0,0,0,0,0);
        t[(int)Op::Bool_Dup] = EFF(0,0,1,0,0,0, 0,0,2,0,0,0);
        t[(int)Op::Bool_Swap]= EFF(0,0,2,0,0,0, 0,0,2,0,0,0);

        // ---- Comparisons (pop 2 of the typed stack, push 1 bool) ----
        t[(int)Op::Float_LT] = EFF(2,0,0,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Float_GT] = EFF(2,0,0,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Float_EQ] = EFF(2,0,0,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Int_LT]   = EFF(0,2,0,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Int_GT]   = EFF(0,2,0,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Int_EQ]   = EFF(0,2,0,0,0,0, 0,0,1,0,0,0);

        // ---- Conversions ----
        t[(int)Op::Float_FromInt] = EFF(0,1,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Int_FromFloat] = EFF(1,0,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Int_FromBool]  = EFF(0,0,1,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Bool_FromFloat]= EFF(1,0,0,0,0,0, 0,0,1,0,0,0);
        t[(int)Op::Bool_FromInt]  = EFF(0,1,0,0,0,0, 0,0,1,0,0,0);

        // ---- FVec ----
        // FVec.Len: peek FVec, push 1 int.  Needs FVec depth >= 1.
        t[(int)Op::FVec_Len]    = EFF(0,0,0,1,0,0, 0,1,0,1,0,0);   // pops & re-pushes (peek model)
        t[(int)Op::FVec_At]     = EFF(0,1,0,1,0,0, 1,0,0,1,0,0);   // pops int, peeks FVec, push float
        t[(int)Op::FVec_Set]    = EFF(1,1,0,1,0,0, 0,0,0,1,0,0);   // peek FVec; modifies in place
        t[(int)Op::FVec_Push]   = EFF(1,0,0,1,0,0, 0,0,0,1,0,0);   // peek FVec; push float consumed
        t[(int)Op::FVec_PopBack]= EFF(0,0,0,1,0,0, 1,0,0,1,0,0);   // peek FVec; pushes float
        t[(int)Op::FVec_New]    = EFF(0,1,0,0,0,0, 0,0,0,1,0,0);
        t[(int)Op::FVec_FromFloat]=EFF(1,0,0,0,0,0,0,0,0,1,0,0);
        t[(int)Op::FVec_Concat] = EFF(0,0,0,2,0,0, 0,0,0,1,0,0);
        t[(int)Op::FVec_Slice]  = EFF(0,2,0,1,0,0, 0,0,0,1,0,0);   // peek FVec
        t[(int)Op::FVec_Reverse]= EFF(0,0,0,1,0,0, 0,0,0,1,0,0);   // peek
        t[(int)Op::FVec_Roll]   = EFF(0,1,0,1,0,0, 0,0,0,1,0,0);   // peek
        t[(int)Op::FVec_Pop]    = EFF(0,0,0,1,0,0, 0,0,0,0,0,0);
        t[(int)Op::FVec_Dup]    = EFF(0,0,0,1,0,0, 0,0,0,2,0,0);
        t[(int)Op::FVec_Swap]   = EFF(0,0,0,2,0,0, 0,0,0,2,0,0);
        t[(int)Op::FVec_Rot]    = EFF(0,0,0,3,0,0, 0,0,0,3,0,0);

        // ---- BVec ----
        t[(int)Op::BVec_Len]    = EFF(0,0,0,0,1,0, 0,1,0,0,1,0);
        t[(int)Op::BVec_At]     = EFF(0,1,0,0,1,0, 0,0,1,0,1,0);
        t[(int)Op::BVec_Set]    = EFF(0,1,1,0,1,0, 0,0,0,0,1,0);
        t[(int)Op::BVec_Push]   = EFF(0,0,1,0,1,0, 0,0,0,0,1,0);
        t[(int)Op::BVec_PopBack]= EFF(0,0,0,0,1,0, 0,0,1,0,1,0);
        t[(int)Op::BVec_New]    = EFF(0,1,0,0,0,0, 0,0,0,0,1,0);
        t[(int)Op::BVec_Concat] = EFF(0,0,0,0,2,0, 0,0,0,0,1,0);
        t[(int)Op::BVec_Slice]  = EFF(0,2,0,0,1,0, 0,0,0,0,1,0);
        t[(int)Op::BVec_Reverse]= EFF(0,0,0,0,1,0, 0,0,0,0,1,0);
        t[(int)Op::BVec_Pop]    = EFF(0,0,0,0,1,0, 0,0,0,0,0,0);
        t[(int)Op::BVec_Dup]    = EFF(0,0,0,0,1,0, 0,0,0,0,2,0);
        t[(int)Op::BVec_Swap]   = EFF(0,0,0,0,2,0, 0,0,0,0,2,0);

        // ---- IVec ----
        t[(int)Op::IVec_Len]    = EFF(0,0,0,0,0,1, 0,1,0,0,0,1);
        t[(int)Op::IVec_At]     = EFF(0,1,0,0,0,1, 0,1,0,0,0,1);
        t[(int)Op::IVec_Set]    = EFF(0,2,0,0,0,1, 0,0,0,0,0,1);
        t[(int)Op::IVec_Push]   = EFF(0,1,0,0,0,1, 0,0,0,0,0,1);
        t[(int)Op::IVec_PopBack]= EFF(0,0,0,0,0,1, 0,1,0,0,0,1);
        t[(int)Op::IVec_New]    = EFF(0,1,0,0,0,0, 0,0,0,0,0,1);
        t[(int)Op::IVec_Concat] = EFF(0,0,0,0,0,2, 0,0,0,0,0,1);
        t[(int)Op::IVec_Slice]  = EFF(0,2,0,0,0,1, 0,0,0,0,0,1);
        t[(int)Op::IVec_Reverse]= EFF(0,0,0,0,0,1, 0,0,0,0,0,1);
        t[(int)Op::IVec_Pop]    = EFF(0,0,0,0,0,1, 0,0,0,0,0,0);
        t[(int)Op::IVec_Dup]    = EFF(0,0,0,0,0,1, 0,0,0,0,0,2);
        t[(int)Op::IVec_Swap]   = EFF(0,0,0,0,0,2, 0,0,0,0,0,2);

        // ---- Memory ----
        t[(int)Op::Mem_Read]    = EFF(0,1,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Mem_Write]   = EFF(1,1,0,0,0,0, 0,0,0,0,0,0);
        t[(int)Op::Mem_ReadVec] = EFF(0,2,0,0,0,0, 0,0,0,1,0,0);
        t[(int)Op::Mem_WriteVec]= EFF(0,1,0,1,0,0, 0,0,0,0,0,0);

        // ---- Env ----
        t[(int)Op::Env_GetChannelLLR] = EFF_LLR(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Env_GetIncomingVec]= EFF(0,0,0,0,0,0, 0,0,0,1,0,0);
        t[(int)Op::Env_GetNoiseVar]   = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);
        t[(int)Op::Env_GetIter]       = EFF(0,0,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Env_GetMaxIter]    = EFF(0,0,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Env_GetDeg]        = EFF(0,0,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Env_GetEdgeIndex]  = EFF(0,0,0,0,0,0, 0,1,0,0,0,0);
        t[(int)Op::Env_GetCodeRate]   = EFF(0,0,0,0,0,0, 1,0,0,0,0,0);

        // ---- Control flow ----
        // Body is generated recursively; its net effect on the parent
        // virtual stack is treated as neutral.  Listed pops are what the
        // op itself consumes *before* entering the body.
        t[(int)Op::Exec_If]      = EFF_CTRL(0,0,1,0,0,0, 0,0,0,0,0,0);
        t[(int)Op::Exec_When]    = EFF_CTRL(0,0,1,0,0,0, 0,0,0,0,0,0);
        t[(int)Op::Exec_DoTimes] = EFF_CTRL(0,1,0,0,0,0, 0,0,0,0,0,0);
        t[(int)Op::Exec_DoRange] = EFF_CTRL(0,2,0,0,0,0, 0,0,0,0,0,0);
        t[(int)Op::Exec_While]   = EFF_CTRL(0,0,0,0,0,0, 0,0,0,0,0,0);
        return t;
    }();
    return T;
}

#undef EFF
#undef EFF_LLR
#undef EFF_CTRL

} // namespace detail

inline const StackEffect& effect_of(Op op) {
    return detail::build_table()[(int)op];
}

// Virtual stack state used by the stack-aware sampler.
struct VStack {
    int f = 0, i = 0, b = 0, fv = 0, bv = 0, iv = 0;
    bool has_channel_llr = false;

    // V2C entry-point state: matches validators._seed_v2c_stacks.
    static VStack v2c_entry() {
        VStack s; s.f = 1; s.i = 4; s.fv = 1; s.has_channel_llr = true; return s;
    }
    // C2V entry-point state: matches validators._seed_c2v_stacks.
    static VStack c2v_entry() {
        VStack s; s.i = 4; s.fv = 1; s.has_channel_llr = false; return s;
    }

    // Returns true if the op's *immediate* preconditions are satisfied.
    bool can_apply(Op op) const {
        const auto& e = effect_of(op);
        if (e.needs_channel_llr && !has_channel_llr) return false;
        return f  >= e.f_pop  && i  >= e.i_pop  && b  >= e.b_pop
            && fv >= e.fv_pop && bv >= e.bv_pop && iv >= e.iv_pop;
    }

    // Apply the op's net effect (assumes can_apply was true).
    void apply(Op op) {
        const auto& e = effect_of(op);
        f  += e.f_push  - e.f_pop;
        i  += e.i_push  - e.i_pop;
        b  += e.b_push  - e.b_pop;
        fv += e.fv_push - e.fv_pop;
        bv += e.bv_push - e.bv_pop;
        iv += e.iv_push - e.iv_pop;
    }

    // Body-entry adjustments for control-flow ops (DoTimes pushes the
    // loop counter to the int stack at the start of each iteration; the
    // rest are no-op since the bool/ints were already popped at parent
    // scope when the op itself was applied).
    static VStack child_for(const VStack& parent, Op op) {
        VStack c = parent;
        if (op == Op::Exec_DoTimes) { c.i += 1; }       // counter i
        else if (op == Op::Exec_DoRange) { c.i += 1; }  // counter i
        // Exec_If/When/While: body sees parent state unchanged.
        return c;
    }
};

} // namespace pushgp_cpp
