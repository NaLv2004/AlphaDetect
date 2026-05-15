// opcodes.hpp — enumerate all 127 opcodes; map name <-> id
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace pushgp_cpp {

// Order matches the categorisation in pushgp/instructions.py.
// Keep stable; tests rely on a known names<->ids mapping.
enum class Op : int {
    // ---- Float arithmetic (18) ----
    Float_Add, Float_Sub, Float_Mul, Float_Div, Float_Mod,
    Float_Min, Float_Max,
    Float_Abs, Float_Neg, Float_Inv, Float_Sqrt, Float_Square,
    Float_Exp, Float_Log, Float_Tanh, Float_Atanh, Float_Sign,
    Float_Floor, Float_Ceil,

    // ---- Float constants (8 + 8 evolved = 16) ----
    Float_Const0, Float_Const1, Float_ConstNeg1, Float_ConstHalf,
    Float_Const2, Float_Const0_1, Float_ConstPi, Float_Const1eNeg6,
    Float_EvoConst0, Float_EvoConst1, Float_EvoConst2, Float_EvoConst3,
    Float_EvoConst4, Float_EvoConst5, Float_EvoConst6, Float_EvoConst7,

    // ---- Float stack ops (6) ----
    Float_Pop, Float_Dup, Float_Swap, Float_Rot, Float_Yank, Float_Shove,

    // ---- Int arithmetic (10) ----
    Int_Add, Int_Sub, Int_Mul, Int_Div, Int_Mod,
    Int_Min, Int_Max, Int_Inc, Int_Dec, Int_Neg,
    // ---- Int constants (4) ----
    Int_Const0, Int_Const1, Int_Const2, Int_ConstNeg1,
    // ---- Int stack ops (4) ----
    Int_Pop, Int_Dup, Int_Swap, Int_Rot,

    // ---- Bool ops (5+2 = 7) ----
    Bool_And, Bool_Or, Bool_Xor, Bool_Not, Bool_True, Bool_False,
    Bool_Pop, Bool_Dup, Bool_Swap,

    // ---- Comparisons (6) ----
    Float_LT, Float_GT, Float_EQ,
    Int_LT, Int_GT, Int_EQ,

    // ---- Conversions (5) ----
    Float_FromInt, Int_FromFloat, Int_FromBool,
    Bool_FromFloat, Bool_FromInt,

    // ---- FloatVec (15: Len At Set Push PopBack New FromFloat Concat Slice Reverse Roll Pop Dup Swap Rot) ----
    FVec_Len, FVec_At, FVec_Set, FVec_Push, FVec_PopBack, FVec_New,
    FVec_FromFloat, FVec_Concat, FVec_Slice, FVec_Reverse, FVec_Roll,
    FVec_Pop, FVec_Dup, FVec_Swap, FVec_Rot,

    // ---- BoolVec (12: Len At Set Push PopBack New Concat Slice Reverse Pop Dup Swap) ----
    BVec_Len, BVec_At, BVec_Set, BVec_Push, BVec_PopBack, BVec_New,
    BVec_Concat, BVec_Slice, BVec_Reverse, BVec_Pop, BVec_Dup, BVec_Swap,

    // ---- IntVec (12) ----
    IVec_Len, IVec_At, IVec_Set, IVec_Push, IVec_PopBack, IVec_New,
    IVec_Concat, IVec_Slice, IVec_Reverse, IVec_Pop, IVec_Dup, IVec_Swap,

    // ---- Memory (4) ----
    Mem_Read, Mem_Write, Mem_ReadVec, Mem_WriteVec,

    // ---- Env (8) ----
    Env_GetChannelLLR, Env_GetIncomingVec, Env_GetNoiseVar,
    Env_GetIter, Env_GetMaxIter, Env_GetDeg,
    Env_GetEdgeIndex, Env_GetCodeRate,

    // ---- Control flow (5) ----
    Exec_If, Exec_When, Exec_DoTimes, Exec_DoRange, Exec_While,

    _COUNT
};

constexpr int OP_COUNT = static_cast<int>(Op::_COUNT);

inline bool op_is_control(Op op) {
    return op == Op::Exec_If || op == Op::Exec_When
        || op == Op::Exec_DoTimes || op == Op::Exec_DoRange
        || op == Op::Exec_While;
}
inline bool op_has_two_blocks(Op op) { return op == Op::Exec_If; }

// Implemented in opcodes.cpp (compiled into a single TU via bindings.cpp).
const std::string& op_to_name(Op op);
bool name_to_op(const std::string& name, Op& out);
const std::vector<std::string>& all_op_names();

}  // namespace pushgp_cpp
