// opcodes_table.hpp — name<->id table generated to keep one source of truth.
// Included once into a TU (bindings.cpp) where OPCODE_TABLE_DEFINE is defined.
#pragma once

#include "opcodes.hpp"

#include <stdexcept>

namespace pushgp_cpp {

namespace detail {

// Order MUST match the Op enum declaration order.
inline const std::vector<std::string>& _build_names() {
    static const std::vector<std::string> NAMES = {
        // Float arith (18)
        "Float.Add","Float.Sub","Float.Mul","Float.Div","Float.Mod",
        "Float.Min","Float.Max",
        "Float.Abs","Float.Neg","Float.Inv","Float.Sqrt","Float.Square",
        "Float.Exp","Float.Log","Float.Tanh","Float.Atanh","Float.Sign",
        "Float.Floor","Float.Ceil",
        // Float consts (16)
        "Float.Const0","Float.Const1","Float.ConstNeg1","Float.ConstHalf",
        "Float.Const2","Float.Const0_1","Float.ConstPi","Float.Const1e-6",
        "Float.EvoConst0","Float.EvoConst1","Float.EvoConst2","Float.EvoConst3",
        "Float.EvoConst4","Float.EvoConst5","Float.EvoConst6","Float.EvoConst7",
        // Float stack (6)
        "Float.Pop","Float.Dup","Float.Swap","Float.Rot","Float.Yank","Float.Shove",
        // Int arith (10)
        "Int.Add","Int.Sub","Int.Mul","Int.Div","Int.Mod",
        "Int.Min","Int.Max","Int.Inc","Int.Dec","Int.Neg",
        // Int consts (4)
        "Int.Const0","Int.Const1","Int.Const2","Int.ConstNeg1",
        // Int stack (4)
        "Int.Pop","Int.Dup","Int.Swap","Int.Rot",
        // Bool (9)
        "Bool.And","Bool.Or","Bool.Xor","Bool.Not","Bool.True","Bool.False",
        "Bool.Pop","Bool.Dup","Bool.Swap",
        // Compare (6)
        "Float.LT","Float.GT","Float.EQ",
        "Int.LT","Int.GT","Int.EQ",
        // Conv (5)
        "Float.FromInt","Int.FromFloat","Int.FromBool",
        "Bool.FromFloat","Bool.FromInt",
        // FVec (15)
        "FVec.Len","FVec.At","FVec.Set","FVec.Push","FVec.PopBack","FVec.New",
        "FVec.FromFloat","FVec.Concat","FVec.Slice","FVec.Reverse","FVec.Roll",
        "FVec.Pop","FVec.Dup","FVec.Swap","FVec.Rot",
        // BVec (12)
        "BVec.Len","BVec.At","BVec.Set","BVec.Push","BVec.PopBack","BVec.New",
        "BVec.Concat","BVec.Slice","BVec.Reverse","BVec.Pop","BVec.Dup","BVec.Swap",
        // IVec (12)
        "IVec.Len","IVec.At","IVec.Set","IVec.Push","IVec.PopBack","IVec.New",
        "IVec.Concat","IVec.Slice","IVec.Reverse","IVec.Pop","IVec.Dup","IVec.Swap",
        // Mem (4)
        "Mem.Read","Mem.Write","Mem.ReadVec","Mem.WriteVec",
        // Env (8)
        "Env.GetChannelLLR","Env.GetIncomingVec","Env.GetNoiseVar",
        "Env.GetIter","Env.GetMaxIter","Env.GetDeg",
        "Env.GetEdgeIndex","Env.GetCodeRate",
        // Control (5)
        "Exec.If","Exec.When","Exec.DoTimes","Exec.DoRange","Exec.While",
    };
    if (static_cast<int>(NAMES.size()) != OP_COUNT) {
        throw std::logic_error("opcodes_table: NAMES.size() != OP_COUNT");
    }
    return NAMES;
}

inline const std::unordered_map<std::string, Op>& _build_index() {
    static const std::unordered_map<std::string, Op> IDX = []() {
        std::unordered_map<std::string, Op> m;
        const auto& names = _build_names();
        m.reserve(names.size());
        for (size_t i = 0; i < names.size(); ++i) {
            m.emplace(names[i], static_cast<Op>(i));
        }
        return m;
    }();
    return IDX;
}

} // namespace detail

inline const std::vector<std::string>& all_op_names() {
    return detail::_build_names();
}
inline const std::string& op_to_name(Op op) {
    return detail::_build_names()[static_cast<int>(op)];
}
inline bool name_to_op(const std::string& name, Op& out) {
    const auto& idx = detail::_build_index();
    auto it = idx.find(name);
    if (it == idx.end()) return false;
    out = it->second;
    return true;
}

}  // namespace pushgp_cpp
