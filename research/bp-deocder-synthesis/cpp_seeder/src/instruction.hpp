// instruction.hpp — Instruction tree (mirrors pushgp/program.py:Instruction)
#pragma once

#include "opcodes.hpp"

#include <memory>
#include <vector>

namespace pushgp_cpp {

struct Instruction {
    Op op;
    // code_block / code_block2 are only present for control instructions.
    // We use unique_ptr<vector<Instruction>> instead of optional<vector> so
    // the recursive type can be forward-declared cleanly.
    std::unique_ptr<std::vector<Instruction>> code_block;
    std::unique_ptr<std::vector<Instruction>> code_block2;

    Instruction() : op(Op::Float_Const0) {}
    explicit Instruction(Op o) : op(o) {}

    Instruction(const Instruction& other) : op(other.op) {
        if (other.code_block) {
            code_block = std::make_unique<std::vector<Instruction>>(*other.code_block);
        }
        if (other.code_block2) {
            code_block2 = std::make_unique<std::vector<Instruction>>(*other.code_block2);
        }
    }
    Instruction& operator=(const Instruction& other) {
        if (this != &other) {
            op = other.op;
            code_block = other.code_block
                ? std::make_unique<std::vector<Instruction>>(*other.code_block) : nullptr;
            code_block2 = other.code_block2
                ? std::make_unique<std::vector<Instruction>>(*other.code_block2) : nullptr;
        }
        return *this;
    }
    Instruction(Instruction&&) noexcept = default;
    Instruction& operator=(Instruction&&) noexcept = default;
};

using Program = std::vector<Instruction>;

inline int program_length(const Program& p) {
    int n = 0;
    for (const auto& ins : p) {
        n += 1;
        if (ins.code_block) n += program_length(*ins.code_block);
        if (ins.code_block2) n += program_length(*ins.code_block2);
    }
    return n;
}

}  // namespace pushgp_cpp
