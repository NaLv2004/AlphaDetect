// vm_state.hpp — typed stacks + execution context (mirrors pushgp/stacks.py)
#pragma once

#include "common.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace pushgp_cpp {

// Forgiving LIFO stack: push beyond max_depth drops the bottom element.
template <typename T, int MaxDepth>
class TypedStack {
   public:
    void clear() { data_.clear(); }
    int  depth() const { return static_cast<int>(data_.size()); }
    bool empty() const { return data_.empty(); }

    void push(const T& v) {
        if (static_cast<int>(data_.size()) >= MaxDepth) data_.pop_front();
        data_.push_back(v);
    }
    void push(T&& v) {
        if (static_cast<int>(data_.size()) >= MaxDepth) data_.pop_front();
        data_.push_back(std::move(v));
    }

    bool pop(T& out) {
        if (data_.empty()) return false;
        out = std::move(data_.back());
        data_.pop_back();
        return true;
    }
    // Discard top, no-op if empty.
    void drop_top() {
        if (!data_.empty()) data_.pop_back();
    }

    // peek(0) = top
    const T* peek(int offset = 0) const {
        if (offset < 0 || offset >= static_cast<int>(data_.size())) return nullptr;
        return &data_[data_.size() - 1 - offset];
    }
    T* peek_mut(int offset = 0) {
        if (offset < 0 || offset >= static_cast<int>(data_.size())) return nullptr;
        return &data_[data_.size() - 1 - offset];
    }

    // Forth ops -----
    void dup() {
        if (data_.empty()) return;
        // copy of back; for vectors this is a deep copy because T owns its storage.
        T copy = data_.back();
        push(std::move(copy));
    }
    void swap() {
        if (data_.size() < 2) return;
        std::swap(data_[data_.size() - 1], data_[data_.size() - 2]);
    }
    void rot() {
        // (a b c -- b c a)
        if (data_.size() < 3) return;
        T a = std::move(data_[data_.size() - 3]);
        data_.erase(data_.begin() + (data_.size() - 3));
        data_.push_back(std::move(a));
    }
    void yank(int n) {
        if (n <= 0 || n >= static_cast<int>(data_.size())) return;
        T v = std::move(data_[data_.size() - 1 - n]);
        data_.erase(data_.begin() + (data_.size() - 1 - n));
        data_.push_back(std::move(v));
    }
    void shove(int n) {
        if (n <= 0 || n >= static_cast<int>(data_.size())) return;
        T v = std::move(data_.back());
        data_.pop_back();
        // Match Python: insert(len(self._data) - n + 1, v)
        // After pop, index = data_.size() - n + 1.
        size_t idx = data_.size() - static_cast<size_t>(n) + 1;
        data_.insert(data_.begin() + idx, std::move(v));
    }

   private:
    std::deque<T> data_;
};

// Specialised vector type — owns float64 storage (matches np.float64 array).
using FVec = std::vector<double>;
using BVec = std::vector<uint8_t>;   // 0/1 stored as bytes (matches numpy bool_)
using IVec = std::vector<int64_t>;

struct VMState {
    TypedStack<double,  MAX_DEPTH_FLOAT> floats;
    TypedStack<int64_t, MAX_DEPTH_INT>   ints;
    TypedStack<uint8_t, MAX_DEPTH_BOOL>  bools;  // 0/1
    TypedStack<FVec,    MAX_DEPTH_FVEC>  fvecs;
    TypedStack<BVec,    MAX_DEPTH_BVEC>  bvecs;
    TypedStack<IVec,    MAX_DEPTH_IVEC>  ivecs;

    // Read-only context
    double  ctx_channel_llr   = 0.0;
    FVec    ctx_incoming;
    double  ctx_noise_var     = 1.0;
    int64_t ctx_iter          = 0;
    int64_t ctx_max_iter      = 25;
    int64_t ctx_deg           = 0;
    int64_t ctx_edge_index    = 0;
    double  ctx_code_rate     = 0.5;
    std::array<double, N_EVO_CONSTS> ctx_evo_constants{{1,1,1,1,1,1,1,1}};
    bool    ctx_has_channel_llr = true;

    // Working memory (16 cells)
    std::array<double, MEMORY_SIZE> memory{};

    // Bookkeeping
    int  step_count = 0;
    int  flop_count = 0;
    bool fault      = false;
    std::string fault_reason;

    void reset_stacks() {
        floats.clear();
        ints.clear();
        bools.clear();
        fvecs.clear();
        bvecs.clear();
        ivecs.clear();
        memory.fill(0.0);
        step_count = 0;
        flop_count = 0;
        fault = false;
        fault_reason.clear();
    }
};

}  // namespace pushgp_cpp
