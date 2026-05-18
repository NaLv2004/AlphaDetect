// symbolic_expr.hpp — hash-consed expression DAG for the shadow VM.
//
// One immutable Expr node per distinct (kind, op, payload, children) tuple.
// All factory functions normalize first (constant folding + commutative
// sorting + idempotent reductions + sign extraction), then look up / insert
// into the global hash-cons table.  Two Expr pointers are equal iff the
// expressions are structurally identical after normalization.
//
// This is the "字符串签名" of the design doc, except we use a tagged tree
// (cheaper than serializing to / parsing from strings).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "opcodes.hpp"

namespace pushgp_cpp {

// ----------------------------- Atom kinds -------------------------------

// Only three true "input atoms" — Env scalars are folded to literals by
// the symbolic VM (user spec: validator provides concrete env values).
enum class AtomKind : uint8_t {
    X = 0,    // incoming[i]
    LV = 1,   // channel LLR (v2c only)
    EVO = 2,  // evo_constants[i]
};

// ----------------------------- Expr kind --------------------------------

enum class ExprKind : uint8_t {
    LitFloat,
    LitInt,
    LitBool,
    Atom,
    UnOp,     // op_tag is the original Op enum; one child
    BinOp,    // op_tag is the original Op enum; two children, ordered
    NAry,     // op_tag is the original Op enum (Add/Mul/Min/Max/And/Or/Xor);
              // sorted by child hash; flattened
    IfElse,   // [cond, then, else]
    Opaque,   // atom set only; structure lost
};

struct Expr;
using ExprRef = std::shared_ptr<const Expr>;

struct Expr {
    ExprKind kind;
    Op op_tag = Op::_COUNT;            // for UnOp / BinOp / NAry
    int idx = -1;                      // AtomKind index (for Atom)
    AtomKind atom_kind = AtomKind::X;  // for Atom
    double lit_d = 0.0;                // for LitFloat
    int64_t lit_i = 0;                 // for LitInt / LitBool (0/1)
    std::vector<ExprRef> ch;
    uint64_t hash = 0;
    // Sorted unique list of (atom_kind, idx) encoded as int (kind*256 + idx).
    // We never use more than 256 indices per kind so this fits.
    std::vector<int> atoms;
};

// Encode (kind, idx) as a single int for the atoms set.
inline int encode_atom(AtomKind k, int idx) {
    return (static_cast<int>(k) << 16) | (idx & 0xffff);
}
inline AtomKind atom_kind_of(int enc) { return static_cast<AtomKind>(enc >> 16); }
inline int atom_idx_of(int enc) { return enc & 0xffff; }

// ----------------------------- Hash mixing ------------------------------

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
inline uint64_t mix(uint64_t a, uint64_t b) {
    return splitmix64(a * 0x9e3779b97f4a7c15ULL + b);
}

// ----------------------------- Hash-cons table --------------------------

struct ExprKeyHash {
    size_t operator()(uint64_t h) const noexcept { return static_cast<size_t>(h); }
};

class ExprTable {
   public:
    // Look up by hash + structural equality.  Multiple Exprs may share
    // a hash; we walk the bucket and compare.
    ExprRef intern(ExprRef e) {
        std::lock_guard<std::mutex> lk(mu_);
        auto range = table_.equal_range(e->hash);
        for (auto it = range.first; it != range.second; ++it) {
            if (struct_equal_(it->second, e)) return it->second;
        }
        table_.emplace(e->hash, e);
        return e;
    }
    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return table_.size();
    }
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        table_.clear();
    }

   private:
    static bool struct_equal_(const ExprRef& a, const ExprRef& b) {
        if (a.get() == b.get()) return true;
        if (a->kind != b->kind) return false;
        if (a->op_tag != b->op_tag) return false;
        if (a->idx != b->idx || a->atom_kind != b->atom_kind) return false;
        if (a->kind == ExprKind::LitFloat) {
            // bitwise equality so NaN==NaN consistently
            uint64_t x, y;
            std::memcpy(&x, &a->lit_d, sizeof(x));
            std::memcpy(&y, &b->lit_d, sizeof(y));
            return x == y;
        }
        if (a->kind == ExprKind::LitInt || a->kind == ExprKind::LitBool) {
            return a->lit_i == b->lit_i;
        }
        if (a->ch.size() != b->ch.size()) return false;
        for (size_t i = 0; i < a->ch.size(); ++i) {
            if (a->ch[i].get() != b->ch[i].get()) return false;
        }
        // For Opaque: also compare atoms (they're not in children).
        if (a->kind == ExprKind::Opaque) {
            if (a->atoms != b->atoms) return false;
        }
        return true;
    }
    mutable std::mutex mu_;
    std::unordered_multimap<uint64_t, ExprRef, ExprKeyHash> table_;
};

inline ExprTable& expr_table() {
    // Thread-local: each worker thread has its own intern table so we
    // avoid cross-thread contention and unbounded growth.  Hash-cons
    // identity is per-thread, which is fine because all Expr objects
    // produced for a single program's validation are created and
    // consumed in the same thread.
    thread_local ExprTable T;
    return T;
}

// ----------------------------- atoms set helpers ------------------------

inline std::vector<int> merge_atoms(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out;
    out.reserve(a.size() + b.size());
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) out.push_back(a[i++]);
        else if (a[i] > b[j]) out.push_back(b[j++]);
        else { out.push_back(a[i]); ++i; ++j; }
    }
    while (i < a.size()) out.push_back(a[i++]);
    while (j < b.size()) out.push_back(b[j++]);
    return out;
}

inline std::vector<int> singleton_atom(AtomKind k, int idx) {
    return {encode_atom(k, idx)};
}

// ----------------------------- factory helpers --------------------------

inline ExprRef make_lit_float(double v) {
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::LitFloat;
    e->lit_d = v;
    uint64_t bits; std::memcpy(&bits, &v, sizeof(bits));
    e->hash = mix(0xF10A7ULL, bits);
    return expr_table().intern(e);
}
inline ExprRef make_lit_int(int64_t v) {
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::LitInt;
    e->lit_i = v;
    e->hash = mix(0x141U, static_cast<uint64_t>(v));
    return expr_table().intern(e);
}
inline ExprRef make_lit_bool(bool v) {
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::LitBool;
    e->lit_i = v ? 1 : 0;
    e->hash = mix(0xB001U, v ? 1 : 0);
    return expr_table().intern(e);
}
inline ExprRef make_atom(AtomKind k, int idx) {
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::Atom;
    e->atom_kind = k;
    e->idx = idx;
    e->hash = mix(0xA70A0ULL, (static_cast<uint64_t>(k) << 32) | static_cast<uint32_t>(idx));
    e->atoms = singleton_atom(k, idx);
    return expr_table().intern(e);
}
inline ExprRef make_opaque(std::vector<int> atoms) {
    std::sort(atoms.begin(), atoms.end());
    atoms.erase(std::unique(atoms.begin(), atoms.end()), atoms.end());
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::Opaque;
    uint64_t h = 0x0FACECAFEULL;
    for (int a : atoms) h = mix(h, static_cast<uint32_t>(a));
    e->hash = h;
    e->atoms = std::move(atoms);
    return expr_table().intern(e);
}

// ---- inspection helpers ----

inline bool is_lit_float(const ExprRef& e) { return e->kind == ExprKind::LitFloat; }
inline bool is_lit_int(const ExprRef& e) { return e->kind == ExprKind::LitInt; }
inline bool is_lit_bool(const ExprRef& e) { return e->kind == ExprKind::LitBool; }
inline bool is_opaque(const ExprRef& e) { return e->kind == ExprKind::Opaque; }
inline bool is_neg(const ExprRef& e) {
    return e->kind == ExprKind::UnOp && e->op_tag == Op::Float_Neg;
}

inline double lit_float_v(const ExprRef& e) { return e->lit_d; }
inline int64_t lit_int_v(const ExprRef& e) { return e->lit_i; }
inline bool lit_bool_v(const ExprRef& e) { return e->lit_i != 0; }

// ----------------------------- common factory: UnOp ---------------------

ExprRef make_unop(Op op, ExprRef a);
ExprRef make_binop(Op op, ExprRef a, ExprRef b);
ExprRef make_nary(Op op, std::vector<ExprRef> children);
ExprRef make_ifelse(ExprRef cond, ExprRef then_, ExprRef else_);

// op classification ------------------------------------------------------

inline bool is_commutative_op(Op op) {
    switch (op) {
        case Op::Float_Add: case Op::Float_Mul:
        case Op::Float_Min: case Op::Float_Max:
        case Op::Int_Add:   case Op::Int_Mul:
        case Op::Int_Min:   case Op::Int_Max:
        case Op::Bool_And:  case Op::Bool_Or: case Op::Bool_Xor:
        case Op::Float_EQ:  case Op::Int_EQ:
            return true;
        default: return false;
    }
}

// ----------------------------- substitute (for σ-checks) ----------------

// Map (kind, idx) → new (kind, idx).  Used for permutation tests and odd-check.
struct AtomMap {
    // mapping[encode_atom(k,i)] = new encoded atom OR -1 to leave alone.
    std::unordered_map<int, int> table;
    // If non-empty, atoms in `negate` are replaced by Float_Neg(atom).
    // Used for odd-check (substitute x→-x then compare to -orig).
    std::unordered_map<int, bool> negate;
};

// Recursively rewrite `e` under map.  Hash-consing means common sub-expressions
// produce identical results in O(1) via cache.
ExprRef substitute(const ExprRef& e, const AtomMap& map);

// ----------------------------- size cap ---------------------------------

inline int SYMBOLIC_NODE_CAP() {
    // total intern table size cap; once exceeded the VM marks the program OPAQUE.
    return 200000;
}

// ===================== factory implementations =========================

inline bool is_zero_float(const ExprRef& e) {
    return e->kind == ExprKind::LitFloat && e->lit_d == 0.0;
}
inline bool is_one_float(const ExprRef& e) {
    return e->kind == ExprKind::LitFloat && e->lit_d == 1.0;
}
inline bool is_zero_int(const ExprRef& e) {
    return e->kind == ExprKind::LitInt && e->lit_i == 0;
}
inline bool is_one_int(const ExprRef& e) {
    return e->kind == ExprKind::LitInt && e->lit_i == 1;
}

// Compute clamped float (matches VM FLOAT_CLAMP semantics for op outputs).
inline double sym_clamp(double x) {
    if (std::isnan(x)) return std::nan("");
    if (x >  FLOAT_CLAMP_DEFAULT) return  FLOAT_CLAMP_DEFAULT;
    if (x < -FLOAT_CLAMP_DEFAULT) return -FLOAT_CLAMP_DEFAULT;
    return x;
}

inline ExprRef make_unop(Op op, ExprRef a) {
    // ---- constant folding ----
    if (is_lit_float(a)) {
        double v = lit_float_v(a);
        bool ok = true;
        double r = 0.0;
        switch (op) {
            case Op::Float_Abs:    r = std::fabs(v); break;
            case Op::Float_Neg:    r = -v; break;
            case Op::Float_Inv:    if (v == 0.0) { ok = false; } else r = 1.0 / v; break;
            case Op::Float_Sqrt:   if (v < 0.0) { ok = false; } else r = std::sqrt(v); break;
            case Op::Float_Square: r = v * v; break;
            case Op::Float_Exp:    r = (v <= 700.0 ? std::exp(v) : std::numeric_limits<double>::infinity()); break;
            case Op::Float_Log:    if (v <= 0.0) { ok = false; } else r = std::log(v); break;
            case Op::Float_Tanh:   r = std::tanh(v); break;
            case Op::Float_Atanh:  if (v <= -1.0 || v >= 1.0) { ok = false; } else r = std::atanh(v); break;
            case Op::Float_Sign:   r = (v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0)); break;
            case Op::Float_Floor:  r = std::floor(v); break;
            case Op::Float_Ceil:   r = std::ceil(v); break;
            default: ok = false; break;
        }
        if (ok) return make_lit_float(sym_clamp(r));
    }
    if (is_lit_int(a)) {
        int64_t v = lit_int_v(a);
        switch (op) {
            case Op::Int_Inc: return make_lit_int(v + 1);
            case Op::Int_Dec: return make_lit_int(v - 1);
            case Op::Int_Neg: return make_lit_int(-v);
            case Op::Float_FromInt: return make_lit_float(static_cast<double>(v));
            case Op::Bool_FromInt: return make_lit_bool(v != 0);
            default: break;
        }
    }
    if (is_lit_bool(a)) {
        if (op == Op::Bool_Not) return make_lit_bool(!lit_bool_v(a));
        if (op == Op::Int_FromBool) return make_lit_int(lit_bool_v(a) ? 1 : 0);
    }
    // ---- idempotent reductions ----
    if (op == Op::Float_Neg && a->kind == ExprKind::UnOp && a->op_tag == Op::Float_Neg) {
        return a->ch[0];  // neg(neg(x)) = x
    }
    if (op == Op::Float_Abs && a->kind == ExprKind::UnOp && a->op_tag == Op::Float_Abs) {
        return a;  // abs(abs(x)) = abs(x)
    }
    if (op == Op::Float_Abs && a->kind == ExprKind::UnOp && a->op_tag == Op::Float_Neg) {
        // abs(neg(x)) = abs(x)
        return make_unop(Op::Float_Abs, a->ch[0]);
    }
    if (op == Op::Float_Sign && a->kind == ExprKind::UnOp && a->op_tag == Op::Float_Sign) {
        return a;
    }
    if (op == Op::Float_Square && a->kind == ExprKind::UnOp && a->op_tag == Op::Float_Neg) {
        // square(neg(x)) = square(x)
        return make_unop(Op::Float_Square, a->ch[0]);
    }
    if (op == Op::Bool_Not && a->kind == ExprKind::UnOp && a->op_tag == Op::Bool_Not) {
        return a->ch[0];
    }
    // ---- "almost-everywhere = 1" simplifications ----
    // Under the standing policy that continuous inputs are non-zero a.e.,
    // the following collapse to the literal 1 (the measure-zero exceptions
    // at x=0 are absorbed by the multi-world "high-probability branch"):
    //   Sign(Square(x)) = 1   (Square ≥ 0, sign of nonneg nonzero = 1)
    //   Sign(Abs(x))    = 1   (Abs ≥ 0)
    //   Square(Sign(x)) = 1   (Sign ∈ {-1, 0, +1} ⇒ Square ∈ {0,1}, =1 a.e.)
    //   Abs(Sign(x))    = 1   (|±1| = 1 a.e.)
    if (op == Op::Float_Sign && a->kind == ExprKind::UnOp &&
        (a->op_tag == Op::Float_Square || a->op_tag == Op::Float_Abs)) {
        return make_lit_float(1.0);
    }
    if ((op == Op::Float_Square || op == Op::Float_Abs) &&
        a->kind == ExprKind::UnOp && a->op_tag == Op::Float_Sign) {
        return make_lit_float(1.0);
    }
    // ---- sign extraction (for odd-symmetry tests) ----
    // sign(-x) = -sign(x);  tanh(-x) = -tanh(x);  atanh(-x) = -atanh(x)
    if ((op == Op::Float_Sign || op == Op::Float_Tanh || op == Op::Float_Atanh) &&
        a->kind == ExprKind::UnOp && a->op_tag == Op::Float_Neg) {
        return make_unop(Op::Float_Neg, make_unop(op, a->ch[0]));
    }
    // ---- distribute Neg over Add / Sub / Mul (needed for oddness check) ----
    // Neg(Σ x_i) = Σ Neg(x_i)
    if (op == Op::Float_Neg && a->kind == ExprKind::NAry && a->op_tag == Op::Float_Add) {
        std::vector<ExprRef> negs;
        negs.reserve(a->ch.size());
        for (const auto& c : a->ch) negs.push_back(make_unop(Op::Float_Neg, c));
        return make_nary(Op::Float_Add, negs);
    }
    if (op == Op::Int_Neg && a->kind == ExprKind::NAry && a->op_tag == Op::Int_Add) {
        std::vector<ExprRef> negs;
        negs.reserve(a->ch.size());
        for (const auto& c : a->ch) negs.push_back(make_unop(Op::Int_Neg, c));
        return make_nary(Op::Int_Add, negs);
    }
    // Neg(a - b) = b - a
    if (op == Op::Float_Neg && a->kind == ExprKind::BinOp && a->op_tag == Op::Float_Sub) {
        return make_binop(Op::Float_Sub, a->ch[1], a->ch[0]);
    }
    if (op == Op::Int_Neg && a->kind == ExprKind::BinOp && a->op_tag == Op::Int_Sub) {
        return make_binop(Op::Int_Sub, a->ch[1], a->ch[0]);
    }
    // Neg(Π) = put Neg on first factor (canonical), then resort
    if (op == Op::Float_Neg && a->kind == ExprKind::NAry && a->op_tag == Op::Float_Mul) {
        std::vector<ExprRef> ch(a->ch.begin(), a->ch.end());
        ch[0] = make_unop(Op::Float_Neg, ch[0]);
        return make_nary(Op::Float_Mul, ch);
    }
    if (op == Op::Int_Neg && a->kind == ExprKind::NAry && a->op_tag == Op::Int_Mul) {
        std::vector<ExprRef> ch(a->ch.begin(), a->ch.end());
        ch[0] = make_unop(Op::Int_Neg, ch[0]);
        return make_nary(Op::Int_Mul, ch);
    }

    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::UnOp;
    e->op_tag = op;
    e->ch = {a};
    e->hash = mix(mix(0xA001U, static_cast<uint64_t>(op)), a->hash);
    e->atoms = a->atoms;
    return expr_table().intern(e);
}

inline ExprRef make_binop(Op op, ExprRef a, ExprRef b) {
    // constant folding
    if (is_lit_float(a) && is_lit_float(b)) {
        double x = lit_float_v(a), y = lit_float_v(b);
        bool ok = true; double r = 0.0;
        switch (op) {
            case Op::Float_Sub: r = x - y; break;
            case Op::Float_Div: if (y == 0.0) ok = false; else r = x / y; break;
            case Op::Float_Mod: if (y == 0.0) ok = false; else r = std::fmod(x, y); break;
            default: ok = false; break;
        }
        if (ok) return make_lit_float(sym_clamp(r));
    }
    if (is_lit_int(a) && is_lit_int(b)) {
        int64_t x = lit_int_v(a), y = lit_int_v(b);
        switch (op) {
            case Op::Int_Sub: return make_lit_int(x - y);
            case Op::Int_Div: {
                if (y == 0) return make_lit_int(0);
                int64_t q = x / y;
                int64_t r = x % y;
                if (r != 0 && ((r < 0) != (y < 0))) --q;
                return make_lit_int(q);
            }
            case Op::Int_Mod: {
                if (y == 0) return make_lit_int(0);
                int64_t r = x % y;
                if (r != 0 && ((r < 0) != (y < 0))) r += y;
                return make_lit_int(r);
            }
            default: break;
        }
    }
    // a - a = 0
    if (op == Op::Float_Sub && a.get() == b.get()) return make_lit_float(0.0);
    if (op == Op::Int_Sub && a.get() == b.get()) return make_lit_int(0);
    // a - 0 = a
    if (op == Op::Float_Sub && is_zero_float(b)) return a;
    if (op == Op::Int_Sub && is_zero_int(b)) return a;
    // 0 - a = -a
    if (op == Op::Float_Sub && is_zero_float(a)) return make_unop(Op::Float_Neg, b);
    if (op == Op::Int_Sub && is_zero_int(a)) return make_unop(Op::Int_Neg, b);
    // a / 1 = a;  a / a = 1 (if a non-zero — conservatively only for literals)
    if (op == Op::Float_Div && is_one_float(b)) return a;
    if (op == Op::Int_Div && is_one_int(b)) return a;
    // a/b where both are literals already handled above.

    // Comparison constant folding
    if (is_lit_float(a) && is_lit_float(b)) {
        double x = lit_float_v(a), y = lit_float_v(b);
        switch (op) {
            case Op::Float_LT: return make_lit_bool(x < y);
            case Op::Float_GT: return make_lit_bool(x > y);
            default: break;
        }
    }
    if (is_lit_int(a) && is_lit_int(b)) {
        int64_t x = lit_int_v(a), y = lit_int_v(b);
        switch (op) {
            case Op::Int_LT: return make_lit_bool(x < y);
            case Op::Int_GT: return make_lit_bool(x > y);
            default: break;
        }
    }

    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::BinOp;
    e->op_tag = op;
    e->ch = {a, b};
    e->hash = mix(mix(mix(0xB100U, static_cast<uint64_t>(op)), a->hash), b->hash);
    e->atoms = merge_atoms(a->atoms, b->atoms);
    return expr_table().intern(e);
}

inline ExprRef make_nary(Op op, std::vector<ExprRef> children) {
    // 1. Flatten same-op children.
    std::vector<ExprRef> flat;
    flat.reserve(children.size() * 2);
    for (auto& c : children) {
        if (c->kind == ExprKind::NAry && c->op_tag == op) {
            for (auto& cc : c->ch) flat.push_back(cc);
        } else {
            flat.push_back(c);
        }
    }
    // 2. Constant folding & identity removal.
    bool is_float_op = (op == Op::Float_Add || op == Op::Float_Mul ||
                        op == Op::Float_Min || op == Op::Float_Max ||
                        op == Op::Float_EQ);
    bool is_int_op = (op == Op::Int_Add || op == Op::Int_Mul ||
                      op == Op::Int_Min || op == Op::Int_Max ||
                      op == Op::Int_EQ);
    bool is_bool_op = (op == Op::Bool_And || op == Op::Bool_Or || op == Op::Bool_Xor);
    (void)is_bool_op;

    if (is_float_op) {
        double acc = 0.0;
        bool have_acc = false;
        std::vector<ExprRef> rest;
        for (auto& c : flat) {
            if (is_lit_float(c)) {
                double v = lit_float_v(c);
                if (!have_acc) { acc = v; have_acc = true; }
                else {
                    switch (op) {
                        case Op::Float_Add: acc += v; break;
                        case Op::Float_Mul: acc *= v; break;
                        case Op::Float_Min: acc = std::min(acc, v); break;
                        case Op::Float_Max: acc = std::max(acc, v); break;
                        case Op::Float_EQ: rest.push_back(c); break;  // can't fold across >2
                        default: break;
                    }
                }
            } else rest.push_back(c);
        }
        // identity for Add: 0;  for Mul: 1;  Mul x 0 = 0.
        if (have_acc) {
            if (op == Op::Float_Mul && acc == 0.0) return make_lit_float(0.0);
            bool skip = (op == Op::Float_Add && acc == 0.0) ||
                        (op == Op::Float_Mul && acc == 1.0);
            if (!skip || rest.empty()) rest.push_back(make_lit_float(sym_clamp(acc)));
        }
        flat = std::move(rest);
    } else if (is_int_op) {
        int64_t acc = 0;
        bool have_acc = false;
        std::vector<ExprRef> rest;
        for (auto& c : flat) {
            if (is_lit_int(c)) {
                int64_t v = lit_int_v(c);
                if (!have_acc) { acc = v; have_acc = true; }
                else {
                    switch (op) {
                        case Op::Int_Add: acc += v; break;
                        case Op::Int_Mul: acc *= v; break;
                        case Op::Int_Min: acc = std::min(acc, v); break;
                        case Op::Int_Max: acc = std::max(acc, v); break;
                        default: break;
                    }
                }
            } else rest.push_back(c);
        }
        if (have_acc) {
            if (op == Op::Int_Mul && acc == 0) return make_lit_int(0);
            bool skip = (op == Op::Int_Add && acc == 0) || (op == Op::Int_Mul && acc == 1);
            if (!skip || rest.empty()) rest.push_back(make_lit_int(acc));
        }
        flat = std::move(rest);
    }

    if (flat.empty()) {
        // identity
        if (op == Op::Float_Add) return make_lit_float(0.0);
        if (op == Op::Float_Mul) return make_lit_float(1.0);
        if (op == Op::Int_Add) return make_lit_int(0);
        if (op == Op::Int_Mul) return make_lit_int(1);
        if (op == Op::Bool_And) return make_lit_bool(true);
        if (op == Op::Bool_Or) return make_lit_bool(false);
        if (op == Op::Bool_Xor) return make_lit_bool(false);
        if (op == Op::Float_Min || op == Op::Float_Max) return make_lit_float(0.0);
        if (op == Op::Int_Min || op == Op::Int_Max) return make_lit_int(0);
    }
    if (flat.size() == 1) return flat[0];

    // 3. Add-specific: pair up x + (-x) = 0.
    if (op == Op::Float_Add) {
        std::vector<ExprRef> remaining;
        std::vector<bool> consumed(flat.size(), false);
        for (size_t i = 0; i < flat.size(); ++i) {
            if (consumed[i]) continue;
            bool paired = false;
            ExprRef neg_i = is_neg(flat[i]) ? flat[i]->ch[0] : ExprRef{};
            for (size_t j = i + 1; j < flat.size(); ++j) {
                if (consumed[j]) continue;
                // x + (-x)
                if (is_neg(flat[j]) && flat[j]->ch[0].get() == flat[i].get()) {
                    consumed[i] = consumed[j] = true; paired = true; break;
                }
                // (-x) + x
                if (neg_i && neg_i.get() == flat[j].get()) {
                    consumed[i] = consumed[j] = true; paired = true; break;
                }
            }
            if (!paired) remaining.push_back(flat[i]);
        }
        if (remaining.empty()) return make_lit_float(0.0);
        if (remaining.size() == 1) return remaining[0];
        flat = std::move(remaining);
    }

    // 4. Sort children by hash for canonical order.
    std::sort(flat.begin(), flat.end(), [](const ExprRef& a, const ExprRef& b){
        return a->hash < b->hash;
    });

    // 4b. Idempotent NAry normalization: Min/Max/And/Or — repeated children
    // are no-ops semantically (a ⊓ a = a, a ⊔ a = a, a ∧ a = a, a ∨ a = a),
    // so collapse consecutive duplicates after the canonical sort.  This is
    // critical for permutation-invariance checks on commutative reductions
    // whose argument lists differ only in multiplicity.
    {
        bool is_idempotent = (op == Op::Float_Min || op == Op::Float_Max ||
                              op == Op::Int_Min   || op == Op::Int_Max   ||
                              op == Op::Bool_And  || op == Op::Bool_Or);
        if (is_idempotent && flat.size() > 1) {
            flat.erase(std::unique(flat.begin(), flat.end(),
                          [](const ExprRef& a, const ExprRef& b){ return a.get() == b.get(); }),
                       flat.end());
            if (flat.size() == 1) return flat[0];
        }
    }

    // 4c. Bool_Xor: pair-cancel identical children (a ⊕ a = false).  Keep
    // only the parity of each unique child group.
    if (op == Op::Bool_Xor && flat.size() > 1) {
        std::vector<ExprRef> keep;
        keep.reserve(flat.size());
        for (size_t i = 0; i < flat.size(); ) {
            size_t j = i + 1;
            while (j < flat.size() && flat[j].get() == flat[i].get()) ++j;
            size_t count = j - i;
            if (count % 2 == 1) keep.push_back(flat[i]);
            i = j;
        }
        if (keep.empty()) return make_lit_bool(false);
        if (keep.size() == 1) return keep[0];
        flat = std::move(keep);
    }

    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::NAry;
    e->op_tag = op;
    e->ch = flat;
    uint64_t h = mix(0xCAFEU, static_cast<uint64_t>(op));
    std::vector<int> atoms;
    for (auto& c : flat) {
        h = mix(h, c->hash);
        atoms = merge_atoms(atoms, c->atoms);
    }
    e->hash = h;
    e->atoms = std::move(atoms);
    return expr_table().intern(e);
}

inline ExprRef make_ifelse(ExprRef cond, ExprRef then_, ExprRef else_) {
    // Concrete bool → pick one.
    if (is_lit_bool(cond)) return lit_bool_v(cond) ? then_ : else_;
    // Branches equal → return either.
    if (then_.get() == else_.get()) return then_;
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::IfElse;
    e->ch = {cond, then_, else_};
    e->hash = mix(mix(mix(0x1FE15EU, cond->hash), then_->hash), else_->hash);
    e->atoms = merge_atoms(merge_atoms(cond->atoms, then_->atoms), else_->atoms);
    return expr_table().intern(e);
}

// ===================== substitute =====================================

inline ExprRef substitute(const ExprRef& e, const AtomMap& map) {
    switch (e->kind) {
        case ExprKind::LitFloat:
        case ExprKind::LitInt:
        case ExprKind::LitBool:
            return e;
        case ExprKind::Atom: {
            int enc = encode_atom(e->atom_kind, e->idx);
            auto it = map.table.find(enc);
            ExprRef out = e;
            if (it != map.table.end()) {
                int newenc = it->second;
                if (newenc != enc) {
                    out = make_atom(atom_kind_of(newenc), atom_idx_of(newenc));
                }
            }
            auto nit = map.negate.find(encode_atom(out->atom_kind, out->idx));
            if (nit != map.negate.end() && nit->second) {
                out = make_unop(Op::Float_Neg, out);
            }
            return out;
        }
        case ExprKind::Opaque: {
            std::vector<int> new_atoms;
            new_atoms.reserve(e->atoms.size());
            for (int a : e->atoms) {
                auto it = map.table.find(a);
                new_atoms.push_back(it == map.table.end() ? a : it->second);
            }
            return make_opaque(std::move(new_atoms));
        }
        case ExprKind::UnOp: {
            ExprRef a = substitute(e->ch[0], map);
            if (a.get() == e->ch[0].get()) return e;
            return make_unop(e->op_tag, a);
        }
        case ExprKind::BinOp: {
            ExprRef a = substitute(e->ch[0], map);
            ExprRef b = substitute(e->ch[1], map);
            if (a.get() == e->ch[0].get() && b.get() == e->ch[1].get()) return e;
            return make_binop(e->op_tag, a, b);
        }
        case ExprKind::NAry: {
            std::vector<ExprRef> nc;
            nc.reserve(e->ch.size());
            bool changed = false;
            for (auto& c : e->ch) {
                ExprRef n = substitute(c, map);
                if (n.get() != c.get()) changed = true;
                nc.push_back(std::move(n));
            }
            if (!changed) return e;
            return make_nary(e->op_tag, std::move(nc));
        }
        case ExprKind::IfElse: {
            ExprRef c = substitute(e->ch[0], map);
            ExprRef t = substitute(e->ch[1], map);
            ExprRef f = substitute(e->ch[2], map);
            if (c.get() == e->ch[0].get() && t.get() == e->ch[1].get() && f.get() == e->ch[2].get())
                return e;
            return make_ifelse(c, t, f);
        }
    }
    return e;
}

}  // namespace pushgp_cpp
