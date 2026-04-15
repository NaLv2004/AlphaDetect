/*
 * evaluator_bp.cpp — C++ Structured BP Stack Decoder for MIMO Algorithm Discovery
 *
 * Self-contained: no external dependencies.
 * Implements the 4-program BP genome evaluation:
 *   prog_down:   F_down(M_parent_down, C_i) -> M_i_down
 *   prog_up:     F_up({C_j, M_j_up}_children) -> M_i_up
 *   prog_belief: F_belief(cum_dist, M_down, M_up) -> score
 *   prog_halt:   H_halt(old_root_m_up, new_root_m_up) -> bool
 *
 * Compile (MSVC):
 *   cl.exe /EHsc /O2 /openmp /std:c++17 evaluator_bp.cpp /LD /Fe:evaluator_bp.dll
 */

#include "evaluator_bp.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <complex>
#include <vector>
#include <queue>
#include <functional>
#include <limits>
#include <atomic>

using cx = std::complex<double>;

// =========================================================================
// Configuration
// =========================================================================
static constexpr int MAX_DIM      = 32;
static constexpr int MAX_NODES    = 8192;
static constexpr int STACK_DEPTH  = 64;
static constexpr int N_MEM        = 16;
static constexpr int MAX_CTRL_ITER = 20;
static constexpr int MAX_DOTIMES   = 15;

// =========================================================================
// Opcodes — MUST match cpp_bridge.py OPCODE_MAP
// Keep identical to evaluator.h + new BP opcodes at the end
// =========================================================================
enum Op : uint16_t {
    // Stack manipulation (17)
    OP_FLOAT_POP = 0, OP_FLOAT_DUP, OP_FLOAT_SWAP, OP_FLOAT_ROT,
    OP_INT_POP, OP_INT_DUP, OP_INT_SWAP,
    OP_BOOL_POP, OP_BOOL_DUP,
    OP_VEC_POP, OP_VEC_DUP, OP_VEC_SWAP,
    OP_MAT_POP, OP_MAT_DUP,
    OP_NODE_POP, OP_NODE_DUP, OP_NODE_SWAP,
    // Float arithmetic (13)
    OP_FLOAT_ADD, OP_FLOAT_SUB, OP_FLOAT_MUL, OP_FLOAT_DIV,
    OP_FLOAT_ABS, OP_FLOAT_NEG, OP_FLOAT_SQRT, OP_FLOAT_SQUARE,
    OP_FLOAT_MIN, OP_FLOAT_MAX, OP_FLOAT_EXP, OP_FLOAT_LOG, OP_FLOAT_TANH,
    // Comparisons (4)
    OP_FLOAT_LT, OP_FLOAT_GT, OP_INT_LT, OP_INT_GT,
    // Int arithmetic (4)
    OP_INT_ADD, OP_INT_SUB, OP_INT_INC, OP_INT_DEC,
    // Bool logic (3)
    OP_BOOL_AND, OP_BOOL_OR, OP_BOOL_NOT,
    // Tensor ops (11)
    OP_MAT_VECMUL, OP_VEC_ADD, OP_VEC_SUB, OP_VEC_DOT, OP_VEC_NORM2,
    OP_VEC_SCALE, OP_VEC_ELEMENTAT, OP_MAT_ELEMENTAT, OP_MAT_ROW,
    OP_VEC_LEN, OP_MAT_ROWS,
    // Peek access (6)
    OP_MAT_PEEKAT, OP_VEC_PEEKAT,
    OP_MAT_PEEKATIM, OP_VEC_PEEKATIM,
    OP_VEC_SECONDPEEKAT, OP_VEC_SECONDPEEKATIM,
    // High-level (2)
    OP_VEC_GETRESIDUE, OP_FLOAT_GETMMSELB,
    // Node memory (7)
    OP_NODE_READMEM, OP_NODE_WRITEMEM,
    OP_NODE_GETCUMDIST, OP_NODE_GETLOCALDIST,
    OP_NODE_GETSYMRE, OP_NODE_GETSYMIM, OP_NODE_GETLAYER,
    // Graph navigation (6)
    OP_GRAPH_GETROOT, OP_NODE_GETPARENT,
    OP_NODE_NUMCHILDREN, OP_NODE_CHILDAT,
    OP_GRAPH_NODECOUNT, OP_GRAPH_FRONTIERCOUNT,
    // BP-essential (3)
    OP_NODE_GETSCORE, OP_NODE_SETSCORE, OP_NODE_ISEXPANDED,
    // Constants (11)
    OP_FLOAT_CONST0, OP_FLOAT_CONST1, OP_FLOAT_CONSTHALF,
    OP_FLOAT_CONSTNEG1, OP_FLOAT_CONST2, OP_FLOAT_CONST0_1,
    OP_INT_CONST0, OP_INT_CONST1, OP_INT_CONST2,
    OP_BOOL_TRUE, OP_BOOL_FALSE,
    // Environment (2)
    OP_FLOAT_GETNOISEVAR, OP_INT_GETNUMSYMBOLS,
    // Type conversion (2)
    OP_FLOAT_FROMINT, OP_INT_FROMFLOAT,
    // Control flow (8)
    OP_EXEC_IF, OP_EXEC_WHILE, OP_EXEC_DOTIMES,
    OP_NODE_FOREACHCHILD, OP_NODE_FOREACHSIBLING, OP_NODE_FOREACHANCESTOR,
    OP_EXEC_FOREACHSYMBOL, OP_EXEC_MINOVERSYMBOLS,
    // Block delimiters
    OP_BLOCK_START, OP_BLOCK_END,
    // NEW: BP message-passing opcodes
    OP_NODE_GETMUP, OP_NODE_SETMUP,
    OP_NODE_GETMDOWN, OP_NODE_SETMDOWN,
    // Evolved constant: next 2 ints encode a double (little-endian)
    OP_FLOAT_PUSHIMMEDIATE,
    // Float inverse (1/x)
    OP_FLOAT_INV,
    // Min-reduce over children (A* heuristic support)
    OP_NODE_FOREACHCHILDMIN,

    OP_COUNT
};


// =========================================================================
// Small complex linear algebra
// =========================================================================
struct CxVec {
    cx data[MAX_DIM];
    int len = 0;
    CxVec() : len(0) { std::memset(data, 0, sizeof(data)); }
    CxVec(int n) : len(n) { std::memset(data, 0, sizeof(data)); }
};

struct CxMat {
    cx data[MAX_DIM][MAX_DIM];
    int rows = 0, cols = 0;
    CxMat() : rows(0), cols(0) { std::memset(data, 0, sizeof(data)); }
    CxMat(int r, int c) : rows(r), cols(c) { std::memset(data, 0, sizeof(data)); }
};

static void qr_mgs(const CxMat& H, CxMat& Q, CxMat& R) {
    int Nr = H.rows, Nt = H.cols;
    Q = CxMat(Nr, Nt);
    R = CxMat(Nt, Nt);
    for (int j = 0; j < Nt; j++)
        for (int i = 0; i < Nr; i++)
            Q.data[i][j] = H.data[i][j];
    for (int j = 0; j < Nt; j++) {
        for (int k = 0; k < j; k++) {
            cx dot(0,0);
            for (int i = 0; i < Nr; i++) dot += std::conj(Q.data[i][k]) * Q.data[i][j];
            R.data[k][j] = dot;
            for (int i = 0; i < Nr; i++) Q.data[i][j] -= dot * Q.data[i][k];
        }
        double nrm = 0;
        for (int i = 0; i < Nr; i++) nrm += std::norm(Q.data[i][j]);
        nrm = std::sqrt(nrm);
        R.data[j][j] = cx(nrm, 0);
        if (nrm > 1e-15)
            for (int i = 0; i < Nr; i++) Q.data[i][j] /= nrm;
    }
}

static bool solve_linear(cx A[MAX_DIM][MAX_DIM], cx b[MAX_DIM], cx x[MAX_DIM], int n) {
    cx Ab[MAX_DIM][MAX_DIM+1];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) Ab[i][j] = A[i][j];
        Ab[i][n] = b[i];
    }
    for (int col = 0; col < n; col++) {
        int best = col;
        double best_abs = std::abs(Ab[col][col]);
        for (int row = col+1; row < n; row++) {
            double a = std::abs(Ab[row][col]);
            if (a > best_abs) { best_abs = a; best = row; }
        }
        if (best_abs < 1e-14) return false;
        if (best != col)
            for (int j = col; j <= n; j++) std::swap(Ab[col][j], Ab[best][j]);
        for (int row = col+1; row < n; row++) {
            cx factor = Ab[row][col] / Ab[col][col];
            for (int j = col; j <= n; j++) Ab[row][j] -= factor * Ab[col][j];
        }
    }
    for (int i = n-1; i >= 0; i--) {
        x[i] = Ab[i][n];
        for (int j = i+1; j < n; j++) x[i] -= Ab[i][j] * x[j];
        if (std::abs(Ab[i][i]) < 1e-14) return false;
        x[i] /= Ab[i][i];
    }
    return true;
}


// =========================================================================
// Tree Node (with m_up, m_down for BP)
// =========================================================================
struct TreeNode {
    int id = -1;
    int layer = 0;
    cx  symbol = 0;
    double local_dist = 0;
    double cum_dist = 0;
    double score = 0;
    double m_up = 0;     // BP upward message
    double m_down = 0;   // BP downward message
    int queue_version = 0;
    bool is_expanded = false;
    double mem[N_MEM] = {};
    int parent_id = -1;
    int children[64];
    int n_children = 0;
    cx partial_symbols[MAX_DIM];
    int n_partial = 0;
};

struct SearchTree {
    TreeNode* nodes;
    int capacity;
    int n_nodes = 0;
    int root_id = -1;

    SearchTree(int cap = MAX_NODES) : capacity(cap) {
        nodes = new TreeNode[cap];
    }
    ~SearchTree() { delete[] nodes; }

    void reset() { n_nodes = 0; root_id = -1; }

    int create_root(int layer) {
        int id = n_nodes++;
        nodes[id] = TreeNode();
        nodes[id].id = id;
        nodes[id].layer = layer;
        root_id = id;
        return id;
    }

    int add_child(int parent_id, int layer, cx sym,
                  double ld, double cd,
                  const cx* partial, int n_partial) {
        if (n_nodes >= capacity) return -1;
        int id = n_nodes++;
        TreeNode& nd = nodes[id];
        nd = TreeNode();
        nd.id = id;
        nd.layer = layer;
        nd.symbol = sym;
        nd.local_dist = ld;
        nd.cum_dist = cd;
        nd.score = cd;
        nd.parent_id = parent_id;
        nd.n_partial = n_partial;
        for (int i = 0; i < n_partial; i++) nd.partial_symbols[i] = partial[i];
        TreeNode& p = nodes[parent_id];
        if (p.n_children < 64) p.children[p.n_children++] = id;
        return id;
    }

    void mark_expanded(int id) { nodes[id].is_expanded = true; }

    // BFS order for sweeps
    void bfs_order(int* out, int& count) const {
        count = 0;
        if (root_id < 0) return;
        int front = 0;
        out[count++] = root_id;
        while (front < count) {
            int nid = out[front++];
            const TreeNode& nd = nodes[nid];
            for (int i = 0; i < nd.n_children; i++)
                out[count++] = nd.children[i];
        }
    }
};


// =========================================================================
// Typed Stacks
// =========================================================================
template<typename T, int Cap = STACK_DEPTH>
struct Stack {
    T data[Cap];
    int top = 0;
    void clear() { top = 0; }
    int depth() const { return top; }
    bool has_pop() const { return top > 0; }
    bool has_n(int n) const { return top >= n; }
    void push(T v) { if (top < Cap) data[top++] = v; }
    T pop() { if (top > 0) return data[--top]; return T{}; }
    T peek() const { if (top > 0) return data[top-1]; return T{}; }
    void dup()  { if (top > 0 && top < Cap) { data[top] = data[top-1]; top++; } }
    void swap() { if (top >= 2) std::swap(data[top-1], data[top-2]); }
    void rot()  {
        if (top >= 3) {
            T tmp = data[top-3];
            data[top-3] = data[top-2];
            data[top-2] = data[top-1];
            data[top-1] = tmp;
        }
    }
};


// =========================================================================
// Push VM (minimal version for BP programs)
// =========================================================================
struct PushVM {
    Stack<double>    fs;
    Stack<int>       is;
    Stack<int>       bs;
    Stack<CxVec, 16> vs;
    Stack<CxMat, 8>  ms;
    Stack<int>       ns;

    SearchTree* tree = nullptr;
    const cx* constellation = nullptr;
    int M = 0;
    int Nt = 0, Nr = 0;
    double noise_var = 1.0;
    int64_t flops_max = 2000000;
    int step_max  = 1500;
    int64_t flops_count = 0;
    int step_count  = 0;
    bool halted = false;

    void reset() {
        fs.clear(); is.clear(); bs.clear();
        vs.clear(); ms.clear(); ns.clear();
        flops_count = 0; step_count = 0; halted = false;
    }

    void charge_flops(int c) {
        flops_count += c;
        if (flops_count > flops_max) halted = true;
    }

    void charge_step() {
        step_count++;
        if (step_count > step_max) halted = true;
    }

    static inline double safe_float(double v) {
        if (std::isnan(v) || std::isinf(v)) return 0.0;
        return v;
    }

    static int find_block_end(const int* prog, int pos, int end) {
        int depth = 1;
        int i = pos + 1;
        while (i < end && depth > 0) {
            if (prog[i] == OP_BLOCK_START) depth++;
            else if (prog[i] == OP_BLOCK_END) depth--;
            i++;
        }
        return i;
    }

    // Execute a block of instructions
    void execute_block(const int* prog, int start, int end) {
        int pc = start;
        while (pc < end && !halted) {
            pc = execute_one(prog, pc, end);
        }
    }

    // Run a program, return float stack top
    double run(const int* prog, int len) {
        execute_block(prog, 0, len);
        if (fs.has_pop()) {
            double r = fs.peek();
            if (std::isfinite(r)) return r;
        }
        return std::numeric_limits<double>::infinity();
    }

    // Run a program, return bool stack top
    bool run_bool(const int* prog, int len) {
        execute_block(prog, 0, len);
        if (bs.has_pop()) return bs.peek() != 0;
        return true; // default: halt
    }

    int execute_one(const int* prog, int pc, int end) {
        if (pc >= end || halted) return end;
        charge_step();
        if (halted) return end;

        int op = prog[pc++];

        switch ((Op)op) {

        // Stack manipulation
        case OP_FLOAT_POP:  fs.pop(); break;
        case OP_FLOAT_DUP:  fs.dup(); break;
        case OP_FLOAT_SWAP: fs.swap(); break;
        case OP_FLOAT_ROT:  fs.rot(); break;
        case OP_INT_POP:    is.pop(); break;
        case OP_INT_DUP:    is.dup(); break;
        case OP_INT_SWAP:   is.swap(); break;
        case OP_BOOL_POP:   bs.pop(); break;
        case OP_BOOL_DUP:   bs.dup(); break;
        case OP_VEC_POP:    vs.pop(); break;
        case OP_VEC_DUP:    vs.dup(); break;
        case OP_VEC_SWAP:   vs.swap(); break;
        case OP_MAT_POP:    ms.pop(); break;
        case OP_MAT_DUP:    ms.dup(); break;
        case OP_NODE_POP:   ns.pop(); break;
        case OP_NODE_DUP:   ns.dup(); break;
        case OP_NODE_SWAP:  ns.swap(); break;

        // Float arithmetic
        case OP_FLOAT_ADD:
            if (fs.has_n(2)) { double a=fs.pop(), b=fs.pop(); charge_flops(1); fs.push(safe_float(a+b)); }
            break;
        case OP_FLOAT_SUB:
            if (fs.has_n(2)) { double a=fs.pop(), b=fs.pop(); charge_flops(1); fs.push(safe_float(b-a)); }
            break;
        case OP_FLOAT_MUL:
            if (fs.has_n(2)) { double a=fs.pop(), b=fs.pop(); charge_flops(1); fs.push(safe_float(a*b)); }
            break;
        case OP_FLOAT_DIV:
            if (fs.has_n(2)) { double a=fs.pop(), b=fs.pop(); charge_flops(1);
                if (std::abs(a) > 1e-30) fs.push(safe_float(b/a));
                else fs.push(0.0);
            } break;
        case OP_FLOAT_ABS:
            if (fs.has_pop()) { charge_flops(1); fs.push(std::abs(fs.pop())); } break;
        case OP_FLOAT_NEG:
            if (fs.has_pop()) { fs.push(-fs.pop()); } break;
        case OP_FLOAT_SQRT:
            if (fs.has_pop()) { double a=fs.pop(); charge_flops(1);
                fs.push(a >= 0 ? std::sqrt(a) : 0.0); } break;
        case OP_FLOAT_SQUARE:
            if (fs.has_pop()) { double a=fs.pop(); charge_flops(1); fs.push(safe_float(a*a)); } break;
        case OP_FLOAT_MIN:
            if (fs.has_n(2)) { double a=fs.pop(), b=fs.pop(); fs.push(std::min(a,b)); } break;
        case OP_FLOAT_MAX:
            if (fs.has_n(2)) { double a=fs.pop(), b=fs.pop(); fs.push(std::max(a,b)); } break;
        case OP_FLOAT_EXP:
            if (fs.has_pop()) { double a=fs.pop(); charge_flops(5);
                a = std::max(-20.0, std::min(a, 20.0));
                fs.push(std::exp(a)); } break;
        case OP_FLOAT_LOG:
            if (fs.has_pop()) { double a=fs.pop(); charge_flops(5);
                fs.push(a > 1e-30 ? std::log(a) : -30.0); } break;
        case OP_FLOAT_TANH:
            if (fs.has_pop()) { double a=fs.pop(); charge_flops(5);
                fs.push(std::tanh(std::max(-20.0, std::min(a, 20.0)))); } break;
        case OP_FLOAT_INV:
            if (fs.has_pop()) { double a=fs.pop(); charge_flops(1);
                if (std::abs(a) > 1e-30) fs.push(1.0/a);
                else fs.push(0.0); } break;

        // Comparisons
        case OP_FLOAT_LT:
            if (fs.has_n(2)) { double a=fs.pop(), b=fs.pop(); bs.push(b < a ? 1 : 0); } break;
        case OP_FLOAT_GT:
            if (fs.has_n(2)) { double a=fs.pop(), b=fs.pop(); bs.push(b > a ? 1 : 0); } break;
        case OP_INT_LT:
            if (is.has_n(2)) { int a=is.pop(), b=is.pop(); bs.push(b < a ? 1 : 0); } break;
        case OP_INT_GT:
            if (is.has_n(2)) { int a=is.pop(), b=is.pop(); bs.push(b > a ? 1 : 0); } break;

        // Int arithmetic
        case OP_INT_ADD:
            if (is.has_n(2)) { int a=is.pop(), b=is.pop(); is.push(a+b); } break;
        case OP_INT_SUB:
            if (is.has_n(2)) { int a=is.pop(), b=is.pop(); is.push(b-a); } break;
        case OP_INT_INC:
            if (is.has_pop()) { is.push(is.pop()+1); } break;
        case OP_INT_DEC:
            if (is.has_pop()) { is.push(is.pop()-1); } break;

        // Bool logic
        case OP_BOOL_AND:
            if (bs.has_n(2)) { int a=bs.pop(), b=bs.pop(); bs.push(a&&b ? 1 : 0); } break;
        case OP_BOOL_OR:
            if (bs.has_n(2)) { int a=bs.pop(), b=bs.pop(); bs.push(a||b ? 1 : 0); } break;
        case OP_BOOL_NOT:
            if (bs.has_pop()) { bs.push(bs.pop() ? 0 : 1); } break;

        // Tensor ops
        case OP_MAT_VECMUL:
            if (ms.has_pop() && vs.has_pop()) {
                CxMat A = ms.pop(); CxVec x = vs.pop();
                if (A.cols == x.len) {
                    CxVec y; y.len = A.rows;
                    for (int i=0; i<A.rows; i++) {
                        cx s(0,0);
                        for (int j=0; j<A.cols; j++) s += A.data[i][j] * x.data[j];
                        y.data[i] = s;
                    }
                    charge_flops(A.rows * A.cols * 8);
                    vs.push(y);
                }
            } break;
        case OP_VEC_ADD:
            if (vs.has_n(2)) { CxVec a=vs.pop(), b=vs.pop();
                CxVec r; r.len = std::min(a.len, b.len);
                for (int i=0; i<r.len; i++) r.data[i] = a.data[i] + b.data[i];
                charge_flops(r.len * 2); vs.push(r);
            } break;
        case OP_VEC_SUB:
            if (vs.has_n(2)) { CxVec a=vs.pop(), b=vs.pop();
                CxVec r; r.len = std::min(a.len, b.len);
                for (int i=0; i<r.len; i++) r.data[i] = b.data[i] - a.data[i];
                charge_flops(r.len * 2); vs.push(r);
            } break;
        case OP_VEC_DOT:
            if (vs.has_n(2)) { CxVec a=vs.pop(), b=vs.pop();
                int n = std::min(a.len, b.len);
                cx s(0,0);
                for (int i=0; i<n; i++) s += std::conj(a.data[i]) * b.data[i];
                charge_flops(n * 8);
                fs.push(std::real(s));
            } break;
        case OP_VEC_NORM2:
            if (vs.has_pop()) { CxVec a=vs.pop();
                double s=0;
                for (int i=0; i<a.len; i++) s += std::norm(a.data[i]);
                charge_flops(a.len * 4);
                fs.push(s);
            } break;
        case OP_VEC_SCALE:
            if (fs.has_pop() && vs.has_pop()) {
                double s = fs.pop(); CxVec v = vs.pop();
                for (int i=0; i<v.len; i++) v.data[i] *= s;
                charge_flops(v.len * 2); vs.push(v);
            } break;
        case OP_VEC_ELEMENTAT:
            if (is.has_pop() && vs.has_pop()) {
                int idx = is.pop(); CxVec v = vs.pop();
                if (idx>=0 && idx<v.len) fs.push(std::real(v.data[idx]));
            } break;
        case OP_MAT_ELEMENTAT:
            if (is.has_n(2) && ms.has_pop()) {
                int c=is.pop(), r=is.pop(); CxMat A=ms.pop();
                if (r>=0 && r<A.rows && c>=0 && c<A.cols)
                    fs.push(std::real(A.data[r][c]));
            } break;
        case OP_MAT_ROW:
            if (is.has_pop() && ms.depth()>0) {
                int r=is.pop(); CxMat& A=ms.data[ms.top-1];
                if (r>=0 && r<A.rows) {
                    CxVec v; v.len = A.cols;
                    for (int j=0; j<A.cols; j++) v.data[j] = A.data[r][j];
                    vs.push(v);
                }
            } break;
        case OP_VEC_LEN:
            if (vs.depth()>0) is.push(vs.data[vs.top-1].len); break;
        case OP_MAT_ROWS:
            if (ms.depth()>0) is.push(ms.data[ms.top-1].rows); break;

        // Peek access
        case OP_MAT_PEEKAT:
            if (is.has_n(2) && ms.depth()>0) {
                int c=is.pop(), r=is.pop(); CxMat& A=ms.data[ms.top-1];
                if (r>=0 && r<A.rows && c>=0 && c<A.cols) fs.push(std::real(A.data[r][c]));
            } break;
        case OP_VEC_PEEKAT:
            if (is.has_pop() && vs.depth()>0) {
                int idx=is.pop(); CxVec& v=vs.data[vs.top-1];
                if (idx>=0 && idx<v.len) fs.push(std::real(v.data[idx]));
            } break;
        case OP_MAT_PEEKATIM:
            if (is.has_n(2) && ms.depth()>0) {
                int c=is.pop(), r=is.pop(); CxMat& A=ms.data[ms.top-1];
                if (r>=0 && r<A.rows && c>=0 && c<A.cols) fs.push(std::imag(A.data[r][c]));
            } break;
        case OP_VEC_PEEKATIM:
            if (is.has_pop() && vs.depth()>0) {
                int idx=is.pop(); CxVec& v=vs.data[vs.top-1];
                if (idx>=0 && idx<v.len) fs.push(std::imag(v.data[idx]));
            } break;
        case OP_VEC_SECONDPEEKAT:
            if (is.has_pop() && vs.has_n(2)) {
                int idx=is.pop(); CxVec& v=vs.data[vs.top-2];
                if (idx>=0 && idx<v.len) fs.push(std::real(v.data[idx]));
            } break;
        case OP_VEC_SECONDPEEKATIM:
            if (is.has_pop() && vs.has_n(2)) {
                int idx=is.pop(); CxVec& v=vs.data[vs.top-2];
                if (idx>=0 && idx<v.len) fs.push(std::imag(v.data[idx]));
            } break;

        // High-level: residue
        case OP_VEC_GETRESIDUE:
            if (is.depth()>0 && vs.has_n(2) && ms.depth()>0) {
                int k=is.peek();
                CxVec& xp=vs.data[vs.top-1]; CxVec& yt=vs.data[vs.top-2]; CxMat& R=ms.data[ms.top-1];
                if (k>0 && k<=R.cols && xp.len==R.cols-k) {
                    CxVec res; res.len=k;
                    int np=xp.len;
                    for (int i=0; i<k; i++) { cx interf(0,0);
                        for (int j=0; j<np; j++) interf += R.data[i][k+j]*xp.data[np-1-j];
                        res.data[i] = yt.data[i]-interf; }
                    charge_flops(k*np*8); vs.push(res);
                }
            } break;

        // High-level: MMSE lower bound
        case OP_FLOAT_GETMMSELB:
            if (is.depth()>0 && vs.has_n(2) && ms.depth()>0) {
                int k=is.peek();
                CxVec& xp=vs.data[vs.top-1]; CxVec& yt=vs.data[vs.top-2]; CxMat& R=ms.data[ms.top-1];
                double nv = noise_var;
                if (k>0 && k<=R.cols && xp.len==R.cols-k) {
                    int np=xp.len;
                    cx residue[MAX_DIM];
                    for (int i=0; i<k; i++) { cx interf(0,0);
                        for (int j=0; j<np; j++) interf += R.data[i][k+j]*xp.data[np-1-j];
                        residue[i] = yt.data[i]-interf; }
                    cx Gram[MAX_DIM][MAX_DIM];
                    for (int i=0; i<k; i++) { for (int j=0; j<k; j++) { cx s(0,0);
                        for (int m=0; m<k; m++) s += std::conj(R.data[m][i])*R.data[m][j];
                        Gram[i][j] = s; }
                        Gram[i][i] += cx(nv,0); }
                    cx Rh_r[MAX_DIM];
                    for (int i=0; i<k; i++) { cx s(0,0);
                        for (int m=0; m<k; m++) s += std::conj(R.data[m][i])*residue[m];
                        Rh_r[i] = s; }
                    cx t[MAX_DIM];
                    if (solve_linear(Gram, Rh_r, t, k)) {
                        cx rHr(0,0); for (int i=0; i<k; i++) rHr += std::conj(residue[i])*residue[i];
                        cx corr(0,0); for (int i=0; i<k; i++) corr += std::conj(Rh_r[i])*t[i];
                        double lb = std::max(0.0, std::real(rHr-corr));
                        charge_flops(k*k*k + k*k*(R.cols-k)*8);
                        fs.push(lb);
                    } else { fs.push(0.0); }
                }
            } break;

        // Node memory
        case OP_NODE_READMEM:
            if (is.has_pop() && ns.has_pop() && tree) {
                int slot=is.pop(); int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes && slot>=0 && slot<N_MEM)
                    fs.push(tree->nodes[nid].mem[slot]);
            } break;
        case OP_NODE_WRITEMEM:
            if (is.has_pop() && fs.has_pop() && ns.has_pop() && tree) {
                int slot=is.pop(); double val=fs.pop(); int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes && slot>=0 && slot<N_MEM && std::isfinite(val))
                    tree->nodes[nid].mem[slot] = val;
            } break;
        case OP_NODE_GETCUMDIST:
            if (ns.has_pop() && tree) { int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes) fs.push(tree->nodes[nid].cum_dist); } break;
        case OP_NODE_GETLOCALDIST:
            if (ns.has_pop() && tree) { int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes) fs.push(tree->nodes[nid].local_dist); } break;
        case OP_NODE_GETSYMRE:
            if (ns.has_pop() && tree) { int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes) fs.push(std::real(tree->nodes[nid].symbol)); } break;
        case OP_NODE_GETSYMIM:
            if (ns.has_pop() && tree) { int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes) fs.push(std::imag(tree->nodes[nid].symbol)); } break;
        case OP_NODE_GETLAYER:
            if (ns.has_pop() && tree) { int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes) is.push(tree->nodes[nid].layer); } break;

        // Graph navigation
        case OP_GRAPH_GETROOT:
            if (tree && tree->root_id>=0) ns.push(tree->root_id); break;
        case OP_NODE_GETPARENT:
            if (ns.has_pop() && tree) { int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes && tree->nodes[nid].parent_id>=0)
                    ns.push(tree->nodes[nid].parent_id); } break;
        case OP_NODE_NUMCHILDREN:
            if (ns.has_pop() && tree) { int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes) is.push(tree->nodes[nid].n_children); } break;
        case OP_NODE_CHILDAT:
            if (is.has_pop() && ns.has_pop() && tree) { int idx=is.pop(); int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes) { TreeNode& nd=tree->nodes[nid];
                    if (idx>=0 && idx<nd.n_children) ns.push(nd.children[idx]); }
            } break;
        case OP_GRAPH_NODECOUNT:
            if (tree) is.push(tree->n_nodes); break;
        case OP_GRAPH_FRONTIERCOUNT:
            // simplified: count non-expanded non-root
            if (tree) { int c=0; for (int i=0; i<tree->n_nodes; i++)
                if (!tree->nodes[i].is_expanded && i!=tree->root_id) c++;
                is.push(c); } break;

        // BP-essential
        case OP_NODE_GETSCORE:
            if (ns.has_pop() && tree) { int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes) fs.push(tree->nodes[nid].score); } break;
        case OP_NODE_SETSCORE:
            if (fs.has_pop() && ns.has_pop() && tree) { double val=fs.pop(); int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes && std::isfinite(val)) {
                    tree->nodes[nid].score = val; tree->nodes[nid].queue_version++; }
            } break;
        case OP_NODE_ISEXPANDED:
            if (ns.has_pop() && tree) { int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes) bs.push(tree->nodes[nid].is_expanded ? 1 : 0); } break;

        // NEW: BP message opcodes
        case OP_NODE_GETMUP:
            if (ns.has_pop() && tree) { int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes) fs.push(tree->nodes[nid].m_up); } break;
        case OP_NODE_SETMUP:
            if (fs.has_pop() && ns.has_pop() && tree) { double val=fs.pop(); int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes && std::isfinite(val))
                    tree->nodes[nid].m_up = val; } break;
        case OP_NODE_GETMDOWN:
            if (ns.has_pop() && tree) { int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes) fs.push(tree->nodes[nid].m_down); } break;
        case OP_NODE_SETMDOWN:
            if (fs.has_pop() && ns.has_pop() && tree) { double val=fs.pop(); int nid=ns.peek();
                if (nid>=0 && nid<tree->n_nodes && std::isfinite(val))
                    tree->nodes[nid].m_down = val; } break;

        // Constants
        case OP_FLOAT_CONST0:    fs.push(0.0); break;
        case OP_FLOAT_CONST1:    fs.push(1.0); break;
        case OP_FLOAT_CONSTHALF: fs.push(0.5); break;
        case OP_FLOAT_CONSTNEG1: fs.push(-1.0); break;
        case OP_FLOAT_CONST2:    fs.push(2.0); break;
        case OP_FLOAT_CONST0_1:  fs.push(0.1); break;
        case OP_INT_CONST0:      is.push(0); break;
        case OP_INT_CONST1:      is.push(1); break;
        case OP_INT_CONST2:      is.push(2); break;
        case OP_BOOL_TRUE:       bs.push(1); break;
        case OP_BOOL_FALSE:      bs.push(0); break;

        // Environment
        case OP_FLOAT_GETNOISEVAR: fs.push(noise_var); break;
        case OP_INT_GETNUMSYMBOLS: is.push(M); break;

        // Type conversion
        case OP_FLOAT_FROMINT:
            if (is.has_pop()) fs.push((double)is.pop()); break;
        case OP_INT_FROMFLOAT:
            if (fs.has_pop()) { double a=fs.pop();
                is.push((int)std::max(-1e6, std::min(a, 1e6))); } break;

        // Control flow
        case OP_EXEC_IF: {
            if (pc < end && prog[pc]==OP_BLOCK_START) {
                int ts=pc+1, te=find_block_end(prog,pc,end);
                int es=-1, ee=te;
                if (te<end && prog[te]==OP_BLOCK_START) { es=te+1; ee=find_block_end(prog,te,end); }
                if (bs.has_pop()) { if (bs.pop()) execute_block(prog,ts,te-1);
                    else if (es>=0) execute_block(prog,es,ee-1); }
                pc = ee;
            } break;
        }
        case OP_EXEC_WHILE: {
            if (pc<end && prog[pc]==OP_BLOCK_START) {
                int bs_=pc+1, be_=find_block_end(prog,pc,end);
                for (int iter=0; iter<MAX_CTRL_ITER && !halted; iter++) {
                    if (!bs.has_pop()) break; if (!bs.pop()) break;
                    execute_block(prog,bs_,be_-1); }
                pc = be_;
            } break;
        }
        case OP_EXEC_DOTIMES: {
            if (pc<end && prog[pc]==OP_BLOCK_START) {
                int bs_=pc+1, be_=find_block_end(prog,pc,end);
                int cnt=0; if (is.has_pop()) cnt=std::max(0,std::min(is.pop(),MAX_DOTIMES));
                for (int i=0; i<cnt && !halted; i++) execute_block(prog,bs_,be_-1);
                pc = be_;
            } break;
        }
        case OP_NODE_FOREACHCHILD: {
            if (pc<end && prog[pc]==OP_BLOCK_START) {
                int bs_=pc+1, be_=find_block_end(prog,pc,end);
                if (ns.has_pop() && tree) {
                    int nid=ns.peek();
                    if (nid>=0 && nid<tree->n_nodes) {
                        TreeNode& nd=tree->nodes[nid];
                        double acc=0;
                        int lim=std::min(nd.n_children, MAX_CTRL_ITER);
                        for (int i=0; i<lim && !halted; i++) {
                            int od=ns.depth(); ns.push(nd.children[i]);
                            execute_block(prog,bs_,be_-1);
                            if (fs.has_pop()) { double v=fs.pop(); if (std::isfinite(v)) acc+=v; }
                            while (ns.depth()>od) ns.pop();
                        }
                        fs.push(acc);
                    }
                }
                pc = be_;
            } break;
        }
        case OP_NODE_FOREACHCHILDMIN: {
            // MinReduce over children — enables A*-style lower-bound heuristics
            if (pc<end && prog[pc]==OP_BLOCK_START) {
                int bs_=pc+1, be_=find_block_end(prog,pc,end);
                if (ns.has_pop() && tree) {
                    int nid=ns.peek();
                    if (nid>=0 && nid<tree->n_nodes) {
                        TreeNode& nd=tree->nodes[nid];
                        double acc=std::numeric_limits<double>::infinity();
                        int lim=std::min(nd.n_children, MAX_CTRL_ITER);
                        for (int i=0; i<lim && !halted; i++) {
                            int od=ns.depth(); ns.push(nd.children[i]);
                            execute_block(prog,bs_,be_-1);
                            if (fs.has_pop()) { double v=fs.pop(); if (std::isfinite(v)) acc=std::min(acc,v); }
                            while (ns.depth()>od) ns.pop();
                        }
                        if (!std::isfinite(acc)) acc=0.0;
                        fs.push(acc);
                    }
                }
                pc = be_;
            } break;
        }
        case OP_NODE_FOREACHSIBLING: {
            if (pc<end && prog[pc]==OP_BLOCK_START) {
                int bs_=pc+1, be_=find_block_end(prog,pc,end);
                if (ns.has_pop() && tree) {
                    int nid=ns.peek();
                    if (nid>=0 && nid<tree->n_nodes && tree->nodes[nid].parent_id>=0) {
                        TreeNode& par=tree->nodes[tree->nodes[nid].parent_id];
                        double acc=0;
                        int lim=std::min(par.n_children, MAX_CTRL_ITER);
                        for (int i=0; i<lim && !halted; i++) {
                            int cid=par.children[i]; if (cid==nid) continue;
                            int od=ns.depth(); ns.push(cid);
                            execute_block(prog,bs_,be_-1);
                            if (fs.has_pop()) { double v=fs.pop(); if (std::isfinite(v)) acc+=v; }
                            while (ns.depth()>od) ns.pop();
                        }
                        fs.push(acc);
                    }
                }
                pc = be_;
            } break;
        }
        case OP_NODE_FOREACHANCESTOR: {
            if (pc<end && prog[pc]==OP_BLOCK_START) {
                int bs_=pc+1, be_=find_block_end(prog,pc,end);
                if (ns.has_pop() && tree) {
                    int nid=ns.peek(); double acc=0;
                    int cur = tree->nodes[nid].parent_id;
                    int cnt=0;
                    while (cur>=0 && cur!=tree->root_id && cnt<MAX_CTRL_ITER && !halted) {
                        int od=ns.depth(); ns.push(cur);
                        execute_block(prog,bs_,be_-1);
                        if (fs.has_pop()) { double v=fs.pop(); if (std::isfinite(v)) acc+=v; }
                        while (ns.depth()>od) ns.pop();
                        cur = tree->nodes[cur].parent_id; cnt++;
                    }
                    fs.push(acc);
                }
                pc = be_;
            } break;
        }
        case OP_EXEC_FOREACHSYMBOL: {
            if (pc<end && prog[pc]==OP_BLOCK_START) {
                int bs_=pc+1, be_=find_block_end(prog,pc,end);
                if (constellation && M>0) {
                    double acc=0;
                    for (int s=0; s<M && !halted; s++) {
                        fs.push(std::real(constellation[s]));
                        fs.push(std::imag(constellation[s]));
                        execute_block(prog,bs_,be_-1);
                        if (fs.has_pop()) { double v=fs.pop(); if (std::isfinite(v)) acc+=v; }
                    }
                    fs.push(acc);
                }
                pc = be_;
            } break;
        }
        case OP_EXEC_MINOVERSYMBOLS: {
            if (pc<end && prog[pc]==OP_BLOCK_START) {
                int bs_=pc+1, be_=find_block_end(prog,pc,end);
                if (constellation && M>0) {
                    double acc=1e30;
                    for (int s=0; s<M && !halted; s++) {
                        fs.push(std::real(constellation[s]));
                        fs.push(std::imag(constellation[s]));
                        execute_block(prog,bs_,be_-1);
                        if (fs.has_pop()) { double v=fs.pop(); if (std::isfinite(v)) acc=std::min(acc,v); }
                    }
                    if (!std::isfinite(acc)) acc=0.0;
                    fs.push(acc);
                }
                pc = be_;
            } break;
        }

        case OP_BLOCK_START: case OP_BLOCK_END: break;
        case OP_FLOAT_PUSHIMMEDIATE: {
            // Next 2 ints encode a double (little-endian uint32 pair)
            if (pc + 1 < end) {
                uint32_t parts[2];
                parts[0] = (uint32_t)prog[pc];
                parts[1] = (uint32_t)prog[pc+1];
                double val;
                std::memcpy(&val, parts, sizeof(double));
                if (std::isfinite(val)) fs.push(val);
                else fs.push(0.0);
                pc += 2;
            }
            break;
        }
        default: break;
        }
        return pc;
    }

    // Setup environment for a program execution on a specific node
    void setup_for_node(int node_id, const CxMat& R, const CxVec& y_tilde,
                        SearchTree* t, const cx* cons, int M_,
                        double nv, int Nt_, int Nr_) {
        reset();
        tree = t;
        constellation = cons;
        M = M_;
        noise_var = nv;
        Nt = Nt_; Nr = Nr_;

        CxMat Rc = R; ms.push(Rc);
        CxVec yt = y_tilde; vs.push(yt);
        TreeNode& nd = t->nodes[node_id];
        CxVec xp; xp.len = nd.n_partial;
        for (int i = 0; i < nd.n_partial; i++) xp.data[i] = nd.partial_symbols[i];
        vs.push(xp);
        ns.push(node_id);
        is.push(nd.layer);
    }
};


// =========================================================================
// BP Stack Decoder
// =========================================================================
struct BPStackDecoder {
    int Nt, Nr, M;
    const cx* constellation;
    int max_nodes;
    int max_bp_iters;
    PushVM vm;
    int total_bp_calls;

    struct PQEntry {
        double score;
        int queue_version;
        int counter;
        int node_id;
        bool operator>(const PQEntry& o) const {
            if (score != o.score) return score > o.score;
            return counter > o.counter;
        }
    };

    // Run full UP sweep (leaves→root)
    int64_t full_up_sweep(SearchTree& tree, const CxMat& R, const CxVec& y_tilde,
                      const int* prog_up, int up_len, double nv) {
        int64_t total_flops = 0;
        int bfs[MAX_NODES]; int bfs_count = 0;
        tree.bfs_order(bfs, bfs_count);

        // Process in reverse BFS order (leaves first)
        for (int i = bfs_count - 1; i >= 0; i--) {
            int nid = bfs[i];
            TreeNode& nd = tree.nodes[nid];
            if (nd.n_children > 0) {
                // Internal node: run F_up
                vm.setup_for_node(nid, R, y_tilde, &tree, constellation, M, nv, Nt, Nr);
                // Push children's (C_j, M_j_up) pairs
                for (int c = 0; c < nd.n_children; c++) {
                    TreeNode& child = tree.nodes[nd.children[c]];
                    vm.fs.push(child.local_dist);
                    vm.fs.push(child.m_up);
                }
                vm.is.push(nd.n_children);

                double result = vm.run(prog_up, up_len);
                total_flops += vm.flops_count;
                if (std::isfinite(result)) nd.m_up = result;
                total_bp_calls++;
            } else {
                // Leaf: m_up = local_dist (provides non-zero values for
                // F_up to aggregate; 0 creates a bootstrap trap where
                // all m_up values stay at 0 forever)
                nd.m_up = nd.local_dist;
            }
        }
        return total_flops;
    }

    // Run full DOWN sweep (root→leaves)
    int64_t full_down_sweep(SearchTree& tree, const CxMat& R, const CxVec& y_tilde,
                        const int* prog_down, int down_len, double nv) {
        int64_t total_flops = 0;
        int bfs[MAX_NODES]; int bfs_count = 0;
        tree.bfs_order(bfs, bfs_count);

        for (int i = 0; i < bfs_count; i++) {
            int nid = bfs[i];
            TreeNode& nd = tree.nodes[nid];
            if (nd.parent_id >= 0) {
                TreeNode& parent = tree.nodes[nd.parent_id];
                vm.setup_for_node(nid, R, y_tilde, &tree, constellation, M, nv, Nt, Nr);
                // Push: M_parent_down (bottom), C_i=local_dist (top)
                vm.fs.push(parent.m_down);
                vm.fs.push(nd.local_dist);

                double result = vm.run(prog_down, down_len);
                total_flops += vm.flops_count;
                if (std::isfinite(result))
                    nd.m_down = result;
                else
                    nd.m_down = parent.m_down + nd.local_dist;
                total_bp_calls++;
            }
        }
        return total_flops;
    }

    // Score ALL frontier nodes with F_belief
    int64_t score_all_frontier(SearchTree& tree, const CxMat& R, const CxVec& y_tilde,
                           const int* prog_belief, int belief_len, double nv) {
        int64_t total_flops = 0;
        for (int i = 0; i < tree.n_nodes; i++) {
            TreeNode& nd = tree.nodes[i];
            if (!nd.is_expanded && i != tree.root_id) {
                vm.setup_for_node(i, R, y_tilde, &tree, constellation, M, nv, Nt, Nr);
                // Push: cum_dist (bottom), M_down, M_up (top)
                vm.fs.push(nd.cum_dist);
                vm.fs.push(nd.m_down);
                vm.fs.push(nd.m_up);

                double result = vm.run(prog_belief, belief_len);
                total_flops += vm.flops_count;
                if (std::isfinite(result)) {
                    nd.score = result;
                    nd.queue_version++;
                } else {
                    nd.score = nd.cum_dist;
                }
            }
        }
        return total_flops;
    }

    // Check halt condition
    bool check_halt(SearchTree& tree, double old_root_m_up,
                    const int* prog_halt, int halt_len, double nv,
                    const CxMat& R, const CxVec& y_tilde) {
        if (tree.root_id < 0) return true;
        int rid = tree.root_id;
        vm.setup_for_node(rid, R, y_tilde, &tree, constellation, M, nv, Nt, Nr);
        vm.fs.push(old_root_m_up);
        vm.fs.push(tree.nodes[rid].m_up);

        vm.execute_block(prog_halt, 0, halt_len);
        if (vm.bs.has_pop()) return vm.bs.peek() != 0;
        return true;
    }

    // Full BP cycle
    int64_t full_bp_cycle(SearchTree& tree, const CxMat& R, const CxVec& y_tilde,
                      const int* prog_down, int down_len,
                      const int* prog_up, int up_len,
                      const int* prog_belief, int belief_len,
                      const int* prog_halt, int halt_len,
                      double nv) {
        int64_t total_flops = 0;
        for (int bp_iter = 0; bp_iter < max_bp_iters; bp_iter++) {
            double old_root_m_up = (tree.root_id >= 0) ? tree.nodes[tree.root_id].m_up : 0.0;
            total_flops += full_up_sweep(tree, R, y_tilde, prog_up, up_len, nv);
            total_flops += full_down_sweep(tree, R, y_tilde, prog_down, down_len, nv);
            total_flops += score_all_frontier(tree, R, y_tilde, prog_belief, belief_len, nv);
            if (bp_iter < max_bp_iters - 1 && tree.root_id >= 0) {
                bool halt = check_halt(tree, old_root_m_up, prog_halt, halt_len, nv, R, y_tilde);
                total_flops += vm.flops_count;
                if (halt) break;
            }
        }
        return total_flops;
    }

    // Complete path greedily
    void complete_path(int node_id, const CxMat& R, const CxVec& y_tilde,
                       SearchTree& tree, cx* x_out, int64_t& comp_flops) {
        // Collect decided symbols
        cx decided[MAX_DIM];
        int n_decided = tree.nodes[node_id].n_partial;
        for (int i = 0; i < n_decided; i++)
            decided[i] = tree.nodes[node_id].partial_symbols[i];

        int cur_layer = tree.nodes[node_id].layer - 1;
        comp_flops = 0;

        while (cur_layer >= 0) {
            cx best_sym = constellation[0];
            double best_ld = 1e30;
            for (int s = 0; s < M; s++) {
                cx sym = constellation[s];
                cx cand[MAX_DIM];
                int nc = n_decided + 1;
                for (int i = 0; i < n_decided; i++) cand[i] = decided[i];
                cand[n_decided] = sym;
                cx interf(0,0);
                for (int j = 0; j < nc; j++) {
                    int col = Nt - 1 - j;
                    interf += R.data[cur_layer][col] * cand[j];
                    comp_flops += 8;
                }
                cx residual = y_tilde.data[cur_layer] - interf;
                double ld = std::norm(residual);
                comp_flops += 5;
                if (ld < best_ld) { best_ld = ld; best_sym = sym; }
            }
            decided[n_decided++] = best_sym;
            cur_layer--;
        }

        // Reverse to get x[0]..x[Nt-1]
        for (int i = 0; i < Nt; i++)
            x_out[i] = (i < n_decided) ? decided[n_decided - 1 - i] : cx(0,0);
    }

    // Main detect function
    double detect(const cx* H_in, const cx* y_in, const cx* x_true,
                  const int* prog_down, int down_len,
                  const int* prog_up, int up_len,
                  const int* prog_belief, int belief_len,
                  const int* prog_halt, int halt_len,
                  double noise_var_in, double* flops_out, int* bp_calls_out) {
        total_bp_calls = 0;
        int64_t flops = 0;

        // Build H matrix
        CxMat H; H.rows = Nr; H.cols = Nt;
        for (int i = 0; i < Nr; i++)
            for (int j = 0; j < Nt; j++)
                H.data[i][j] = H_in[i * Nt + j];

        // QR decomposition
        CxMat Q, R;
        qr_mgs(H, Q, R);
        flops += 2 * Nr * Nt * Nt;

        // y_tilde = Q^H @ y
        CxVec y_tilde; y_tilde.len = Nt;
        for (int i = 0; i < Nt; i++) {
            cx s(0,0);
            for (int j = 0; j < Nr; j++) s += std::conj(Q.data[j][i]) * y_in[j];
            y_tilde.data[i] = s;
        }
        flops += 8 * Nt * Nr;

        // Init search tree
        int tree_cap = std::min(max_nodes + M * 64, (int)MAX_NODES);
        SearchTree tree(tree_cap);
        int root = tree.create_root(Nt);
        tree.nodes[root].m_up = 0;
        tree.nodes[root].m_down = 0;

        // Expand root
        int k0 = Nt - 1;
        for (int s = 0; s < M; s++) {
            cx sym = constellation[s];
            cx residual = y_tilde.data[k0] - R.data[k0][k0] * sym;
            double ld = std::norm(residual);
            flops += 11;
            cx part[1] = {sym};
            tree.add_child(root, k0, sym, ld, ld, part, 1);
        }

        // Full BP cycle
        flops += full_bp_cycle(tree, R, y_tilde,
                               prog_down, down_len, prog_up, up_len,
                               prog_belief, belief_len, prog_halt, halt_len,
                               noise_var_in);

        // Build initial PQ
        std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;
        int counter = 0;
        for (int i = 0; i < tree.n_nodes; i++) {
            if (!tree.nodes[i].is_expanded && i != root) {
                pq.push({tree.nodes[i].score, tree.nodes[i].queue_version, counter, i});
                counter++;
            }
        }

        int best_id = -1;
        double best_score = 1e30;

        while (!pq.empty() && tree.n_nodes < max_nodes + 1) {
            PQEntry top = pq.top(); pq.pop();
            if (top.queue_version != tree.nodes[top.node_id].queue_version) continue;

            if (top.score < best_score) { best_score = top.score; best_id = top.node_id; }

            TreeNode& node = tree.nodes[top.node_id];
            if (node.layer == 0) {
                // Complete solution found
                cx x_out[MAX_DIM];
                for (int i = 0; i < Nt; i++)
                    x_out[i] = (i < node.n_partial) ? node.partial_symbols[node.n_partial - 1 - i] : cx(0,0);
                if (flops_out) *flops_out = (double)flops;
                if (bp_calls_out) *bp_calls_out = total_bp_calls;
                int errors = 0;
                for (int i = 0; i < Nt; i++)
                    if (x_out[i] != x_true[i]) errors++;
                return (double)errors / Nt;
            }

            tree.mark_expanded(top.node_id);
            int next_layer = node.layer - 1;

            // Expand all M children
            for (int s = 0; s < M; s++) {
                cx sym = constellation[s];
                cx new_partial[MAX_DIM];
                int n_new = node.n_partial + 1;
                for (int i = 0; i < node.n_partial; i++) new_partial[i] = node.partial_symbols[i];
                new_partial[node.n_partial] = sym;

                cx interf(0,0);
                for (int j = 0; j < n_new; j++) {
                    int col = Nt - 1 - j;
                    interf += R.data[next_layer][col] * new_partial[j];
                    flops += 8;
                }
                cx residual = y_tilde.data[next_layer] - interf;
                flops += 2;
                double ld = std::norm(residual);
                flops += 3;
                double cd = node.cum_dist + ld;
                flops += 1;
                tree.add_child(top.node_id, next_layer, sym, ld, cd, new_partial, n_new);
            }

            // Full BP cycle on entire tree
            flops += full_bp_cycle(tree, R, y_tilde,
                                   prog_down, down_len, prog_up, up_len,
                                   prog_belief, belief_len, prog_halt, halt_len,
                                   noise_var_in);

            // Rebuild PQ from ALL frontier nodes
            while (!pq.empty()) pq.pop();
            counter = 0;
            for (int i = 0; i < tree.n_nodes; i++) {
                if (!tree.nodes[i].is_expanded && i != root) {
                    pq.push({tree.nodes[i].score, tree.nodes[i].queue_version, counter, i});
                    counter++;
                }
            }
        }

        // Fallback: greedy completion
        cx x_out[MAX_DIM];
        int64_t comp_flops = 0;
        if (best_id >= 0)
            complete_path(best_id, R, y_tilde, tree, x_out, comp_flops);
        else
            for (int i = 0; i < Nt; i++) x_out[i] = cx(0,0);
        flops += comp_flops;

        if (flops_out) *flops_out = (double)flops;
        if (bp_calls_out) *bp_calls_out = total_bp_calls;

        int errors = 0;
        for (int i = 0; i < Nt; i++)
            if (x_out[i] != x_true[i]) errors++;
        return (double)errors / Nt;
    }
};


// =========================================================================
// Context
// =========================================================================
struct BPEvalContext {
    int Nt, Nr, M;
    cx constellation[256];
    int max_nodes;
    int64_t flops_max;
    int step_max;
    int max_bp_iters;
};


// =========================================================================
// Standalone MMSE-LB computation (for baseline stack decoder)
// =========================================================================
static double compute_mmse_lb_standalone(const CxMat& R, const CxVec& y_tilde,
                                          const cx* x_partial, int n_partial,
                                          int layer, double noise_var, int Nt) {
    int k = layer;  // remaining undecoded layers (0 to k-1)
    if (k <= 0) return 0.0;

    // 1. Compute residue with interference cancellation
    cx residue[MAX_DIM];
    for (int i = 0; i < k; i++) {
        cx interf(0.0, 0.0);
        for (int j = 0; j < n_partial; j++)
            interf += R.data[i][k + j] * x_partial[n_partial - 1 - j];
        residue[i] = y_tilde.data[i] - interf;
    }

    // 2. Gram matrix: R_sub^H R_sub + σ²I
    cx Gram[MAX_DIM][MAX_DIM];
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            cx s(0.0, 0.0);
            for (int m = 0; m < k; m++)
                s += std::conj(R.data[m][i]) * R.data[m][j];
            Gram[i][j] = s;
        }
        Gram[i][i] += cx(noise_var, 0.0);
    }

    // 3. R_sub^H * residue
    cx Rh_r[MAX_DIM];
    for (int i = 0; i < k; i++) {
        cx s(0.0, 0.0);
        for (int m = 0; m < k; m++)
            s += std::conj(R.data[m][i]) * residue[m];
        Rh_r[i] = s;
    }

    // 4. Solve Gram * t = Rh_r
    cx t[MAX_DIM];
    if (!solve_linear(Gram, Rh_r, t, k)) return 0.0;

    // LB = r^H r - (Rh_r)^H t
    cx rHr(0.0, 0.0);
    for (int i = 0; i < k; i++) rHr += std::conj(residue[i]) * residue[i];
    cx corr(0.0, 0.0);
    for (int i = 0; i < k; i++) corr += std::conj(Rh_r[i]) * t[i];
    return std::max(0.0, std::real(rHr - corr));
}


// =========================================================================
// MMSE-LB Stack Decoder (pure baseline — no BP, no evolved programs)
// Uses score = cum_dist + MMSE_LB as priority queue ordering.
// This is the standard algorithm from literature.
// =========================================================================
static double mmselb_stack_detect(const cx* H_in, const cx* y_in, const cx* x_true,
                                   int Nt, int Nr, int M, const cx* constellation,
                                   int max_nodes, double noise_var) {
    // QR decomposition
    CxMat H(Nr, Nt);
    for (int i = 0; i < Nr; i++)
        for (int j = 0; j < Nt; j++)
            H.data[i][j] = H_in[i * Nt + j];

    CxMat Q, R;
    qr_mgs(H, Q, R);

    // y_tilde = Q^H y
    CxVec y_tilde; y_tilde.len = Nt;
    for (int i = 0; i < Nt; i++) {
        cx s(0, 0);
        for (int j = 0; j < Nr; j++) s += std::conj(Q.data[j][i]) * y_in[j];
        y_tilde.data[i] = s;
    }

    // Priority queue
    struct PQE {
        double score;
        int node_id;
        bool operator>(const PQE& o) const { return score > o.score; }
    };

    // Simple node storage
    struct SNode {
        cx partial[MAX_DIM];
        int n_partial;
        int layer;
        double cum_dist;
    };
    std::vector<SNode> nodes;
    nodes.reserve(max_nodes + M * 2);

    std::priority_queue<PQE, std::vector<PQE>, std::greater<PQE>> pq;

    // Expand root: layer Nt-1
    int k0 = Nt - 1;
    for (int s = 0; s < M; s++) {
        cx sym = constellation[s];
        cx residual = y_tilde.data[k0] - R.data[k0][k0] * sym;
        double ld = std::norm(residual);

        SNode nd;
        nd.n_partial = 1;
        nd.partial[0] = sym;
        nd.layer = k0;  // layer of the symbol
        nd.cum_dist = ld;

        double lb = compute_mmse_lb_standalone(R, y_tilde, nd.partial, nd.n_partial,
                                                k0, noise_var, Nt);
        int nid = (int)nodes.size();
        nodes.push_back(nd);
        pq.push({nd.cum_dist + lb, nid});
    }

    int expansions = 0;
    while (!pq.empty() && expansions < max_nodes) {
        PQE top = pq.top(); pq.pop();
        SNode parent = nodes[top.node_id];  // copy — push_back may reallocate
        int next_layer = parent.layer - 1;

        if (next_layer < 0) {
            // Complete solution
            cx x_out[MAX_DIM];
            for (int i = 0; i < Nt; i++)
                x_out[i] = (i < parent.n_partial) ? parent.partial[parent.n_partial - 1 - i] : cx(0, 0);
            int errors = 0;
            for (int i = 0; i < Nt; i++)
                if (x_out[i] != x_true[i]) errors++;
            return (double)errors / Nt;
        }

        // Expand
        for (int s = 0; s < M; s++) {
            cx sym = constellation[s];
            SNode child;
            child.n_partial = parent.n_partial + 1;
            std::memcpy(child.partial, parent.partial, parent.n_partial * sizeof(cx));
            child.partial[parent.n_partial] = sym;
            child.layer = next_layer;

            // Compute local distance at next_layer
            cx interf(0, 0);
            for (int j = 0; j < child.n_partial; j++) {
                int col = Nt - 1 - j;
                interf += R.data[next_layer][col] * child.partial[j];
            }
            cx residual = y_tilde.data[next_layer] - interf;
            double ld = std::norm(residual);
            child.cum_dist = parent.cum_dist + ld;

            // MMSE-LB for remaining layers
            double lb = 0.0;
            if (next_layer > 0) {
                lb = compute_mmse_lb_standalone(R, y_tilde, child.partial, child.n_partial,
                                                next_layer, noise_var, Nt);
            }

            int cid = (int)nodes.size();
            nodes.push_back(child);
            pq.push({child.cum_dist + lb, cid});
        }
        expansions++;
    }

    // Fallback: no complete solution found within max_nodes
    // Return best partial solution (greedy completion)
    if (!pq.empty()) {
        PQE best = pq.top();
        SNode bnode = nodes[best.node_id];  // copy for safety
        // Greedy completion
        cx decided[MAX_DIM];
        int n_decided = bnode.n_partial;
        for (int i = 0; i < n_decided; i++) decided[i] = bnode.partial[i];
        int cur_layer = bnode.layer - 1;
        while (cur_layer >= 0) {
            cx best_sym = constellation[0];
            double best_ld = 1e30;
            for (int si = 0; si < M; si++) {
                cx sym = constellation[si];
                cx interf(0, 0);
                int nc = n_decided + 1;
                for (int j = 0; j < n_decided; j++) {
                    int col = Nt - 1 - j;
                    interf += R.data[cur_layer][col] * decided[j];
                }
                interf += R.data[cur_layer][Nt - 1 - n_decided] * sym;
                cx residual = y_tilde.data[cur_layer] - interf;
                double ld = std::norm(residual);
                if (ld < best_ld) { best_ld = ld; best_sym = sym; }
            }
            decided[n_decided++] = best_sym;
            cur_layer--;
        }
        cx x_out[MAX_DIM];
        for (int i = 0; i < Nt; i++)
            x_out[i] = (i < n_decided) ? decided[n_decided - 1 - i] : cx(0, 0);
        int errors = 0;
        for (int i = 0; i < Nt; i++)
            if (x_out[i] != x_true[i]) errors++;
        return (double)errors / Nt;
    }

    return 1.0;  // complete failure
}


// =========================================================================
// C API Implementation
// =========================================================================
static cx* deinterleave(const double* interleaved, int n) {
    cx* out = new cx[n];
    for (int i = 0; i < n; i++)
        out[i] = cx(interleaved[2*i], interleaved[2*i+1]);
    return out;
}

extern "C" {

BP_EXPORT void* bp_eval_create(int Nt, int Nr, int M,
                                const double* constellation,
                                int max_nodes, int flops_max, int step_max,
                                int max_bp_iters) {
    auto* ctx = new BPEvalContext;
    ctx->Nt = Nt;
    ctx->Nr = Nr;
    ctx->M = M;
    ctx->max_nodes = max_nodes;
    ctx->flops_max = flops_max;
    ctx->step_max = step_max;
    ctx->max_bp_iters = max_bp_iters;
    for (int i = 0; i < M; i++)
        ctx->constellation[i] = cx(constellation[2*i], constellation[2*i+1]);
    return ctx;
}

BP_EXPORT void bp_eval_destroy(void* ctx) {
    delete (BPEvalContext*)ctx;
}

BP_EXPORT double bp_eval_one(void* ctx_,
                              const int* prog_down, int down_len,
                              const int* prog_up, int up_len,
                              const int* prog_belief, int belief_len,
                              const int* prog_halt, int halt_len,
                              const double* H, const double* y,
                              const double* x_true,
                              double noise_var,
                              double* flops_out,
                              int* bp_calls_out) {
    auto* ctx = (BPEvalContext*)ctx_;
    int Nt = ctx->Nt, Nr = ctx->Nr, M = ctx->M;

    // De-interleave
    cx* H_cx = deinterleave(H, Nr * Nt);
    cx* y_cx = deinterleave(y, Nr);
    cx* x_cx = deinterleave(x_true, Nt);

    BPStackDecoder dec;
    dec.Nt = Nt;
    dec.Nr = Nr;
    dec.M = M;
    dec.constellation = ctx->constellation;
    dec.max_nodes = ctx->max_nodes;
    dec.max_bp_iters = ctx->max_bp_iters;
    dec.vm.flops_max = ctx->flops_max;
    dec.vm.step_max = ctx->step_max;

    double ber = dec.detect(H_cx, y_cx, x_cx,
                            prog_down, down_len, prog_up, up_len,
                            prog_belief, belief_len, prog_halt, halt_len,
                            noise_var, flops_out, bp_calls_out);

    delete[] H_cx;
    delete[] y_cx;
    delete[] x_cx;
    return ber;
}

BP_EXPORT double bp_eval_dataset(void* ctx_,
                                  const int* prog_down, int down_len,
                                  const int* prog_up, int up_len,
                                  const int* prog_belief, int belief_len,
                                  const int* prog_halt, int halt_len,
                                  int n_samples,
                                  const double* H_all,
                                  const double* y_all,
                                  const double* x_true_all,
                                  const double* noise_vars,
                                  double* avg_flops_out,
                                  int* total_faults_out,
                                  double* avg_bp_calls_out) {
    auto* ctx = (BPEvalContext*)ctx_;
    int Nt = ctx->Nt, Nr = ctx->Nr, M = ctx->M;
    int H_stride = 2 * Nr * Nt;
    int y_stride = 2 * Nr;
    int x_stride = 2 * Nt;

    double total_ber = 0;
    double total_flops = 0;
    double total_bp = 0;
    int total_faults = 0;

    std::atomic<int> ds_done(0);
    int ds_next_pct = 5;  // next percentage threshold to print

    #pragma omp parallel for reduction(+:total_ber,total_flops,total_bp,total_faults) schedule(dynamic)
    for (int s = 0; s < n_samples; s++) {
        cx* H_cx = deinterleave(H_all + s * H_stride, Nr * Nt);
        cx* y_cx = deinterleave(y_all + s * y_stride, Nr);
        cx* x_cx = deinterleave(x_true_all + s * x_stride, Nt);

        BPStackDecoder dec;
        dec.Nt = Nt; dec.Nr = Nr; dec.M = M;
        dec.constellation = ctx->constellation;
        dec.max_nodes = ctx->max_nodes;
        dec.max_bp_iters = ctx->max_bp_iters;
        dec.vm.flops_max = ctx->flops_max;
        dec.vm.step_max = ctx->step_max;

        double flops_out = 0;
        int bp_calls_out = 0;
        double ber = 1.0;
        int fault = 0;
        try {
            ber = dec.detect(H_cx, y_cx, x_cx,
                             prog_down, down_len, prog_up, up_len,
                             prog_belief, belief_len, prog_halt, halt_len,
                             noise_vars[s], &flops_out, &bp_calls_out);
        } catch (...) {
            ber = 1.0;
            flops_out = ctx->flops_max * 10.0;
            fault = 1;
        }

        total_ber += ber;
        total_flops += flops_out;
        total_bp += bp_calls_out;
        total_faults += fault;

        delete[] H_cx;
        delete[] y_cx;
        delete[] x_cx;

        int completed = ds_done.fetch_add(1) + 1;
        int pct = (int)((100LL * completed) / n_samples);
        #pragma omp critical
        {
            if (pct >= ds_next_pct) {
                printf("\r  [dataset] %d/%d samples (%d%%)", completed, n_samples, pct);
                fflush(stdout);
                ds_next_pct = pct + 5;
            }
        }
    }
    if (n_samples > 0) {
        printf("\r  [dataset] %d/%d samples (100%%)\n", n_samples, n_samples);
        fflush(stdout);
    }

    if (avg_flops_out) *avg_flops_out = total_flops / n_samples;
    if (total_faults_out) *total_faults_out = total_faults;
    if (avg_bp_calls_out) *avg_bp_calls_out = total_bp / n_samples;
    return total_ber / n_samples;
}

BP_EXPORT void bp_eval_batch(void* ctx_,
                              int n_genomes,
                              const int* prog_all,
                              const int* prog_offsets,
                              const int* prog_lengths,
                              int n_samples,
                              const double* H_all,
                              const double* y_all,
                              const double* x_true_all,
                              const double* noise_vars,
                              double* ber_out,
                              double* flops_out,
                              int* faults_out,
                              double* bp_calls_out) {
    auto* ctx = (BPEvalContext*)ctx_;
    int Nt = ctx->Nt, Nr = ctx->Nr, M = ctx->M;
    int H_stride = 2 * Nr * Nt;
    int y_stride = 2 * Nr;
    int x_stride = 2 * Nt;

    // Pre-deinterleave all samples
    std::vector<cx*> H_cx(n_samples), y_cx(n_samples), x_cx(n_samples);
    for (int s = 0; s < n_samples; s++) {
        H_cx[s] = deinterleave(H_all + s * H_stride, Nr * Nt);
        y_cx[s] = deinterleave(y_all + s * y_stride, Nr);
        x_cx[s] = deinterleave(x_true_all + s * x_stride, Nt);
    }

    // Zero output arrays
    for (int g = 0; g < n_genomes; g++) {
        ber_out[g] = 0; flops_out[g] = 0; faults_out[g] = 0; bp_calls_out[g] = 0;
    }

    // Flatten (genome, sample) pairs for better OpenMP load balancing.
    // This eliminates the straggler problem where a few heavy genomes
    // monopolize cores while lighter genomes finish early.
    int total_work = n_genomes * n_samples;
    std::atomic<int> batch_done(0);
    int batch_next_pct = 5;  // next percentage threshold to print

    printf("  [batch] evaluating %d genomes x %d samples = %d tasks\n",
           n_genomes, n_samples, total_work);
    fflush(stdout);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int idx = 0; idx < total_work; idx++) {
        int g = idx / n_samples;
        int s = idx % n_samples;

        const int* pd = prog_all + prog_offsets[g * 4 + 0];
        int dl = prog_lengths[g * 4 + 0];
        const int* pu = prog_all + prog_offsets[g * 4 + 1];
        int ul = prog_lengths[g * 4 + 1];
        const int* pb = prog_all + prog_offsets[g * 4 + 2];
        int bl = prog_lengths[g * 4 + 2];
        const int* ph = prog_all + prog_offsets[g * 4 + 3];
        int hl = prog_lengths[g * 4 + 3];

        BPStackDecoder dec;
        dec.Nt = Nt; dec.Nr = Nr; dec.M = M;
        dec.constellation = ctx->constellation;
        dec.max_nodes = ctx->max_nodes;
        dec.max_bp_iters = ctx->max_bp_iters;
        dec.vm.flops_max = ctx->flops_max;
        dec.vm.step_max = ctx->step_max;

        double fo = 0; int bpo = 0;
        double ber = 1.0;
        try {
            ber = dec.detect(H_cx[s], y_cx[s], x_cx[s],
                             pd, dl, pu, ul, pb, bl, ph, hl,
                             noise_vars[s], &fo, &bpo);
        } catch (...) {
            ber = 1.0;
            fo = ctx->flops_max * 10.0;
            #pragma omp atomic
            faults_out[g]++;
        }
        #pragma omp atomic
        ber_out[g] += ber;
        #pragma omp atomic
        flops_out[g] += fo;
        #pragma omp atomic
        bp_calls_out[g] += (double)bpo;

        int completed = batch_done.fetch_add(1) + 1;
        int pct = (int)((100LL * completed) / total_work);
        #pragma omp critical
        {
            if (pct >= batch_next_pct) {
                printf("\r  [batch] %d/%d tasks (%d%%)", completed, total_work, pct);
                fflush(stdout);
                batch_next_pct = pct + 5;
            }
        }
    }
    printf("\r  [batch] %d/%d tasks (100%%)\n", total_work, total_work);
    fflush(stdout);

    // Normalize by n_samples
    for (int g = 0; g < n_genomes; g++) {
        ber_out[g] /= n_samples;
        flops_out[g] /= n_samples;
        bp_calls_out[g] /= n_samples;
    }

    // Cleanup
    for (int s = 0; s < n_samples; s++) {
        delete[] H_cx[s];
        delete[] y_cx[s];
        delete[] x_cx[s];
    }
}

// =========================================================================
// Baseline detectors: LMMSE, K-Best
// =========================================================================

// LMMSE detector
static double lmmse_detect(const cx* H, const cx* y, const cx* x_true,
                            int Nt, int Nr, int M, const cx* constellation,
                            double noise_var) {
    // Build H matrix (Nr x Nt)
    CxMat Hm(Nr, Nt);
    for (int r = 0; r < Nr; r++)
        for (int c = 0; c < Nt; c++)
            Hm.data[r][c] = H[r * Nt + c];

    // H^H H  (Nt x Nt)
    CxMat HhH(Nt, Nt);
    for (int i = 0; i < Nt; i++)
        for (int j = 0; j < Nt; j++) {
            cx s = 0;
            for (int k = 0; k < Nr; k++)
                s += std::conj(Hm.data[k][i]) * Hm.data[k][j];
            HhH.data[i][j] = s;
        }

    // Regularize: HhH + noise_var * I
    for (int i = 0; i < Nt; i++)
        HhH.data[i][i] += noise_var;

    // H^H y  (Nt x 1)
    cx Hy[MAX_DIM];
    for (int i = 0; i < Nt; i++) {
        cx s = 0;
        for (int k = 0; k < Nr; k++)
            s += std::conj(Hm.data[k][i]) * y[k];
        Hy[i] = s;
    }

    // Solve via Gaussian elimination with partial pivoting
    // Augmented matrix [HhH | Hy]
    cx A[MAX_DIM][MAX_DIM + 1];
    for (int i = 0; i < Nt; i++) {
        for (int j = 0; j < Nt; j++)
            A[i][j] = HhH.data[i][j];
        A[i][Nt] = Hy[i];
    }

    for (int col = 0; col < Nt; col++) {
        // Partial pivoting
        int pivot = col;
        double maxval = std::abs(A[col][col]);
        for (int row = col + 1; row < Nt; row++) {
            double v = std::abs(A[row][col]);
            if (v > maxval) { maxval = v; pivot = row; }
        }
        if (pivot != col)
            for (int j = col; j <= Nt; j++)
                std::swap(A[col][j], A[pivot][j]);

        // Eliminate
        cx diag = A[col][col];
        if (std::abs(diag) < 1e-30) continue;
        for (int row = col + 1; row < Nt; row++) {
            cx factor = A[row][col] / diag;
            for (int j = col; j <= Nt; j++)
                A[row][j] -= factor * A[col][j];
        }
    }

    // Back substitution
    cx x_hat[MAX_DIM];
    for (int i = Nt - 1; i >= 0; i--) {
        cx s = A[i][Nt];
        for (int j = i + 1; j < Nt; j++)
            s -= A[i][j] * x_hat[j];
        if (std::abs(A[i][i]) > 1e-30)
            x_hat[i] = s / A[i][i];
        else
            x_hat[i] = 0;
    }

    // Slice to nearest constellation point
    cx detected[MAX_DIM];
    for (int i = 0; i < Nt; i++) {
        double best_d = 1e30;
        int best_idx = 0;
        for (int m = 0; m < M; m++) {
            double d = std::norm(constellation[m] - x_hat[i]);
            if (d < best_d) { best_d = d; best_idx = m; }
        }
        detected[i] = constellation[best_idx];
    }

    // Count errors
    int errors = 0;
    for (int i = 0; i < Nt; i++)
        if (std::abs(detected[i] - x_true[i]) > 1e-6)
            errors++;
    return (double)errors / Nt;
}

// K-Best detector
struct KBestCandidate {
    cx symbols[MAX_DIM]; // partial symbols (in decode order: Nt-1, Nt-2, ...)
    int n_sym;
    double cum_dist;
    bool operator>(const KBestCandidate& o) const { return cum_dist > o.cum_dist; }
};

static double kbest_detect(const cx* H, const cx* y, const cx* x_true,
                            int Nt, int Nr, int M, const cx* constellation,
                            int K) {
    // Build H matrix
    CxMat Hm(Nr, Nt);
    for (int r = 0; r < Nr; r++)
        for (int c = 0; c < Nt; c++)
            Hm.data[r][c] = H[r * Nt + c];

    // QR decomposition
    CxMat Q(Nr, Nt), R(Nt, Nt);
    qr_mgs(Hm, Q, R);

    // y_tilde = Q^H y
    cx y_tilde[MAX_DIM];
    for (int i = 0; i < Nt; i++) {
        cx s = 0;
        for (int k = 0; k < Nr; k++)
            s += std::conj(Q.data[k][i]) * y[k];
        y_tilde[i] = s;
    }

    // Layer Nt-1 (bottom): create initial M candidates
    int max_cands = K * M + M;  // max candidates at any layer
    std::vector<KBestCandidate> candidates(M);
    for (int m = 0; m < M; m++) {
        candidates[m].n_sym = 1;
        candidates[m].symbols[0] = constellation[m];
        cx residual = y_tilde[Nt - 1] - R.data[Nt - 1][Nt - 1] * constellation[m];
        candidates[m].cum_dist = std::norm(residual);
    }
    // Sort and keep K best
    std::sort(candidates.begin(), candidates.end(),
              [](const KBestCandidate& a, const KBestCandidate& b) {
                  return a.cum_dist < b.cum_dist; });
    if ((int)candidates.size() > K) candidates.resize(K);

    // Extend layer by layer (Nt-2 down to 0)
    for (int layer = Nt - 2; layer >= 0; layer--) {
        std::vector<KBestCandidate> new_cands;
        new_cands.reserve(candidates.size() * M);
        for (auto& cand : candidates) {
            for (int m = 0; m < M; m++) {
                KBestCandidate nc;
                nc.n_sym = cand.n_sym + 1;
                std::memcpy(nc.symbols, cand.symbols, cand.n_sym * sizeof(cx));
                nc.symbols[cand.n_sym] = constellation[m];

                // Compute interference at this layer
                cx interference = 0;
                for (int j = 0; j < nc.n_sym; j++) {
                    int col = Nt - 1 - j;
                    interference += R.data[layer][col] * nc.symbols[j];
                }
                cx residual = y_tilde[layer] - interference;
                nc.cum_dist = cand.cum_dist + std::norm(residual);
                new_cands.push_back(nc);
            }
        }
        std::sort(new_cands.begin(), new_cands.end(),
                  [](const KBestCandidate& a, const KBestCandidate& b) {
                      return a.cum_dist < b.cum_dist; });
        if ((int)new_cands.size() > K) new_cands.resize(K);
        candidates = std::move(new_cands);
    }

    // Best candidate: reverse to get natural order
    cx x_hat[MAX_DIM];
    for (int i = 0; i < Nt; i++)
        x_hat[i] = candidates[0].symbols[Nt - 1 - i];

    // Count errors
    int errors = 0;
    for (int i = 0; i < Nt; i++)
        if (std::abs(x_hat[i] - x_true[i]) > 1e-6)
            errors++;
    return (double)errors / Nt;
}


BP_EXPORT void bp_eval_baselines(void* ctx_,
                                  int n_samples,
                                  const double* H_all,
                                  const double* y_all,
                                  const double* x_true_all,
                                  const double* noise_vars,
                                  double* ber_lmmse_out,
                                  double* ber_kb16_out,
                                  double* ber_kb32_out) {
    auto* ctx = (BPEvalContext*)ctx_;
    int Nt = ctx->Nt, Nr = ctx->Nr, M = ctx->M;
    int H_stride = 2 * Nr * Nt;
    int y_stride = 2 * Nr;
    int x_stride = 2 * Nt;

    double sum_lmmse = 0, sum_kb16 = 0, sum_kb32 = 0;

    #pragma omp parallel for reduction(+:sum_lmmse,sum_kb16,sum_kb32) schedule(dynamic)
    for (int s = 0; s < n_samples; s++) {
        cx* H_cx = deinterleave(H_all + s * H_stride, Nr * Nt);
        cx* y_cx = deinterleave(y_all + s * y_stride, Nr);
        cx* x_cx = deinterleave(x_true_all + s * x_stride, Nt);

        if (ber_lmmse_out)
            sum_lmmse += lmmse_detect(H_cx, y_cx, x_cx, Nt, Nr, M,
                                       ctx->constellation, noise_vars[s]);
        if (ber_kb16_out)
            sum_kb16 += kbest_detect(H_cx, y_cx, x_cx, Nt, Nr, M,
                                      ctx->constellation, 16);
        if (ber_kb32_out)
            sum_kb32 += kbest_detect(H_cx, y_cx, x_cx, Nt, Nr, M,
                                      ctx->constellation, 32);

        delete[] H_cx;
        delete[] y_cx;
        delete[] x_cx;
    }

    if (ber_lmmse_out) *ber_lmmse_out = sum_lmmse / n_samples;
    if (ber_kb16_out) *ber_kb16_out = sum_kb16 / n_samples;
    if (ber_kb32_out) *ber_kb32_out = sum_kb32 / n_samples;
}


// =========================================================================
// MMSE-LB Stack Decoder baseline evaluation
// =========================================================================
BP_EXPORT double bp_eval_mmselb_stack(void* ctx_,
                                       int n_samples,
                                       const double* H_all,
                                       const double* y_all,
                                       const double* x_true_all,
                                       const double* noise_vars,
                                       int override_max_nodes) {
    auto* ctx = (BPEvalContext*)ctx_;
    int Nt = ctx->Nt, Nr = ctx->Nr, M = ctx->M;
    int H_stride = 2 * Nr * Nt;
    int y_stride = 2 * Nr;
    int x_stride = 2 * Nt;
    int mn = (override_max_nodes > 0) ? override_max_nodes : ctx->max_nodes;

    double total_ber = 0;

    #pragma omp parallel for reduction(+:total_ber) schedule(dynamic)
    for (int s = 0; s < n_samples; s++) {
        cx* H_cx = deinterleave(H_all + s * H_stride, Nr * Nt);
        cx* y_cx = deinterleave(y_all + s * y_stride, Nr);
        cx* x_cx = deinterleave(x_true_all + s * x_stride, Nt);

        double ber = mmselb_stack_detect(H_cx, y_cx, x_cx, Nt, Nr, M,
                                          ctx->constellation, mn, noise_vars[s]);

        total_ber += ber;
        delete[] H_cx; delete[] y_cx; delete[] x_cx;
    }

    return total_ber / n_samples;
}


// =========================================================================
// Multi-node-limit evaluation for evolved BP programs
// =========================================================================
BP_EXPORT void bp_eval_multi_nodes(void* ctx_,
                                    const int* prog_down, int down_len,
                                    const int* prog_up, int up_len,
                                    const int* prog_belief, int belief_len,
                                    const int* prog_halt, int halt_len,
                                    int n_samples,
                                    const double* H_all,
                                    const double* y_all,
                                    const double* x_true_all,
                                    const double* noise_vars,
                                    const int* node_limits,
                                    int n_limits,
                                    double* ber_out) {
    auto* ctx = (BPEvalContext*)ctx_;
    int Nt = ctx->Nt, Nr = ctx->Nr, M = ctx->M;
    int H_stride = 2 * Nr * Nt;
    int y_stride = 2 * Nr;
    int x_stride = 2 * Nt;

    for (int lim_idx = 0; lim_idx < n_limits; lim_idx++) {
        int mn = node_limits[lim_idx];
        double total_ber = 0;

        #pragma omp parallel for reduction(+:total_ber) schedule(dynamic)
        for (int s = 0; s < n_samples; s++) {
            cx* H_cx = deinterleave(H_all + s * H_stride, Nr * Nt);
            cx* y_cx = deinterleave(y_all + s * y_stride, Nr);
            cx* x_cx = deinterleave(x_true_all + s * x_stride, Nt);

            BPStackDecoder dec;
            dec.Nt = Nt; dec.Nr = Nr; dec.M = M;
            dec.constellation = ctx->constellation;
            dec.max_nodes = mn;
            dec.max_bp_iters = ctx->max_bp_iters;
            dec.vm.flops_max = ctx->flops_max;
            dec.vm.step_max = ctx->step_max;

            double flops_out = 0;
            int bp_calls_out = 0;
            double ber = 1.0;
            try {
                ber = dec.detect(H_cx, y_cx, x_cx,
                                 prog_down, down_len, prog_up, up_len,
                                 prog_belief, belief_len, prog_halt, halt_len,
                                 noise_vars[s], &flops_out, &bp_calls_out);
            } catch (...) {
                ber = 1.0;
            }

            total_ber += ber;
            delete[] H_cx; delete[] y_cx; delete[] x_cx;
        }

        ber_out[lim_idx] = total_ber / n_samples;
    }
}


// =========================================================================
// Multi-node-limit evaluation for MMSE-LB stack decoder baseline
// =========================================================================
BP_EXPORT void bp_eval_mmselb_multi_nodes(void* ctx_,
                                           int n_samples,
                                           const double* H_all,
                                           const double* y_all,
                                           const double* x_true_all,
                                           const double* noise_vars,
                                           const int* node_limits,
                                           int n_limits,
                                           double* ber_out) {
    auto* ctx = (BPEvalContext*)ctx_;
    int Nt = ctx->Nt, Nr = ctx->Nr, M = ctx->M;
    int H_stride = 2 * Nr * Nt;
    int y_stride = 2 * Nr;
    int x_stride = 2 * Nt;

    for (int lim_idx = 0; lim_idx < n_limits; lim_idx++) {
        int mn = node_limits[lim_idx];
        double total_ber = 0;

        #pragma omp parallel for reduction(+:total_ber) schedule(dynamic)
        for (int s = 0; s < n_samples; s++) {
            cx* H_cx = deinterleave(H_all + s * H_stride, Nr * Nt);
            cx* y_cx = deinterleave(y_all + s * y_stride, Nr);
            cx* x_cx = deinterleave(x_true_all + s * x_stride, Nt);

            double ber = mmselb_stack_detect(H_cx, y_cx, x_cx, Nt, Nr, M,
                                              ctx->constellation, mn, noise_vars[s]);

            total_ber += ber;
            delete[] H_cx; delete[] y_cx; delete[] x_cx;
        }

        ber_out[lim_idx] = total_ber / n_samples;
    }
}

} // extern "C"
