/*
 * evaluator.cpp — C++ Push VM + Stack Decoder for MIMO Algorithm Discovery
 *
 * Self-contained implementation: no external dependencies (no Eigen, no BLAS).
 * Small matrix sizes (up to 32×32) mean simple loop kernels are sufficient.
 *
 * Compile (MSVC):
 *   cl.exe /EHsc /O2 /openmp /std:c++17 evaluator.cpp /LD /Fe:evaluator.dll
 *
 * Compile (GCC/Linux):
 *   g++ -O2 -fopenmp -std=c++17 -shared -fPIC evaluator.cpp -o evaluator.so
 */

#include "evaluator.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <complex>
#include <vector>
#include <queue>
#include <functional>
#include <limits>
#include <cstdio>

using cx = std::complex<double>;

// =========================================================================
// Configuration
// =========================================================================
static constexpr int MAX_DIM   = 32;   // max Nr or Nt
static constexpr int MAX_NODES = 4096; // max search tree nodes (heap-allocated)
static constexpr int STACK_DEPTH = 64; // per-type stack depth
static constexpr int N_MEM      = 16;  // per-node memory slots
static constexpr int MAX_PROG   = 512; // max program length (flat)
static constexpr int MAX_CTRL_ITER = 20; // while/ForEach iteration cap
static constexpr int MAX_DOTIMES   = 15;
static constexpr double CLAMP_EXP  = 20.0;

// =========================================================================
// Small complex linear algebra (no Eigen needed for 8×16)
// =========================================================================
struct CxVec {
    cx data[MAX_DIM];
    int len = 0;
    CxVec() : len(0) { memset(data, 0, sizeof(data)); }
    CxVec(int n) : len(n) { memset(data, 0, sizeof(data)); }
};

struct CxMat {
    cx data[MAX_DIM][MAX_DIM]; // row-major: data[row][col]
    int rows = 0, cols = 0;
    CxMat() : rows(0), cols(0) { memset(data, 0, sizeof(data)); }
    CxMat(int r, int c) : rows(r), cols(c) { memset(data, 0, sizeof(data)); }
};

// y = A * x  (A is rows×cols, x is cols, y is rows)
static void mat_vec_mul(const CxMat& A, const CxVec& x, CxVec& y) {
    y.len = A.rows;
    for (int i = 0; i < A.rows; i++) {
        cx s(0,0);
        for (int j = 0; j < A.cols; j++) s += A.data[i][j] * x.data[j];
        y.data[i] = s;
    }
}

// QR decomposition via modified Gram-Schmidt (thin QR: Q is Nr×Nt, R is Nt×Nt)
static void qr_mgs(const CxMat& H, CxMat& Q, CxMat& R) {
    int Nr = H.rows, Nt = H.cols;
    Q = CxMat(Nr, Nt);
    R = CxMat(Nt, Nt);
    // Copy columns of H into Q
    for (int j = 0; j < Nt; j++)
        for (int i = 0; i < Nr; i++)
            Q.data[i][j] = H.data[i][j];
    for (int j = 0; j < Nt; j++) {
        // Orthogonalize against previous columns
        for (int k = 0; k < j; k++) {
            cx dot(0,0);
            for (int i = 0; i < Nr; i++) dot += std::conj(Q.data[i][k]) * Q.data[i][j];
            R.data[k][j] = dot;
            for (int i = 0; i < Nr; i++) Q.data[i][j] -= dot * Q.data[i][k];
        }
        // Normalize
        double nrm = 0;
        for (int i = 0; i < Nr; i++) nrm += std::norm(Q.data[i][j]);
        nrm = std::sqrt(nrm);
        R.data[j][j] = cx(nrm, 0);
        if (nrm > 1e-15)
            for (int i = 0; i < Nr; i++) Q.data[i][j] /= nrm;
    }
}

// Solve Ax = b where A is n×n using Gaussian elimination with partial pivoting
static bool solve_linear(cx A[MAX_DIM][MAX_DIM], cx b[MAX_DIM], cx x[MAX_DIM], int n) {
    // Augmented matrix [A|b] in-place
    cx Ab[MAX_DIM][MAX_DIM+1];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) Ab[i][j] = A[i][j];
        Ab[i][n] = b[i];
    }
    for (int col = 0; col < n; col++) {
        // Partial pivoting
        int best = col;
        double best_abs = std::abs(Ab[col][col]);
        for (int row = col+1; row < n; row++) {
            double a = std::abs(Ab[row][col]);
            if (a > best_abs) { best_abs = a; best = row; }
        }
        if (best_abs < 1e-14) return false;
        if (best != col)
            for (int j = col; j <= n; j++) std::swap(Ab[col][j], Ab[best][j]);
        // Eliminate
        for (int row = col+1; row < n; row++) {
            cx factor = Ab[row][col] / Ab[col][col];
            for (int j = col; j <= n; j++) Ab[row][j] -= factor * Ab[col][j];
        }
    }
    // Back substitution
    for (int i = n-1; i >= 0; i--) {
        x[i] = Ab[i][n];
        for (int j = i+1; j < n; j++) x[i] -= Ab[i][j] * x[j];
        if (std::abs(Ab[i][i]) < 1e-14) return false;
        x[i] /= Ab[i][i];
    }
    return true;
}


// =========================================================================
// Tree Node
// =========================================================================
struct TreeNode {
    int id = -1;
    int layer = 0;
    cx  symbol = 0;
    double local_dist = 0;
    double cum_dist = 0;
    double score = 0;
    int queue_version = 0;
    bool is_expanded = false;
    double mem[N_MEM] = {};
    TreeNode* parent = nullptr;
    int children[64]; // indices into SearchTree::nodes
    int n_children = 0;
    // Partial symbols (deepest-first order, like Python)
    cx partial_symbols[MAX_DIM];
    int n_partial = 0;
};

// =========================================================================
// Search Tree
// =========================================================================
struct SearchTree {
    TreeNode* nodes;
    int capacity;
    int n_nodes = 0;
    int root_id = -1;
    int* dirty;
    int n_dirty = 0;

    SearchTree(int cap = MAX_NODES) : capacity(cap) {
        nodes = new TreeNode[cap];
        dirty = new int[cap];
    }
    ~SearchTree() { delete[] nodes; delete[] dirty; }
    SearchTree(const SearchTree&) = delete;
    SearchTree& operator=(const SearchTree&) = delete;

    void reset() {
        n_nodes = 0; root_id = -1; n_dirty = 0;
    }

    int create_root(int layer) {
        int id = n_nodes++;
        TreeNode& nd = nodes[id];
        nd = TreeNode();
        nd.id = id;
        nd.layer = layer;
        nd.score = 0;
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
        nd.score = cd; // default score = cum_dist
        nd.parent = &nodes[parent_id];
        nd.n_partial = n_partial;
        for (int i = 0; i < n_partial; i++) nd.partial_symbols[i] = partial[i];
        // Add to parent's children
        TreeNode& p = nodes[parent_id];
        if (p.n_children < 64) p.children[p.n_children++] = id;
        return id;
    }

    void mark_expanded(int id) { nodes[id].is_expanded = true; }

    void add_dirty(int id) {
        if (n_dirty < capacity) dirty[n_dirty++] = id;
    }

    void clear_dirty() { n_dirty = 0; }

    // Siblings: other children of the same parent (excluding self)
    int get_siblings(int id, int* out, int max_out) const {
        const TreeNode& nd = nodes[id];
        if (!nd.parent) return 0;
        int count = 0;
        for (int i = 0; i < nd.parent->n_children && count < max_out; i++) {
            int cid = nd.parent->children[i];
            if (cid != id) out[count++] = cid;
        }
        return count;
    }

    // Ancestors: parent chain up to (but not including) root
    int get_ancestors(int id, int* out, int max_out) const {
        const TreeNode& nd = nodes[id];
        TreeNode* p = nd.parent;
        int count = 0;
        while (p && p->id != root_id && count < max_out) {
            out[count++] = p->id;
            p = p->parent;
        }
        return count;
    }

    int frontier_count() const {
        int c = 0;
        for (int i = 0; i < n_nodes; i++)
            if (!nodes[i].is_expanded && i != root_id) c++;
        return c;
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
    bool empty() const { return top == 0; }
    int depth() const { return top; }

    void push(T v) { if (top < Cap) data[top++] = v; }
    T pop() {
        if (top > 0) return data[--top];
        return T{};
    }
    T peek() const {
        if (top > 0) return data[top-1];
        return T{};
    }
    bool has_pop() const { return top > 0; }
    bool has_n(int n) const { return top >= n; }

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
// Push VM
// =========================================================================
struct PushVM {
    // Stacks
    Stack<double>    fs;   // float
    Stack<int>       is;   // int
    Stack<int>       bs;   // bool (0/1)
    Stack<CxVec, 16> vs;   // vector
    Stack<CxMat, 8>  ms;   // matrix
    Stack<int>       ns;   // node (index into SearchTree)
    Stack<int, 4>    gs;   // graph (always 0 — single graph)

    // Environment
    SearchTree* tree = nullptr;
    const cx* constellation = nullptr;
    int M = 0;         // constellation size
    int Nt = 0, Nr = 0;
    double noise_var = 1.0;

    // Limits
    int flops_max = 2000000;
    int step_max  = 1500;
    int flops_count = 0;
    int step_count  = 0;
    bool halted = false;

    void reset() {
        fs.clear(); is.clear(); bs.clear();
        vs.clear(); ms.clear(); ns.clear(); gs.clear();
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

    // Inject environment (like Python inject_environment)
    void inject_environment(const CxMat& R, const CxVec& y_tilde,
                            const cx* x_partial, int n_partial,
                            SearchTree* t, int candidate_id,
                            int depth_k,
                            const cx* cons, int M_,
                            double nv, int Nt_, int Nr_) {
        reset();
        Nt = Nt_; Nr = Nr_;
        noise_var = nv;
        constellation = cons;
        M = M_;
        tree = t;

        CxMat Rcopy = R;
        ms.push(Rcopy);
        CxVec yt = y_tilde;
        vs.push(yt);
        CxVec xp;
        xp.len = n_partial;
        for (int i = 0; i < n_partial; i++) xp.data[i] = x_partial[i];
        vs.push(xp);
        gs.push(0); // graph index (always 0)
        ns.push(candidate_id);
        is.push(depth_k);
    }

    // ---- Execute a flat-encoded program ----
    // Returns the program counter after execution
    double run(const int* prog, int len) {
        execute_block(prog, 0, len);
        if (fs.has_pop()) {
            double r = fs.peek();
            if (std::isfinite(r)) return r;
        }
        return std::numeric_limits<double>::infinity();
    }

    // Execute instructions from prog[start..end)
    void execute_block(const int* prog, int start, int end) {
        int pc = start;
        while (pc < end && !halted) {
            pc = execute_one(prog, pc, end);
        }
    }

    // Find matching BLOCK_END for a BLOCK_START at position pos
    static int find_block_end(const int* prog, int pos, int end) {
        // pos should point to BLOCK_START
        int depth = 1;
        int i = pos + 1;
        while (i < end && depth > 0) {
            if (prog[i] == OP_BLOCK_START) depth++;
            else if (prog[i] == OP_BLOCK_END) depth--;
            i++;
        }
        return i; // points one past the matching BLOCK_END
    }

    // Execute one instruction at prog[pc], return next pc
    int execute_one(const int* prog, int pc, int end) {
        if (pc >= end || halted) return end;
        charge_step();
        if (halted) return end;

        int op = prog[pc];
        pc++;

        switch ((Op)op) {

        // ============ Stack manipulation ============
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

        // ============ Float arithmetic ============
        case OP_FLOAT_ADD: {
            if (fs.has_n(2)) {
                double a = fs.pop(), b = fs.pop();
                charge_flops(1);
                fs.push(safe_float(a + b));
            }
            break;
        }
        case OP_FLOAT_SUB: {
            if (fs.has_n(2)) {
                double a = fs.pop(), b = fs.pop();
                charge_flops(1);
                fs.push(safe_float(b - a));
            }
            break;
        }
        case OP_FLOAT_MUL: {
            if (fs.has_n(2)) {
                double a = fs.pop(), b = fs.pop();
                charge_flops(1);
                fs.push(safe_float(a * b));
            }
            break;
        }
        case OP_FLOAT_DIV: {
            if (fs.has_n(2)) {
                double a = fs.pop(), b = fs.pop();
                charge_flops(1);
                if (std::abs(a) > 1e-30) fs.push(safe_float(b / a));
                else fs.push(0.0);
            }
            break;
        }
        case OP_FLOAT_ABS: {
            if (fs.has_pop()) { charge_flops(1); fs.push(std::abs(fs.pop())); }
            break;
        }
        case OP_FLOAT_NEG: {
            if (fs.has_pop()) fs.push(-fs.pop());
            break;
        }
        case OP_FLOAT_SQRT: {
            if (fs.has_pop()) {
                double a = fs.pop(); charge_flops(1);
                fs.push(std::sqrt(std::max(a, 0.0)));
            }
            break;
        }
        case OP_FLOAT_SQUARE: {
            if (fs.has_pop()) {
                double a = fs.pop(); charge_flops(1);
                fs.push(safe_float(a * a));
            }
            break;
        }
        case OP_FLOAT_MIN: {
            if (fs.has_n(2)) {
                double a = fs.pop(), b = fs.pop();
                charge_flops(1);
                fs.push(std::min(a, b));
            }
            break;
        }
        case OP_FLOAT_MAX: {
            if (fs.has_n(2)) {
                double a = fs.pop(), b = fs.pop();
                charge_flops(1);
                fs.push(std::max(a, b));
            }
            break;
        }
        case OP_FLOAT_EXP: {
            if (fs.has_pop()) {
                double a = fs.pop(); charge_flops(4);
                double c = std::max(-CLAMP_EXP, std::min(a, CLAMP_EXP));
                fs.push(std::exp(c));
            }
            break;
        }
        case OP_FLOAT_LOG: {
            if (fs.has_pop()) {
                double a = fs.pop(); charge_flops(4);
                fs.push(std::log(std::max(a, 1e-30)));
            }
            break;
        }
        case OP_FLOAT_TANH: {
            if (fs.has_pop()) {
                double a = fs.pop(); charge_flops(4);
                fs.push(std::tanh(a));
            }
            break;
        }

        // ============ Comparisons ============
        case OP_FLOAT_LT: {
            if (fs.has_n(2)) { double a=fs.pop(), b=fs.pop(); bs.push(b < a ? 1 : 0); }
            break;
        }
        case OP_FLOAT_GT: {
            if (fs.has_n(2)) { double a=fs.pop(), b=fs.pop(); bs.push(b > a ? 1 : 0); }
            break;
        }
        case OP_INT_LT: {
            if (is.has_n(2)) { int a=is.pop(), b=is.pop(); bs.push(b < a ? 1 : 0); }
            break;
        }
        case OP_INT_GT: {
            if (is.has_n(2)) { int a=is.pop(), b=is.pop(); bs.push(b > a ? 1 : 0); }
            break;
        }

        // ============ Int arithmetic ============
        case OP_INT_ADD: {
            if (is.has_n(2)) { int a=is.pop(), b=is.pop(); is.push(a+b); }
            break;
        }
        case OP_INT_SUB: {
            if (is.has_n(2)) { int a=is.pop(), b=is.pop(); is.push(b-a); }
            break;
        }
        case OP_INT_INC: {
            if (is.has_pop()) is.push(is.pop() + 1);
            break;
        }
        case OP_INT_DEC: {
            if (is.has_pop()) is.push(is.pop() - 1);
            break;
        }

        // ============ Bool logic ============
        case OP_BOOL_AND: {
            if (bs.has_n(2)) { int a=bs.pop(), b=bs.pop(); bs.push((a && b) ? 1 : 0); }
            break;
        }
        case OP_BOOL_OR: {
            if (bs.has_n(2)) { int a=bs.pop(), b=bs.pop(); bs.push((a || b) ? 1 : 0); }
            break;
        }
        case OP_BOOL_NOT: {
            if (bs.has_pop()) bs.push(bs.pop() ? 0 : 1);
            break;
        }

        // ============ Tensor ops ============
        case OP_MAT_VECMUL: {
            if (ms.has_pop() && vs.has_pop()) {
                CxMat A = ms.pop();
                CxVec x = vs.pop();
                if (A.cols == x.len && A.cols > 0) {
                    CxVec y;
                    mat_vec_mul(A, x, y);
                    charge_flops(A.rows * A.cols * 8);
                    vs.push(y);
                }
            }
            break;
        }
        case OP_VEC_ADD: {
            if (vs.has_n(2)) {
                CxVec a = vs.pop(), b = vs.pop();
                int n = std::min(a.len, b.len);
                CxVec r; r.len = n;
                for (int i=0; i<n; i++) r.data[i] = a.data[i] + b.data[i];
                charge_flops(n*2);
                vs.push(r);
            }
            break;
        }
        case OP_VEC_SUB: {
            if (vs.has_n(2)) {
                CxVec a = vs.pop(), b = vs.pop();
                int n = std::min(a.len, b.len);
                CxVec r; r.len = n;
                for (int i=0; i<n; i++) r.data[i] = b.data[i] - a.data[i];
                charge_flops(n*2);
                vs.push(r);
            }
            break;
        }
        case OP_VEC_DOT: {
            if (vs.has_n(2)) {
                CxVec a = vs.pop(), b = vs.pop();
                int n = std::min(a.len, b.len);
                cx dot(0,0);
                for (int i=0; i<n; i++) dot += std::conj(a.data[i]) * b.data[i];
                charge_flops(n*8);
                fs.push(std::real(dot));
            }
            break;
        }
        case OP_VEC_NORM2: {
            if (vs.has_pop()) {
                CxVec v = vs.pop();
                double s=0;
                for (int i=0; i<v.len; i++) s += std::norm(v.data[i]);
                charge_flops(v.len*5);
                fs.push(s);
            }
            break;
        }
        case OP_VEC_SCALE: {
            if (fs.has_pop() && vs.has_pop()) {
                double s = fs.pop();
                CxVec v = vs.pop();
                for (int i=0; i<v.len; i++) v.data[i] *= s;
                charge_flops(v.len*2);
                vs.push(v);
            }
            break;
        }
        case OP_VEC_ELEMENTAT: {
            if (is.has_pop() && vs.has_pop()) {
                int idx = is.pop();
                CxVec v = vs.pop();
                if (idx >= 0 && idx < v.len) {
                    fs.push(std::real(v.data[idx]));
                }
            }
            break;
        }
        case OP_MAT_ELEMENTAT: {
            if (is.has_n(2) && ms.has_pop()) {
                int c = is.pop(), r = is.pop();
                CxMat A = ms.pop();
                if (r >= 0 && r < A.rows && c >= 0 && c < A.cols) {
                    fs.push(std::real(A.data[r][c]));
                }
            }
            break;
        }
        case OP_MAT_ROW: {
            if (is.has_pop() && ms.has_pop()) {
                int r = is.pop();
                CxMat A = ms.pop();
                if (r >= 0 && r < A.rows) {
                    CxVec row; row.len = A.cols;
                    for (int j=0; j<A.cols; j++) row.data[j] = A.data[r][j];
                    vs.push(row);
                }
            }
            break;
        }
        case OP_VEC_LEN: {
            if (vs.has_pop()) {
                CxVec v = vs.peek();
                is.push(v.len);
            }
            break;
        }
        case OP_MAT_ROWS: {
            if (ms.has_pop()) {
                CxMat A = ms.peek();
                is.push(A.rows);
            }
            break;
        }

        // ============ Peek-based access ============
        case OP_MAT_PEEKAT: {
            if (is.has_n(2) && ms.depth() > 0) {
                int c = is.pop(), r = is.pop();
                CxMat& A = ms.data[ms.top-1];
                if (r>=0 && r<A.rows && c>=0 && c<A.cols)
                    fs.push(std::real(A.data[r][c]));
            }
            break;
        }
        case OP_VEC_PEEKAT: {
            if (is.has_pop() && vs.depth() > 0) {
                int idx = is.pop();
                CxVec& v = vs.data[vs.top-1];
                if (idx>=0 && idx<v.len) fs.push(std::real(v.data[idx]));
            }
            break;
        }
        case OP_MAT_PEEKATIM: {
            if (is.has_n(2) && ms.depth() > 0) {
                int c = is.pop(), r = is.pop();
                CxMat& A = ms.data[ms.top-1];
                if (r>=0 && r<A.rows && c>=0 && c<A.cols)
                    fs.push(std::imag(A.data[r][c]));
            }
            break;
        }
        case OP_VEC_PEEKATIM: {
            if (is.has_pop() && vs.depth() > 0) {
                int idx = is.pop();
                CxVec& v = vs.data[vs.top-1];
                if (idx>=0 && idx<v.len) fs.push(std::imag(v.data[idx]));
            }
            break;
        }
        case OP_VEC_SECONDPEEKAT: {
            if (is.has_pop() && vs.depth() >= 2) {
                int idx = is.pop();
                CxVec& v = vs.data[vs.top-2]; // second on stack
                if (idx>=0 && idx<v.len) fs.push(std::real(v.data[idx]));
            }
            break;
        }
        case OP_VEC_SECONDPEEKATIM: {
            if (is.has_pop() && vs.depth() >= 2) {
                int idx = is.pop();
                CxVec& v = vs.data[vs.top-2];
                if (idx>=0 && idx<v.len) fs.push(std::imag(v.data[idx]));
            }
            break;
        }

        // ============ High-level primitives ============
        case OP_VEC_GETRESIDUE: {
            // residue = y_tilde[0:k] - R[0:k, k:Nt] @ x_partial_reversed
            if (is.depth() > 0 && vs.depth() >= 2 && ms.depth() >= 1) {
                int k = is.peek();
                CxVec& xp = vs.data[vs.top-1];  // top: x_partial
                CxVec& yt = vs.data[vs.top-2];  // second: y_tilde
                CxMat& R  = ms.data[ms.top-1];
                if (k > 0 && k <= R.cols && xp.len == R.cols - k) {
                    CxVec res; res.len = k;
                    int n_part = xp.len;
                    for (int i=0; i<k; i++) {
                        cx interference(0,0);
                        // x_partial is deepest-first: xp[0]=x[Nt-1], ..., xp[-1]=x[k]
                        // R[i, k:Nt] @ x_ordered where x_ordered[j] = xp[n_part-1-j]
                        for (int j=0; j<n_part; j++) {
                            int col = k + j;
                            interference += R.data[i][col] * xp.data[n_part-1-j];
                        }
                        res.data[i] = yt.data[i] - interference;
                    }
                    charge_flops(k * n_part * 8);
                    vs.push(res);
                }
            }
            break;
        }
        case OP_FLOAT_GETMMSELB: {
            // MMSE LB = r^H r - r^H R_sub (R_sub^H R_sub + nv*I)^{-1} R_sub^H r
            if (is.depth() > 0 && vs.depth() >= 2 && ms.depth() >= 1) {
                int k = is.peek();
                CxVec& xp = vs.data[vs.top-1];
                CxVec& yt = vs.data[vs.top-2];
                CxMat& R  = ms.data[ms.top-1];
                double nv = noise_var;
                if (k > 0 && k <= R.cols && xp.len == R.cols - k) {
                    int n_part = xp.len;
                    // Compute residue
                    cx residue[MAX_DIM];
                    for (int i=0; i<k; i++) {
                        cx interf(0,0);
                        for (int j=0; j<n_part; j++)
                            interf += R.data[i][k+j] * xp.data[n_part-1-j];
                        residue[i] = yt.data[i] - interf;
                    }
                    // R_sub = R[0:k, 0:k]
                    // Gram = R_sub^H R_sub + nv*I
                    cx Gram[MAX_DIM][MAX_DIM];
                    for (int i=0; i<k; i++) {
                        for (int j=0; j<k; j++) {
                            cx s(0,0);
                            for (int m=0; m<k; m++)
                                s += std::conj(R.data[m][i]) * R.data[m][j];
                            Gram[i][j] = s;
                        }
                        Gram[i][i] += cx(nv, 0);
                    }
                    // Rh_r = R_sub^H @ residue
                    cx Rh_r[MAX_DIM];
                    for (int i=0; i<k; i++) {
                        cx s(0,0);
                        for (int m=0; m<k; m++)
                            s += std::conj(R.data[m][i]) * residue[m];
                        Rh_r[i] = s;
                    }
                    // Solve Gram @ t = Rh_r
                    cx t[MAX_DIM];
                    if (solve_linear(Gram, Rh_r, t, k)) {
                        // mmse_lb = r^H r - Rh_r^H t
                        cx rHr(0,0);
                        for (int i=0; i<k; i++) rHr += std::conj(residue[i]) * residue[i];
                        cx corr(0,0);
                        for (int i=0; i<k; i++) corr += std::conj(Rh_r[i]) * t[i];
                        double lb = std::max(0.0, std::real(rHr - corr));
                        charge_flops(k*k*k + k*k*(R.cols-k)*8);
                        fs.push(lb);
                    } else {
                        fs.push(0.0);
                    }
                }
            }
            break;
        }

        // ============ Node memory ============
        case OP_NODE_READMEM: {
            if (is.has_pop() && ns.has_pop()) {
                int slot = is.pop();
                int nid = ns.peek();
                if (tree && nid >= 0 && nid < tree->n_nodes &&
                    slot >= 0 && slot < N_MEM)
                    fs.push(tree->nodes[nid].mem[slot]);
            }
            break;
        }
        case OP_NODE_WRITEMEM: {
            if (is.has_pop() && fs.has_pop() && ns.has_pop()) {
                int slot = is.pop();
                double val = fs.pop();
                int nid = ns.peek();
                if (tree && nid >= 0 && nid < tree->n_nodes &&
                    slot >= 0 && slot < N_MEM && std::isfinite(val))
                    tree->nodes[nid].mem[slot] = val;
            }
            break;
        }
        case OP_NODE_GETCUMDIST: {
            if (ns.has_pop() && tree) {
                int nid = ns.peek();
                if (nid >= 0 && nid < tree->n_nodes)
                    fs.push(tree->nodes[nid].cum_dist);
            }
            break;
        }
        case OP_NODE_GETLOCALDIST: {
            if (ns.has_pop() && tree) {
                int nid = ns.peek();
                if (nid >= 0 && nid < tree->n_nodes)
                    fs.push(tree->nodes[nid].local_dist);
            }
            break;
        }
        case OP_NODE_GETSYMRE: {
            if (ns.has_pop() && tree) {
                int nid = ns.peek();
                if (nid >= 0 && nid < tree->n_nodes)
                    fs.push(std::real(tree->nodes[nid].symbol));
            }
            break;
        }
        case OP_NODE_GETSYMIM: {
            if (ns.has_pop() && tree) {
                int nid = ns.peek();
                if (nid >= 0 && nid < tree->n_nodes)
                    fs.push(std::imag(tree->nodes[nid].symbol));
            }
            break;
        }
        case OP_NODE_GETLAYER: {
            if (ns.has_pop() && tree) {
                int nid = ns.peek();
                if (nid >= 0 && nid < tree->n_nodes)
                    is.push(tree->nodes[nid].layer);
            }
            break;
        }

        // ============ Graph navigation ============
        case OP_GRAPH_GETROOT: {
            if (tree && tree->root_id >= 0)
                ns.push(tree->root_id);
            break;
        }
        case OP_NODE_GETPARENT: {
            if (ns.has_pop() && tree) {
                int nid = ns.peek();
                if (nid >= 0 && nid < tree->n_nodes && tree->nodes[nid].parent)
                    ns.push(tree->nodes[nid].parent->id);
            }
            break;
        }
        case OP_NODE_NUMCHILDREN: {
            if (ns.has_pop() && tree) {
                int nid = ns.peek();
                if (nid >= 0 && nid < tree->n_nodes)
                    is.push(tree->nodes[nid].n_children);
            }
            break;
        }
        case OP_NODE_CHILDAT: {
            if (is.has_pop() && ns.has_pop() && tree) {
                int idx = is.pop();
                int nid = ns.peek();
                if (nid >= 0 && nid < tree->n_nodes) {
                    TreeNode& nd = tree->nodes[nid];
                    if (idx >= 0 && idx < nd.n_children)
                        ns.push(nd.children[idx]);
                }
            }
            break;
        }
        case OP_GRAPH_NODECOUNT: {
            if (tree) is.push(tree->n_nodes);
            break;
        }
        case OP_GRAPH_FRONTIERCOUNT: {
            if (tree) is.push(tree->frontier_count());
            break;
        }

        // ============ BP-essential ============
        case OP_NODE_GETSCORE: {
            if (ns.has_pop() && tree) {
                int nid = ns.peek();
                if (nid >= 0 && nid < tree->n_nodes)
                    fs.push(tree->nodes[nid].score);
            }
            break;
        }
        case OP_NODE_SETSCORE: {
            if (fs.has_pop() && ns.has_pop() && tree) {
                double val = fs.pop();
                int nid = ns.peek();
                if (nid >= 0 && nid < tree->n_nodes && std::isfinite(val)) {
                    tree->nodes[nid].score = val;
                    tree->nodes[nid].queue_version++;
                    tree->add_dirty(nid);
                }
            }
            break;
        }
        case OP_NODE_ISEXPANDED: {
            if (ns.has_pop() && tree) {
                int nid = ns.peek();
                if (nid >= 0 && nid < tree->n_nodes)
                    bs.push(tree->nodes[nid].is_expanded ? 1 : 0);
            }
            break;
        }

        // ============ Constants ============
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

        // ============ Environment ============
        case OP_FLOAT_GETNOISEVAR: fs.push(noise_var); break;
        case OP_INT_GETNUMSYMBOLS: is.push(M); break;

        // ============ Type conversion ============
        case OP_FLOAT_FROMINT: {
            if (is.has_pop()) fs.push((double)is.pop());
            break;
        }
        case OP_INT_FROMFLOAT: {
            if (fs.has_pop()) {
                double a = fs.pop();
                int v = (int)std::max(-1e6, std::min(a, 1e6));
                is.push(v);
            }
            break;
        }

        // ============ Control flow ============
        case OP_EXEC_IF: {
            // Expects: BLOCK_START <then> BLOCK_END BLOCK_START <else> BLOCK_END
            if (pc < end && prog[pc] == OP_BLOCK_START) {
                int then_start = pc + 1;
                int then_end_pos = find_block_end(prog, pc, end);
                int else_start = -1, else_end_pos = then_end_pos;
                if (then_end_pos < end && prog[then_end_pos] == OP_BLOCK_START) {
                    else_start = then_end_pos + 1;
                    else_end_pos = find_block_end(prog, then_end_pos, end);
                }
                if (bs.has_pop()) {
                    int cond = bs.pop();
                    if (cond)
                        execute_block(prog, then_start, then_end_pos - 1);
                    else if (else_start >= 0)
                        execute_block(prog, else_start, else_end_pos - 1);
                }
                pc = else_end_pos;
            }
            break;
        }
        case OP_EXEC_WHILE: {
            if (pc < end && prog[pc] == OP_BLOCK_START) {
                int body_start = pc + 1;
                int body_end_pos = find_block_end(prog, pc, end);
                for (int iter = 0; iter < MAX_CTRL_ITER && !halted; iter++) {
                    if (!bs.has_pop()) break;
                    int cond = bs.pop();
                    if (!cond) break;
                    execute_block(prog, body_start, body_end_pos - 1);
                }
                pc = body_end_pos;
            }
            break;
        }
        case OP_EXEC_DOTIMES: {
            if (pc < end && prog[pc] == OP_BLOCK_START) {
                int body_start = pc + 1;
                int body_end_pos = find_block_end(prog, pc, end);
                int count = 0;
                if (is.has_pop()) count = std::max(0, std::min(is.pop(), MAX_DOTIMES));
                for (int i = 0; i < count && !halted; i++)
                    execute_block(prog, body_start, body_end_pos - 1);
                pc = body_end_pos;
            }
            break;
        }
        case OP_NODE_FOREACHCHILD: {
            if (pc < end && prog[pc] == OP_BLOCK_START) {
                int body_start = pc + 1;
                int body_end_pos = find_block_end(prog, pc, end);
                if (ns.has_pop() && tree) {
                    int nid = ns.peek();
                    if (nid >= 0 && nid < tree->n_nodes) {
                        TreeNode& nd = tree->nodes[nid];
                        double acc = 0;
                        int limit = std::min(nd.n_children, MAX_CTRL_ITER);
                        for (int i = 0; i < limit && !halted; i++) {
                            int orig_depth = ns.depth();
                            ns.push(nd.children[i]);
                            execute_block(prog, body_start, body_end_pos - 1);
                            if (fs.has_pop()) {
                                double v = fs.pop();
                                if (std::isfinite(v)) acc += v;
                            }
                            while (ns.depth() > orig_depth) ns.pop();
                        }
                        fs.push(acc);
                    }
                }
                pc = body_end_pos;
            }
            break;
        }
        case OP_NODE_FOREACHSIBLING: {
            if (pc < end && prog[pc] == OP_BLOCK_START) {
                int body_start = pc + 1;
                int body_end_pos = find_block_end(prog, pc, end);
                if (ns.has_pop() && tree) {
                    int nid = ns.peek();
                    int sibs[64];
                    int n_sibs = tree->get_siblings(nid, sibs, 64);
                    double acc = 0;
                    int limit = std::min(n_sibs, MAX_CTRL_ITER);
                    for (int i = 0; i < limit && !halted; i++) {
                        int orig_depth = ns.depth();
                        ns.push(sibs[i]);
                        execute_block(prog, body_start, body_end_pos - 1);
                        if (fs.has_pop()) {
                            double v = fs.pop();
                            if (std::isfinite(v)) acc += v;
                        }
                        while (ns.depth() > orig_depth) ns.pop();
                    }
                    fs.push(acc);
                }
                pc = body_end_pos;
            }
            break;
        }
        case OP_NODE_FOREACHANCESTOR: {
            if (pc < end && prog[pc] == OP_BLOCK_START) {
                int body_start = pc + 1;
                int body_end_pos = find_block_end(prog, pc, end);
                if (ns.has_pop() && tree) {
                    int nid = ns.peek();
                    int ancs[MAX_DIM];
                    int n_ancs = tree->get_ancestors(nid, ancs, MAX_DIM);
                    double acc = 0;
                    int limit = std::min(n_ancs, MAX_CTRL_ITER);
                    for (int i = 0; i < limit && !halted; i++) {
                        int orig_depth = ns.depth();
                        ns.push(ancs[i]);
                        execute_block(prog, body_start, body_end_pos - 1);
                        if (fs.has_pop()) {
                            double v = fs.pop();
                            if (std::isfinite(v)) acc += v;
                        }
                        while (ns.depth() > orig_depth) ns.pop();
                    }
                    fs.push(acc);
                }
                pc = body_end_pos;
            }
            break;
        }
        case OP_EXEC_FOREACHSYMBOL: {
            if (pc < end && prog[pc] == OP_BLOCK_START) {
                int body_start = pc + 1;
                int body_end_pos = find_block_end(prog, pc, end);
                if (constellation && M > 0) {
                    // Save outer float stack
                    double outer[STACK_DEPTH];
                    int outer_n = fs.top;
                    for (int i=0; i<outer_n; i++) outer[i] = fs.data[i];
                    fs.clear();

                    double acc = 0;
                    for (int s = 0; s < M && !halted; s++) {
                        fs.push(std::real(constellation[s]));
                        fs.push(std::imag(constellation[s]));
                        execute_block(prog, body_start, body_end_pos - 1);
                        if (fs.has_pop()) {
                            double v = fs.pop();
                            if (std::isfinite(v)) acc += v;
                        }
                        fs.clear();
                    }

                    // Restore outer float stack
                    for (int i=0; i<outer_n; i++) fs.data[i] = outer[i];
                    fs.top = outer_n;
                    fs.push(acc);
                }
                pc = body_end_pos;
            }
            break;
        }
        case OP_EXEC_MINOVERSYMBOLS: {
            if (pc < end && prog[pc] == OP_BLOCK_START) {
                int body_start = pc + 1;
                int body_end_pos = find_block_end(prog, pc, end);
                if (constellation && M > 0) {
                    double outer[STACK_DEPTH];
                    int outer_n = fs.top;
                    for (int i=0; i<outer_n; i++) outer[i] = fs.data[i];
                    fs.clear();

                    double acc = std::numeric_limits<double>::infinity();
                    for (int s = 0; s < M && !halted; s++) {
                        fs.push(std::real(constellation[s]));
                        fs.push(std::imag(constellation[s]));
                        execute_block(prog, body_start, body_end_pos - 1);
                        if (fs.has_pop()) {
                            double v = fs.pop();
                            if (std::isfinite(v)) acc = std::min(acc, v);
                        }
                        fs.clear();
                    }

                    for (int i=0; i<outer_n; i++) fs.data[i] = outer[i];
                    fs.top = outer_n;
                    if (!std::isfinite(acc)) acc = 0.0;
                    fs.push(acc);
                }
                pc = body_end_pos;
            }
            break;
        }

        // Block markers (should not be reached during normal execution)
        case OP_BLOCK_START:
        case OP_BLOCK_END:
            break;

        default:
            break;
        }

        return pc;
    }
};


// =========================================================================
// Stack Decoder
// =========================================================================
struct PQEntry {
    double score;
    int queue_version;
    int counter;
    int node_id;

    bool operator>(const PQEntry& o) const {
        if (score != o.score) return score > o.score;
        if (queue_version != o.queue_version) return queue_version > o.queue_version;
        return counter > o.counter;
    }
};

struct Decoder {
    int Nt, Nr, M;
    int max_nodes;
    const cx* constellation;
    PushVM vm;

    void init(int Nt_, int Nr_, int M_, const cx* cons,
              int max_nodes_, int flops_max_, int step_max_) {
        Nt = Nt_; Nr = Nr_; M = M_;
        constellation = cons;
        max_nodes = max_nodes_;
        vm.flops_max = flops_max_;
        vm.step_max = step_max_;
    }

    // Score a node by running the program
    double score_node(int nid, const CxMat& R, const CxVec& y_tilde,
                      const int* prog, int prog_len,
                      SearchTree& tree, double noise_var) {
        TreeNode& nd = tree.nodes[nid];
        vm.inject_environment(R, y_tilde,
                              nd.partial_symbols, nd.n_partial,
                              &tree, nid, nd.layer,
                              constellation, M, noise_var, Nt, Nr);
        double correction = vm.run(prog, prog_len);
        if (!std::isfinite(correction))
            return nd.cum_dist;
        return nd.cum_dist + correction;
    }

    // Process dirty nodes — update PQ after program-driven SetScore
    int process_dirty(SearchTree& tree,
                      std::priority_queue<PQEntry, std::vector<PQEntry>,
                                          std::greater<PQEntry>>& pq,
                      int counter) {
        for (int i = 0; i < tree.n_dirty; i++) {
            int nid = tree.dirty[i];
            TreeNode& nd = tree.nodes[nid];
            if (!nd.is_expanded) {
                pq.push({nd.score, nd.queue_version, counter, nid});
                counter++;
            }
        }
        tree.clear_dirty();
        return counter;
    }

    // Greedy completion of a partial path
    void complete_path(int nid, const CxMat& R, const CxVec& y_tilde,
                       SearchTree& tree, cx* x_out) {
        TreeNode& nd = tree.nodes[nid];
        // Copy partial symbols
        cx decided[MAX_DIM];
        int n_dec = nd.n_partial;
        for (int i = 0; i < n_dec; i++) decided[i] = nd.partial_symbols[i];

        int cur_layer = nd.layer - 1;
        while (cur_layer >= 0) {
            cx best_sym = constellation[0];
            double best_ld = 1e30;
            for (int s = 0; s < M; s++) {
                cx sym = constellation[s];
                // Compute local distance at cur_layer with decided + sym
                cx interf(0,0);
                int n_cand = n_dec + 1;
                for (int j = 0; j < n_dec; j++) {
                    int col = Nt - 1 - j;
                    interf += R.data[cur_layer][col] * decided[j];
                }
                interf += R.data[cur_layer][Nt - 1 - n_dec] * sym;
                cx residual = y_tilde.data[cur_layer] - interf;
                double ld = std::norm(residual);
                if (ld < best_ld) {
                    best_ld = ld;
                    best_sym = sym;
                }
            }
            decided[n_dec++] = best_sym;
            cur_layer--;
        }

        // Output: decided is in deepest-first order, reverse for x_out
        for (int i = 0; i < Nt; i++)
            x_out[i] = (i < n_dec) ? decided[n_dec - 1 - i] : cx(0,0);
    }

    // Main detection
    double detect(const cx* H_flat, const cx* y_in, const cx* x_true,
                  const int* prog, int prog_len,
                  double noise_var, int* flops_out) {
        // Build H matrix
        CxMat H(Nr, Nt);
        for (int i = 0; i < Nr; i++)
            for (int j = 0; j < Nt; j++)
                H.data[i][j] = H_flat[i * Nt + j];

        // QR decomposition
        CxMat Q, R;
        qr_mgs(H, Q, R);
        int flops = 2 * Nr * Nt * Nt;

        // y_tilde = Q^H @ y
        CxVec y_vec; y_vec.len = Nr;
        for (int i = 0; i < Nr; i++) y_vec.data[i] = y_in[i];
        CxVec y_tilde; y_tilde.len = Nt;
        for (int i = 0; i < Nt; i++) {
            cx s(0,0);
            for (int j = 0; j < Nr; j++) s += std::conj(Q.data[j][i]) * y_vec.data[j];
            y_tilde.data[i] = s;
        }
        flops += 8 * Nt * Nr;

        // Init search tree (heap-allocated to avoid stack overflow)
        SearchTree* treep = new SearchTree(max_nodes + M * max_nodes + 16);
        SearchTree& tree = *treep;
        int root = tree.create_root(Nt);

        std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;
        int counter = 0;

        // Expand root
        int k0 = Nt - 1;
        for (int s = 0; s < M; s++) {
            cx sym = constellation[s];
            cx residual = y_tilde.data[k0] - R.data[k0][k0] * sym;
            double ld = std::norm(residual);
            flops += 11;
            cx part[1] = {sym};
            int cid = tree.add_child(root, k0, sym, ld, ld, part, 1);
            if (cid < 0) break;
            double sc = score_node(cid, R, y_tilde, prog, prog_len, tree, noise_var);
            tree.nodes[cid].score = sc;
            flops += vm.flops_count;
            pq.push({sc, tree.nodes[cid].queue_version, counter, cid});
            counter++;
        }
        counter = process_dirty(tree, pq, counter);

        int best_id = -1;
        double best_score = 1e30;

        while (!pq.empty() && tree.n_nodes < max_nodes + 1) {
            PQEntry top = pq.top();
            pq.pop();
            if (top.queue_version != tree.nodes[top.node_id].queue_version)
                continue; // stale

            if (top.score < best_score) {
                best_score = top.score;
                best_id = top.node_id;
            }

            TreeNode& node = tree.nodes[top.node_id];
            if (node.layer == 0) {
                // Complete path found
                cx x_out[MAX_DIM];
                for (int i = 0; i < Nt; i++)
                    x_out[i] = (i < node.n_partial)
                        ? node.partial_symbols[node.n_partial - 1 - i] : cx(0,0);
                if (flops_out) *flops_out = flops;
                int errors = 0;
                for (int i = 0; i < Nt; i++)
                    if (x_out[i] != x_true[i]) errors++;
                double ber = (double)errors / Nt;
                delete treep;
                return ber;
            }

            tree.mark_expanded(top.node_id);
            int next_layer = node.layer - 1;

            for (int s = 0; s < M; s++) {
                cx sym = constellation[s];
                // Build new partial
                cx new_partial[MAX_DIM];
                int n_new = node.n_partial + 1;
                for (int i = 0; i < node.n_partial; i++)
                    new_partial[i] = node.partial_symbols[i];
                new_partial[node.n_partial] = sym;

                // Compute local distance
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

                int cid = tree.add_child(top.node_id, next_layer, sym,
                                         ld, cd, new_partial, n_new);
                if (cid < 0) break;

                double sc = score_node(cid, R, y_tilde, prog, prog_len,
                                       tree, noise_var);
                tree.nodes[cid].score = sc;
                flops += vm.flops_count;
                pq.push({sc, tree.nodes[cid].queue_version, counter, cid});
                counter++;
            }

            counter = process_dirty(tree, pq, counter);
        }

        // Greedy completion
        cx x_out[MAX_DIM];
        if (best_id >= 0)
            complete_path(best_id, R, y_tilde, tree, x_out);
        else
            for (int i = 0; i < Nt; i++) x_out[i] = cx(0,0);

        if (flops_out) *flops_out = flops;

        int errors = 0;
        for (int i = 0; i < Nt; i++)
            if (x_out[i] != x_true[i]) errors++;
        double ber = (double)errors / Nt;
        delete treep;
        return ber;
    }
};


// =========================================================================
// Evaluator Context
// =========================================================================
struct EvalContext {
    int Nt, Nr, M;
    cx constellation[256]; // max 256-QAM
    int max_nodes;
    int flops_max, step_max;
};


// =========================================================================
// C API Implementation
// =========================================================================
extern "C" {

EXPORT void* eval_create(int Nt, int Nr, int M,
                         const double* cons_interleaved,
                         int max_nodes, int flops_max, int step_max) {
    auto* ctx = new EvalContext();
    ctx->Nt = Nt;
    ctx->Nr = Nr;
    ctx->M = M;
    ctx->max_nodes = max_nodes;
    ctx->flops_max = flops_max;
    ctx->step_max = step_max;
    for (int i = 0; i < M; i++)
        ctx->constellation[i] = cx(cons_interleaved[2*i], cons_interleaved[2*i+1]);
    return ctx;
}

EXPORT void eval_destroy(void* ctx) {
    delete (EvalContext*)ctx;
}

EXPORT double eval_one(void* ctx_,
                       const int* program, int prog_len,
                       const double* H_interleaved,
                       const double* y_interleaved,
                       const double* x_true_interleaved,
                       double noise_var,
                       double* flops_out) {
    auto* ctx = (EvalContext*)ctx_;
    int Nt = ctx->Nt, Nr = ctx->Nr;

    // Deinterleave complex arrays
    cx H_flat[MAX_DIM * MAX_DIM];
    for (int i = 0; i < Nr * Nt; i++)
        H_flat[i] = cx(H_interleaved[2*i], H_interleaved[2*i+1]);

    cx y[MAX_DIM];
    for (int i = 0; i < Nr; i++)
        y[i] = cx(y_interleaved[2*i], y_interleaved[2*i+1]);

    cx x_true[MAX_DIM];
    for (int i = 0; i < Nt; i++)
        x_true[i] = cx(x_true_interleaved[2*i], x_true_interleaved[2*i+1]);

    Decoder dec;
    dec.init(Nt, Nr, ctx->M, ctx->constellation,
             ctx->max_nodes, ctx->flops_max, ctx->step_max);

    int fl = 0;
    double ber = dec.detect(H_flat, y, x_true, program, prog_len,
                            noise_var, &fl);
    if (flops_out) *flops_out = (double)fl;
    return ber;
}

EXPORT double eval_dataset(void* ctx_,
                           const int* program, int prog_len,
                           int n_samples,
                           const double* H_all,
                           const double* y_all,
                           const double* x_true_all,
                           const double* noise_vars,
                           double* avg_flops_out) {
    auto* ctx = (EvalContext*)ctx_;
    int Nt = ctx->Nt, Nr = ctx->Nr;
    int H_stride = 2 * Nr * Nt;
    int y_stride = 2 * Nr;
    int x_stride = 2 * Nt;

    double total_ber = 0;
    double total_flops = 0;

    for (int s = 0; s < n_samples; s++) {
        double fl = 0;
        double ber = eval_one(ctx_, program, prog_len,
                              H_all + s * H_stride,
                              y_all + s * y_stride,
                              x_true_all + s * x_stride,
                              noise_vars[s], &fl);
        total_ber += ber;
        total_flops += fl;
    }

    if (avg_flops_out) *avg_flops_out = total_flops / n_samples;
    return total_ber / n_samples;
}

EXPORT void eval_batch(void* ctx_,
                       int n_programs,
                       const int** programs,
                       const int* prog_lengths,
                       int n_samples,
                       const double* H_all,
                       const double* y_all,
                       const double* x_true_all,
                       const double* noise_vars,
                       double* ber_out,
                       double* flops_out) {
    #pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < n_programs; p++) {
        double avg_fl = 0;
        double ber = eval_dataset(ctx_, programs[p], prog_lengths[p],
                                  n_samples, H_all, y_all, x_true_all,
                                  noise_vars, &avg_fl);
        ber_out[p] = ber;
        if (flops_out) flops_out[p] = avg_fl;
    }
}

}  // extern "C"
