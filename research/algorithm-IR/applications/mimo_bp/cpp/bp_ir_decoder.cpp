/*
 * bp_ir_decoder.cpp — Structured BP Stack Decoder using IR expression evaluator.
 *
 * Combines the tree-search stack decoder with BP message passing,
 * where the 4 evolved programs (f_down, f_up, f_belief, h_halt)
 * are evaluated via ir_eval() from flat opcode arrays.
 *
 * Compile (MSVC):
 *   cl.exe /EHsc /O2 /openmp /std:c++17 /DBUILD_DLL bp_ir_decoder.cpp /LD /Fe:bp_ir_eval.dll
 */

#include "ir_eval.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <algorithm>
#include <vector>

#ifdef _WIN32
  #define IR_EXPORT extern "C" __declspec(dllexport)
#else
  #define IR_EXPORT extern "C" __attribute__((visibility("default")))
#endif

static constexpr int MAX_DIM = 32;
static constexpr int MAX_NODES = 8192;

using cx = std::complex<double>;

// ============================================================================
// Complex linear algebra (stack-allocated, no external deps)
// ============================================================================

struct CxVec {
    cx data[MAX_DIM];
    int len;
    CxVec() : len(0) { std::memset(data, 0, sizeof(data)); }
    CxVec(int n) : len(n) { std::memset(data, 0, sizeof(data)); }
};

struct CxMat {
    cx data[MAX_DIM][MAX_DIM];
    int rows, cols;
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

// LMMSE detection: x_hat = (H^H H + sigma^2 I)^{-1} H^H y
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

// ============================================================================
// Tree node + search tree
// ============================================================================

struct TreeNode {
    int id = -1;
    int layer = 0;
    cx  symbol{0, 0};
    double local_dist = 0;
    double cum_dist = 0;
    double score = 0;
    double m_up = 0;
    double m_down = 0;
    int queue_version = 0;
    bool is_expanded = false;
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

    void reset() {
        n_nodes = 0;
        root_id = -1;
    }

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

// ============================================================================
// Priority queue entry
// ============================================================================

struct PQEntry {
    double score;
    int node_id;
    int version;
    bool operator>(const PQEntry& o) const { return score > o.score; }
};

// Simple min-heap (we'll just use sorted vectors for simplicity)

// ============================================================================
// BP-IR Stack Decoder
// ============================================================================

struct BPIRDecoder {
    int Nt, Nr, M;
    cx constellation[256];
    int max_nodes;
    int max_bp_iters;

    // Precomputed
    CxMat R;
    CxVec y_tilde;

    BPIRDecoder(int nt, int nr, int m, const cx* cons,
                int max_n, int max_bp)
        : Nt(nt), Nr(nr), M(m), max_nodes(max_n), max_bp_iters(max_bp)
    {
        for (int i = 0; i < m; i++) constellation[i] = cons[i];
    }

    // Preprocess: QR decomposition and y_tilde = Q^H y
    void preprocess(const cx* H_flat, const cx* y_flat) {
        CxMat H(Nr, Nt);
        for (int i = 0; i < Nr; i++)
            for (int j = 0; j < Nt; j++)
                H.data[i][j] = H_flat[i * Nt + j];

        CxMat Q(Nr, Nt);
        qr_mgs(H, Q, R);

        y_tilde = CxVec(Nt);
        for (int j = 0; j < Nt; j++) {
            cx sum(0, 0);
            for (int i = 0; i < Nr; i++)
                sum += std::conj(Q.data[i][j]) * y_flat[i];
            y_tilde.data[j] = sum;
        }
    }

    // Expand a node: create M children at next layer
    int expand(SearchTree& tree, int parent_id) {
        TreeNode& parent = tree.nodes[parent_id];
        int next_layer = parent.layer - 1;
        if (next_layer < 0) return 0;

        int count = 0;
        for (int s = 0; s < M; s++) {
            cx sym = constellation[s];

            // Build partial symbol vector
            cx partial[MAX_DIM];
            int n_partial = parent.n_partial + 1;
            for (int i = 0; i < parent.n_partial; i++)
                partial[i] = parent.partial_symbols[i];
            partial[parent.n_partial] = sym;

            // Compute local distance: |y_tilde[layer] - sum R[layer][j]*partial[j]|^2
            cx interference(0, 0);
            for (int j = 0; j < n_partial; j++) {
                int col = Nt - 1 - j;  // reverse layer ordering
                if (col >= 0 && col < Nt && next_layer < Nt)
                    interference += R.data[next_layer][col] * partial[j];
            }
            double ld = std::norm(y_tilde.data[next_layer] - interference);
            double cd = parent.cum_dist + ld;

            int child_id = tree.add_child(parent_id, next_layer, sym, ld, cd, partial, n_partial);
            if (child_id >= 0) count++;
        }
        tree.mark_expanded(parent_id);
        return count;
    }

    // ------- BP message passing using IR programs -------

    void full_up_sweep(SearchTree& tree,
                       const int* prog_up, int up_len) {
        int bfs[MAX_NODES];
        int bfs_count = 0;
        tree.bfs_order(bfs, bfs_count);

        // Reverse BFS: leaves first
        for (int idx = bfs_count - 1; idx >= 0; idx--) {
            TreeNode& nd = tree.nodes[bfs[idx]];
            if (nd.n_children == 0) {
                // Leaf: m_up = local_dist
                nd.m_up = nd.local_dist;
            } else {
                // Internal: aggregate children via prog_up
                // prog_up args: [child_local_dists..., child_m_ups..., n_children]
                // Simplified: pass (sum_child_ld, sum_child_m_up, n_children)
                double sum_ld = 0, sum_mu = 0;
                for (int c = 0; c < nd.n_children; c++) {
                    TreeNode& ch = tree.nodes[nd.children[c]];
                    sum_ld += ch.local_dist;
                    sum_mu += ch.m_up;
                }
                double args[3] = { sum_ld, sum_mu, (double)nd.n_children };
                nd.m_up = ir_eval(prog_up, up_len, args, 3);
            }
        }
    }

    void full_down_sweep(SearchTree& tree,
                         const int* prog_down, int down_len) {
        int bfs[MAX_NODES];
        int bfs_count = 0;
        tree.bfs_order(bfs, bfs_count);

        // Forward BFS: root first (skip root itself)
        for (int idx = 0; idx < bfs_count; idx++) {
            TreeNode& nd = tree.nodes[bfs[idx]];
            if (nd.parent_id < 0) {
                nd.m_down = 0.0;  // root has no parent message
                continue;
            }
            TreeNode& parent = tree.nodes[nd.parent_id];
            double args[2] = { parent.m_down, nd.local_dist };
            nd.m_down = ir_eval(prog_down, down_len, args, 2);
        }
    }

    void score_all_frontier(SearchTree& tree,
                            const int* prog_belief, int belief_len) {
        for (int i = 0; i < tree.n_nodes; i++) {
            TreeNode& nd = tree.nodes[i];
            if (nd.id == tree.root_id) continue;
            if (nd.is_expanded) continue;
            double args[3] = { nd.cum_dist, nd.m_down, nd.m_up };
            nd.score = ir_eval(prog_belief, belief_len, args, 3);
        }
    }

    bool check_halt(const int* prog_halt, int halt_len,
                    double old_root_m_up, double new_root_m_up) {
        double args[2] = { old_root_m_up, new_root_m_up };
        double result = ir_eval(prog_halt, halt_len, args, 2);
        return result > 0.5;
    }

    void full_bp_cycle(SearchTree& tree,
                       const int* prog_down, int down_len,
                       const int* prog_up, int up_len,
                       const int* prog_belief, int belief_len,
                       const int* prog_halt, int halt_len) {
        for (int bp_iter = 0; bp_iter < max_bp_iters; bp_iter++) {
            double old_root_mu = tree.nodes[tree.root_id].m_up;

            full_up_sweep(tree, prog_up, up_len);
            full_down_sweep(tree, prog_down, down_len);
            score_all_frontier(tree, prog_belief, belief_len);

            if (bp_iter < max_bp_iters - 1) {
                double new_root_mu = tree.nodes[tree.root_id].m_up;
                if (check_halt(prog_halt, halt_len, old_root_mu, new_root_mu))
                    break;
            }
        }
    }

    // Complete a partial solution greedily (pick min-distance symbol)
    void complete_greedy(SearchTree& tree, int node_id, cx* x_out) {
        TreeNode& nd = tree.nodes[node_id];

        // Copy known partial symbols
        for (int i = 0; i < nd.n_partial; i++)
            x_out[Nt - 1 - i] = nd.partial_symbols[i];

        // Greedily fill remaining layers
        for (int layer = nd.layer - 1; layer >= 0; layer--) {
            double best_dist = 1e30;
            cx best_sym(0, 0);
            for (int s = 0; s < M; s++) {
                cx sym = constellation[s];
                // Quick distance: just squared magnitude of interference
                cx residual(0, 0);
                residual = y_tilde.data[layer] - R.data[layer][layer] * sym;
                double dist = std::norm(residual);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_sym = sym;
                }
            }
            x_out[layer] = best_sym;
        }
    }

    // Nearest constellation point slicing
    cx slice(cx val) const {
        double best_d = 1e30;
        cx best_s(0, 0);
        for (int s = 0; s < M; s++) {
            double d = std::norm(val - constellation[s]);
            if (d < best_d) { best_d = d; best_s = constellation[s]; }
        }
        return best_s;
    }

    // Main detection entry point
    double detect(const cx* H_flat, const cx* y_flat, const cx* x_true,
                  double noise_var,
                  const int* prog_down, int down_len,
                  const int* prog_up, int up_len,
                  const int* prog_belief, int belief_len,
                  const int* prog_halt, int halt_len,
                  double* flops_out) {
        preprocess(H_flat, y_flat);

        SearchTree tree(max_nodes);
        int root_id = tree.create_root(Nt);

        // Expand root
        expand(tree, root_id);

        // Initial BP cycle
        full_bp_cycle(tree, prog_down, down_len, prog_up, up_len,
                      prog_belief, belief_len, prog_halt, halt_len);

        // Priority queue (min-score)
        auto rebuild_pq = [&](std::vector<PQEntry>& pq) {
            pq.clear();
            for (int i = 0; i < tree.n_nodes; i++) {
                TreeNode& nd = tree.nodes[i];
                if (nd.id == tree.root_id) continue;
                if (nd.is_expanded) continue;
                pq.push_back({nd.score, nd.id, nd.queue_version});
            }
            std::sort(pq.begin(), pq.end(), [](const PQEntry& a, const PQEntry& b) {
                return a.score < b.score;
            });
        };

        std::vector<PQEntry> pq;
        rebuild_pq(pq);

        int best_partial_id = -1;
        double best_partial_score = 1e30;

        while (!pq.empty() && tree.n_nodes < max_nodes) {
            PQEntry top = pq.front();
            pq.erase(pq.begin());

            TreeNode& nd = tree.nodes[top.node_id];
            if (nd.is_expanded) continue;

            // Check if this is a leaf (layer == 0) → complete solution
            if (nd.layer == 0) {
                // Count symbol errors
                int errors = 0;
                for (int i = 0; i < nd.n_partial && i < Nt; i++) {
                    if (nd.partial_symbols[i] != x_true[Nt - 1 - i])
                        errors++;
                }
                if (flops_out) *flops_out = (double)tree.n_nodes;
                return (double)errors / Nt;
            }

            // Track best partial
            if (nd.score < best_partial_score) {
                best_partial_score = nd.score;
                best_partial_id = nd.id;
            }

            // Expand and re-run BP
            expand(tree, top.node_id);
            full_bp_cycle(tree, prog_down, down_len, prog_up, up_len,
                          prog_belief, belief_len, prog_halt, halt_len);
            rebuild_pq(pq);
        }

        // No leaf found — greedy completion from best partial
        cx x_hat[MAX_DIM];
        std::memset(x_hat, 0, sizeof(x_hat));

        if (best_partial_id >= 0) {
            complete_greedy(tree, best_partial_id, x_hat);
        } else {
            // LMMSE fallback
            for (int i = 0; i < Nt; i++)
                x_hat[i] = slice(y_tilde.data[i] / R.data[i][i]);
        }

        int errors = 0;
        for (int i = 0; i < Nt; i++) {
            if (x_hat[i] != x_true[i]) errors++;
        }
        if (flops_out) *flops_out = (double)tree.n_nodes;
        return (double)errors / Nt;
    }

    // LMMSE baseline
    double lmmse_detect(const cx* H_flat, const cx* y_flat,
                        const cx* x_true, double noise_var) {
        CxMat H(Nr, Nt);
        for (int i = 0; i < Nr; i++)
            for (int j = 0; j < Nt; j++)
                H.data[i][j] = H_flat[i * Nt + j];

        // W = (H^H H + sigma^2 I)^{-1} H^H
        cx HhH[MAX_DIM][MAX_DIM];
        for (int i = 0; i < Nt; i++)
            for (int j = 0; j < Nt; j++) {
                cx sum(0,0);
                for (int k = 0; k < Nr; k++)
                    sum += std::conj(H.data[k][i]) * H.data[k][j];
                HhH[i][j] = sum;
            }
        for (int i = 0; i < Nt; i++)
            HhH[i][i] += cx(noise_var, 0);

        cx Hhy[MAX_DIM];
        for (int i = 0; i < Nt; i++) {
            cx sum(0,0);
            for (int k = 0; k < Nr; k++)
                sum += std::conj(H.data[k][i]) * y_flat[k];
            Hhy[i] = sum;
        }

        cx x_hat[MAX_DIM];
        if (!solve_linear(HhH, Hhy, x_hat, Nt)) {
            return 1.0;  // fallback: all errors
        }

        int errors = 0;
        for (int i = 0; i < Nt; i++) {
            cx sliced = slice(x_hat[i]);
            if (sliced != x_true[i]) errors++;
        }
        return (double)errors / Nt;
    }
};

// ============================================================================
// Context for the DLL API
// ============================================================================

struct BPIRContext {
    int Nt, Nr, M;
    cx constellation[256];
    int max_nodes;
    int max_bp_iters;
};

// ============================================================================
// Exported C API
// ============================================================================

// constellation_interleaved: [re0, im0, re1, im1, ...]
IR_EXPORT void* bp_ir_create(int Nt, int Nr, int M,
                              const double* constellation_interleaved,
                              int max_nodes, int max_bp_iters) {
    auto* ctx = new BPIRContext();
    ctx->Nt = Nt;
    ctx->Nr = Nr;
    ctx->M = M;
    ctx->max_nodes = max_nodes;
    ctx->max_bp_iters = max_bp_iters;
    for (int i = 0; i < M; i++)
        ctx->constellation[i] = cx(constellation_interleaved[2*i],
                                    constellation_interleaved[2*i+1]);
    return ctx;
}

IR_EXPORT void bp_ir_destroy(void* handle) {
    delete static_cast<BPIRContext*>(handle);
}

// Evaluate one sample with 4 IR programs
IR_EXPORT double bp_ir_eval_one(void* handle,
                                 const int* prog_down, int down_len,
                                 const int* prog_up, int up_len,
                                 const int* prog_belief, int belief_len,
                                 const int* prog_halt, int halt_len,
                                 const double* H_interleaved,
                                 const double* y_interleaved,
                                 const double* x_true_interleaved,
                                 double noise_var,
                                 double* flops_out) {
    auto* ctx = static_cast<BPIRContext*>(handle);

    // De-interleave complex arrays
    cx H[MAX_DIM * MAX_DIM];
    cx y[MAX_DIM];
    cx x_true[MAX_DIM];
    for (int i = 0; i < ctx->Nr * ctx->Nt; i++)
        H[i] = cx(H_interleaved[2*i], H_interleaved[2*i+1]);
    for (int i = 0; i < ctx->Nr; i++)
        y[i] = cx(y_interleaved[2*i], y_interleaved[2*i+1]);
    for (int i = 0; i < ctx->Nt; i++)
        x_true[i] = cx(x_true_interleaved[2*i], x_true_interleaved[2*i+1]);

    BPIRDecoder dec(ctx->Nt, ctx->Nr, ctx->M, ctx->constellation,
                    ctx->max_nodes, ctx->max_bp_iters);
    return dec.detect(H, y, x_true, noise_var,
                      prog_down, down_len, prog_up, up_len,
                      prog_belief, belief_len, prog_halt, halt_len,
                      flops_out);
}

// Evaluate dataset (OpenMP parallel)
IR_EXPORT double bp_ir_eval_dataset(void* handle,
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
                                     int* total_faults_out) {
    auto* ctx = static_cast<BPIRContext*>(handle);
    int Nt = ctx->Nt, Nr = ctx->Nr;
    int H_stride = 2 * Nr * Nt;
    int y_stride = 2 * Nr;
    int x_stride = 2 * Nt;

    double total_ber = 0;
    double total_flops = 0;
    int faults = 0;

    #pragma omp parallel for schedule(dynamic, 4) reduction(+:total_ber, total_flops, faults)
    for (int s = 0; s < n_samples; s++) {
        cx H[MAX_DIM * MAX_DIM];
        cx y_vec[MAX_DIM];
        cx x_true[MAX_DIM];
        for (int i = 0; i < Nr * Nt; i++)
            H[i] = cx(H_all[s * H_stride + 2*i], H_all[s * H_stride + 2*i+1]);
        for (int i = 0; i < Nr; i++)
            y_vec[i] = cx(y_all[s * y_stride + 2*i], y_all[s * y_stride + 2*i+1]);
        for (int i = 0; i < Nt; i++)
            x_true[i] = cx(x_true_all[s * x_stride + 2*i], x_true_all[s * x_stride + 2*i+1]);

        BPIRDecoder dec(Nt, Nr, ctx->M, ctx->constellation,
                        ctx->max_nodes, ctx->max_bp_iters);
        double flops = 0;
        double ber = dec.detect(H, y_vec, x_true, noise_vars[s],
                                prog_down, down_len, prog_up, up_len,
                                prog_belief, belief_len, prog_halt, halt_len,
                                &flops);
        total_ber += ber;
        total_flops += flops;
    }

    double avg_ber = total_ber / std::max(n_samples, 1);
    if (avg_flops_out) *avg_flops_out = total_flops / std::max(n_samples, 1);
    if (total_faults_out) *total_faults_out = faults;
    return avg_ber;
}

// Evaluate single pure IR expression (for testing)
IR_EXPORT double ir_eval_expr(const int* prog, int prog_len,
                               const double* args, int n_args) {
    return ir_eval(prog, prog_len, args, n_args);
}

// LMMSE baseline
IR_EXPORT double bp_ir_lmmse(void* handle,
                              int n_samples,
                              const double* H_all,
                              const double* y_all,
                              const double* x_true_all,
                              const double* noise_vars) {
    auto* ctx = static_cast<BPIRContext*>(handle);
    int Nt = ctx->Nt, Nr = ctx->Nr;
    int H_stride = 2 * Nr * Nt;
    int y_stride = 2 * Nr;
    int x_stride = 2 * Nt;

    double total_ber = 0;

    #pragma omp parallel for schedule(dynamic, 4) reduction(+:total_ber)
    for (int s = 0; s < n_samples; s++) {
        cx H[MAX_DIM * MAX_DIM];
        cx y_vec[MAX_DIM];
        cx x_true[MAX_DIM];
        for (int i = 0; i < Nr * Nt; i++)
            H[i] = cx(H_all[s * H_stride + 2*i], H_all[s * H_stride + 2*i+1]);
        for (int i = 0; i < Nr; i++)
            y_vec[i] = cx(y_all[s * y_stride + 2*i], y_all[s * y_stride + 2*i+1]);
        for (int i = 0; i < Nt; i++)
            x_true[i] = cx(x_true_all[s * x_stride + 2*i], x_true_all[s * x_stride + 2*i+1]);

        BPIRDecoder dec(Nt, Nr, ctx->M, ctx->constellation,
                        ctx->max_nodes, ctx->max_bp_iters);
        total_ber += dec.lmmse_detect(H, y_vec, x_true, noise_vars[s]);
    }

    return total_ber / std::max(n_samples, 1);
}
