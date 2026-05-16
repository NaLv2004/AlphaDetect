// bp_decoder.hpp — C++ port of ldpc_5g.decode_bp.
//
// Strict design contract: byte-for-byte equivalent to the Python BP
// loop in ldpc_5g.decode_bp (rounded to the DCE comparison precision).
// All loops, accumulator orders, and VM-seed sequences must match the
// Python reference; reuses cpp_seeder's VM / Program / configure_vm_*
// helpers — no new VM, no new program type.
//
// Reference Python locations:
//   ldpc_5g.py:821-919   decode_bp
//   ldpc_5g.py:775-782   _build_vn_to_cn
//   pushgp_ldpc/adapter.py:_seed_v2c / _seed_c2v
//
// max_iter handling: the Python adapter passes max_iter via ctx and
// pushes it as the 4th int onto the stack.  cpp_seeder's
// configure_vm_v2c hard-codes ctx_max_iter=25 (only used by the static
// behavioral panel which always uses max_iter=25); for BP we override
// ctx_max_iter BEFORE seed_*_stacks so the int pushed on the stack
// matches Python.
#pragma once

#include "../../cpp_seeder/src/common.hpp"
#include "../../cpp_seeder/src/instruction.hpp"
#include "../../cpp_seeder/src/validator.hpp"   // configure_vm_v2c/c2v, seed_*, run_*
#include "../../cpp_seeder/src/vm.hpp"

#include "parity_struct.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace pushgp_cpp_dce {

using pushgp_cpp::N_EVO_CONSTS;
using pushgp_cpp::FVec;
using pushgp_cpp::Program;
using pushgp_cpp::VM;

// One-shot v2c invocation matching Python adapter._seed_v2c + vm.run.
// Returns finite output, or 0.0 on fault / non-finite (mirrors Python's
// `except: msg = 0.0` and `not np.isfinite(msg)` branches).
inline double bp_call_v2c(VM& vm, const Program& prog,
                          double L_v, const FVec& incoming, int deg,
                          int iter_idx, int max_iter,
                          const std::array<double, N_EVO_CONSTS>& evo) {
    pushgp_cpp::configure_vm_v2c(vm, incoming, L_v, deg, iter_idx, evo);
    vm.state.ctx_max_iter = max_iter;  // override hard-coded 25
    double out = 0.0;
    if (!pushgp_cpp::run_v2c(vm, prog, out)) return 0.0;
    if (!std::isfinite(out)) return 0.0;
    return out;
}

inline double bp_call_c2v(VM& vm, const Program& prog,
                          const FVec& incoming, int deg,
                          int iter_idx, int max_iter,
                          const std::array<double, N_EVO_CONSTS>& evo) {
    pushgp_cpp::configure_vm_c2v(vm, incoming, deg, iter_idx, evo);
    vm.state.ctx_max_iter = max_iter;
    double out = 0.0;
    if (!pushgp_cpp::run_c2v(vm, prog, out)) return 0.0;
    if (!std::isfinite(out)) return 0.0;
    return out;
}

// Strict port of ldpc_5g.decode_bp.  rx_llr length must equal par.N.
//
// Returns post_llr of length N.  iters_run (optional) is filled with the
// number of BP iterations actually performed (1..max_iter).
inline std::vector<double> decode_bp_cpp(
    const std::vector<double>& rx_llr_in,
    const LiftedParityC& par,
    const Program& prog_v2c,
    const Program& prog_c2v,
    const std::array<double, N_EVO_CONSTS>& evo,
    int max_iter,
    double /*offset*/      = 0.25,
    double /*code_rate*/   = 0.5,
    int* iters_run_out     = nullptr)
{
    const int N = par.N;
    const int M = par.M;
    if (static_cast<int>(rx_llr_in.size()) != N) {
        throw std::runtime_error("decode_bp_cpp: rx_llr size != par.N");
    }

    // llr_c2v[c], llr_v2c[c] each of length cn_to_vn[c].size, zero-init.
    std::vector<std::vector<double>> llr_c2v(M);
    std::vector<std::vector<double>> llr_v2c(M);
    for (int c = 0; c < M; ++c) {
        const size_t dc = par.cn_to_vn[c].size();
        llr_c2v[c].assign(dc, 0.0);
        llr_v2c[c].assign(dc, 0.0);
    }

    // Working buffers reused across the loop to avoid allocator churn.
    FVec incoming_buf;   incoming_buf.reserve(64);
    FVec c2v_vec;        c2v_vec.reserve(64);

    // Two VMs (one per side), reused across edges.  configure_vm_* calls
    // reset_stacks() so the VMs are always at a clean state.
    VM vm_v2c, vm_c2v;

    std::vector<double> rx_llr = rx_llr_in;  // local copy (Python does astype(copy=False) but we want safety)
    std::vector<double> llr_post = rx_llr;

    int iters_run = 0;

    for (int it = 0; it < max_iter; ++it) {
        iters_run = it + 1;

        // ---- V2C update ----
        for (int v = 0; v < N; ++v) {
            const auto& edges_in = par.vn_to_cn[v];
            const int dv = static_cast<int>(edges_in.size());
            if (dv == 0) continue;
            c2v_vec.resize(dv);
            for (int ei = 0; ei < dv; ++ei) {
                int c = edges_in[ei].first;
                int p = edges_in[ei].second;
                c2v_vec[ei] = llr_c2v[c][p];
            }
            const double L_v = rx_llr[v];
            for (int ei = 0; ei < dv; ++ei) {
                // incoming = np.concatenate((c2v_vec[:ei], c2v_vec[ei+1:]))
                if (dv == 1) {
                    incoming_buf.clear();
                } else {
                    incoming_buf.resize(dv - 1);
                    int w = 0;
                    for (int j = 0; j < dv; ++j) {
                        if (j == ei) continue;
                        incoming_buf[w++] = c2v_vec[j];
                    }
                }
                double msg = bp_call_v2c(vm_v2c, prog_v2c, L_v,
                                          incoming_buf, dv, it, max_iter, evo);
                int c = edges_in[ei].first;
                int p = edges_in[ei].second;
                llr_v2c[c][p] = msg;
            }
        }

        // ---- C2V update ----
        for (int c = 0; c < M; ++c) {
            auto& v2c_vec = llr_v2c[c];
            const int dc = static_cast<int>(v2c_vec.size());
            if (dc == 0) continue;
            std::vector<double> new_c2v(dc, 0.0);
            for (int ei = 0; ei < dc; ++ei) {
                if (dc == 1) {
                    incoming_buf.clear();
                } else {
                    incoming_buf.resize(dc - 1);
                    int w = 0;
                    for (int j = 0; j < dc; ++j) {
                        if (j == ei) continue;
                        incoming_buf[w++] = v2c_vec[j];
                    }
                }
                double msg = bp_call_c2v(vm_c2v, prog_c2v,
                                          incoming_buf, dc, it, max_iter, evo);
                new_c2v[ei] = msg;
            }
            llr_c2v[c] = std::move(new_c2v);
        }

        // ---- VN posterior ----
        llr_post = rx_llr;
        for (int v = 0; v < N; ++v) {
            double s = 0.0;
            for (const auto& cp : par.vn_to_cn[v]) {
                s += llr_c2v[cp.first][cp.second];
            }
            llr_post[v] += s;
        }

        // ---- Syndrome check (early stop) ----
        // bits[v] = 1 if llr_post[v] < 0 else 0
        // ok = all parity checks pass
        bool ok = true;
        for (int c = 0; c < M && ok; ++c) {
            const auto& vn = par.cn_to_vn[c];
            if (vn.empty()) continue;
            int sum = 0;
            for (int v : vn) sum += (llr_post[v] < 0.0) ? 1 : 0;
            if (sum & 1) { ok = false; }
        }
        if (ok) break;
    }

    if (iters_run_out) *iters_run_out = iters_run;
    return llr_post;
}

}  // namespace pushgp_cpp_dce
