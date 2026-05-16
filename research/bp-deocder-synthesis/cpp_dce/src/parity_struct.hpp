// parity_struct.hpp — C++ mirror of ldpc_5g.LiftedParity.
//
// Only the data needed by the BP decoder is mirrored.  We do NOT
// reimplement build_parity in C++ — the Python side calls build_parity
// once and ships the resulting structure down via from_python().
//
// Storage layout matches `ldpc_5g.LiftedParity` exactly:
//   cn_to_vn[c] = sorted vn indices touched by check row c
//   vn_to_cn[v] = list of (cn_index, position_in_cn_to_vn[c])
//                 built by mirroring `_build_vn_to_cn` byte-for-byte
//                 (insertion order matters because BP iterates over
//                  edges_in in this exact order).
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>

namespace pushgp_cpp_dce {

namespace py = pybind11;

struct LiftedParityC {
    int N = 0;   // cols
    int M = 0;   // rows
    int zc = 0;
    int bgn = 0;
    int set_idx = 0;

    // cn_to_vn[c] = list of VN indices for check c
    std::vector<std::vector<int>> cn_to_vn;
    // vn_to_cn[v] = list of (c, p) such that cn_to_vn[c][p] == v
    std::vector<std::vector<std::pair<int, int>>> vn_to_cn;

    // Build from a Python LiftedParity object.  Reads .cols .rows .zc
    // .bgn .set_idx .cn_to_vn and builds vn_to_cn by the SAME insertion
    // order as `ldpc_5g._build_vn_to_cn`.
    static LiftedParityC from_python(py::object par) {
        LiftedParityC out;
        out.N = par.attr("cols").cast<int>();
        out.M = par.attr("rows").cast<int>();
        out.zc = par.attr("zc").cast<int>();
        out.bgn = par.attr("bgn").cast<int>();
        out.set_idx = par.attr("set_idx").cast<int>();

        py::list cn_list = par.attr("cn_to_vn").cast<py::list>();
        if ((int)cn_list.size() != out.M) {
            throw std::runtime_error("parity_struct: cn_to_vn length != rows");
        }
        out.cn_to_vn.resize(out.M);
        for (int c = 0; c < out.M; ++c) {
            py::array arr = cn_list[c].cast<py::array>();
            auto buf = arr.request();
            if (buf.ndim != 1) throw std::runtime_error("cn_to_vn[c] not 1-D");
            const int n = static_cast<int>(buf.shape[0]);
            out.cn_to_vn[c].resize(n);
            // Python stores np.int64; cast element by element.
            if (buf.itemsize == 8) {
                const int64_t* p = static_cast<const int64_t*>(buf.ptr);
                for (int i = 0; i < n; ++i) out.cn_to_vn[c][i] = static_cast<int>(p[i]);
            } else if (buf.itemsize == 4) {
                const int32_t* p = static_cast<const int32_t*>(buf.ptr);
                for (int i = 0; i < n; ++i) out.cn_to_vn[c][i] = static_cast<int>(p[i]);
            } else {
                throw std::runtime_error("cn_to_vn[c] unexpected dtype");
            }
        }

        // _build_vn_to_cn: for c in range(M): for pos, v in enumerate(cn_to_vn[c])
        //                     vn_to_cn[v].append((c, pos))
        out.vn_to_cn.assign(out.N, {});
        for (int c = 0; c < out.M; ++c) {
            const auto& vns = out.cn_to_vn[c];
            for (int pos = 0; pos < (int)vns.size(); ++pos) {
                int v = vns[pos];
                if (v < 0 || v >= out.N) {
                    throw std::runtime_error("vn index out of range");
                }
                out.vn_to_cn[v].emplace_back(c, pos);
            }
        }
        return out;
    }
};

}  // namespace pushgp_cpp_dce
