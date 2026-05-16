// bindings.cpp — pybind11 module `pushgp_cpp_dce`.
//
// Exposes ONLY new functionality.  Programs are still built via
// pushgp_cpp_seeder.build_program; we accept Program dict-lists here
// and convert them locally (it is cheap and keeps the modules
// independent at the Python ABI level).
#include "../../cpp_seeder/src/common.hpp"
#include "../../cpp_seeder/src/instruction.hpp"
#include "../../cpp_seeder/src/opcodes.hpp"
#include "../../cpp_seeder/src/opcodes_table.hpp"   // include exactly once
#include "../../cpp_seeder/src/validator.hpp"
#include "../../cpp_seeder/src/vm.hpp"
#include "../../cpp_seeder/src/vm_state.hpp"

#include "parity_struct.hpp"
#include "bp_decoder.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pushgp_cpp;
using pushgp_cpp_dce::LiftedParityC;
using pushgp_cpp_dce::decode_bp_cpp;

// ---- Program conversion (mirrors cpp_seeder/src/bindings.cpp) ----
static Program dict_to_program(const py::list& lst);
static Instruction dict_to_instruction(const py::dict& d) {
    std::string name = d["name"].cast<std::string>();
    Op op;
    if (!name_to_op(name, op)) {
        throw std::invalid_argument("unknown opcode: " + name);
    }
    Instruction ins(op);
    if (d.contains("code_block")) {
        ins.code_block = std::make_unique<Program>(
            dict_to_program(d["code_block"].cast<py::list>()));
    }
    if (d.contains("code_block2")) {
        ins.code_block2 = std::make_unique<Program>(
            dict_to_program(d["code_block2"].cast<py::list>()));
    }
    return ins;
}
static Program dict_to_program(const py::list& lst) {
    Program p;
    p.reserve(lst.size());
    for (auto&& item : lst) {
        p.push_back(dict_to_instruction(item.cast<py::dict>()));
    }
    return p;
}

static std::array<double, N_EVO_CONSTS> numpy_to_evo(
    py::array_t<double, py::array::c_style | py::array::forcecast> a)
{
    std::array<double, N_EVO_CONSTS> evo{{1,1,1,1,1,1,1,1}};
    auto buf = a.unchecked<1>();
    py::ssize_t n = std::min<py::ssize_t>(buf.size(), N_EVO_CONSTS);
    for (py::ssize_t i = 0; i < n; ++i) evo[static_cast<size_t>(i)] = buf(i);
    return evo;
}

// Parity handle wrapper.
struct ParityHandle {
    std::shared_ptr<LiftedParityC> par;
};

PYBIND11_MODULE(pushgp_cpp_dce, m) {
    m.doc() = "C++ acceleration for DCE behavioral_reduce_bp (Step 1+).";

    py::class_<ParityHandle>(m, "ParityHandle")
        .def_property_readonly("N", [](const ParityHandle& h) { return h.par->N; })
        .def_property_readonly("M", [](const ParityHandle& h) { return h.par->M; })
        .def_property_readonly("zc", [](const ParityHandle& h) { return h.par->zc; })
        // Return list-of-list-of-(c,p) tuples for inspection.  Used by
        // the Step-1 byte-equality test against `_build_vn_to_cn`.
        .def("vn_to_cn", [](const ParityHandle& h) {
            py::list out;
            for (const auto& row : h.par->vn_to_cn) {
                py::list inner;
                for (const auto& cp : row) {
                    inner.append(py::make_tuple(cp.first, cp.second));
                }
                out.append(std::move(inner));
            }
            return out;
        })
        .def("cn_to_vn", [](const ParityHandle& h) {
            py::list out;
            for (const auto& row : h.par->cn_to_vn) {
                py::list inner;
                for (int v : row) inner.append(v);
                out.append(std::move(inner));
            }
            return out;
        });

    m.def("build_parity_handle", [](py::object par_py) {
        ParityHandle h;
        h.par = std::make_shared<LiftedParityC>(
            LiftedParityC::from_python(par_py));
        return h;
    }, py::arg("par"),
       "Build a ParityHandle from a Python ldpc_5g.LiftedParity instance.");

    m.def("decode_bp", [](py::array_t<double, py::array::c_style | py::array::forcecast> rx_llr,
                          const ParityHandle& parH,
                          py::list prog_v2c_dict,
                          py::list prog_c2v_dict,
                          py::array_t<double, py::array::c_style | py::array::forcecast> evo,
                          int max_iter,
                          double offset,
                          double code_rate) {
        Program prog_v2c = dict_to_program(prog_v2c_dict);
        Program prog_c2v = dict_to_program(prog_c2v_dict);
        auto evo_arr = numpy_to_evo(evo);
        auto buf = rx_llr.unchecked<1>();
        std::vector<double> rx(buf.size());
        for (py::ssize_t i = 0; i < buf.size(); ++i) rx[i] = buf(i);
        int iters_run = 0;
        std::vector<double> post = decode_bp_cpp(
            rx, *parH.par, prog_v2c, prog_c2v, evo_arr,
            max_iter, offset, code_rate, &iters_run);
        // Return (post_llr, iters_run)
        py::array_t<double> out(static_cast<py::ssize_t>(post.size()));
        auto obuf = out.mutable_unchecked<1>();
        for (size_t i = 0; i < post.size(); ++i) obuf(i) = post[i];
        return py::make_tuple(out, iters_run);
    },
        py::arg("rx_llr"), py::arg("parity_handle"),
        py::arg("prog_v2c"), py::arg("prog_c2v"),
        py::arg("evo"), py::arg("max_iter"),
        py::arg("offset")    = 0.25,
        py::arg("code_rate") = 0.5,
        "Run the C++ BP decoder; returns (post_llr, iters_run).");
}
