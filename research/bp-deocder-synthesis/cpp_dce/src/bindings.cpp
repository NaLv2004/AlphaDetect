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
#include "dce.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using namespace pushgp_cpp;
using pushgp_cpp_dce::LiftedParityC;
using pushgp_cpp_dce::decode_bp_cpp;
using pushgp_cpp_dce::behavioral_reduce_bp_cpp;
using pushgp_cpp_dce::Position;
using pushgp_cpp_dce::PosStep;
using pushgp_cpp_dce::ReduceStats;

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

static py::list program_to_dict(const Program& p);
static py::dict instruction_to_dict(const Instruction& ins) {
    py::dict d;
    d["name"] = op_to_name(ins.op);
    if (ins.code_block)  d["code_block"]  = program_to_dict(*ins.code_block);
    if (ins.code_block2) d["code_block2"] = program_to_dict(*ins.code_block2);
    return d;
}
static py::list program_to_dict(const Program& p) {
    py::list out;
    for (const auto& ins : p) out.append(instruction_to_dict(ins));
    return out;
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

    // Convert a Python list of np.ndarray to std::vector<std::vector<double>>.
    auto rx_list_to_cpp = [](py::list rx_list) {
        std::vector<std::vector<double>> out;
        out.reserve(rx_list.size());
        for (auto&& item : rx_list) {
            py::array_t<double, py::array::c_style | py::array::forcecast> a =
                item.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            auto buf = a.unchecked<1>();
            std::vector<double> v(buf.size());
            for (py::ssize_t i = 0; i < buf.size(); ++i) v[i] = buf(i);
            out.push_back(std::move(v));
        }
        return out;
    };

    m.def("reduce_bp", [rx_list_to_cpp](
        py::list prog_dict,
        const std::string& side,
        py::list peer_dict,
        const ParityHandle& parH,
        py::list rx_list,
        py::array_t<double, py::array::c_style | py::array::forcecast> evo,
        int max_iter,
        int max_passes,
        int max_decode_evals,
        int decimals) {
        if (side != "v2c" && side != "c2v") {
            throw std::invalid_argument("side must be 'v2c' or 'c2v'");
        }
        Program prog = dict_to_program(prog_dict);
        Program peer = dict_to_program(peer_dict);
        auto rxs = rx_list_to_cpp(rx_list);
        auto evo_arr = numpy_to_evo(evo);
        ReduceStats stats;
        Program reduced;
        {
            py::gil_scoped_release release;
            reduced = behavioral_reduce_bp_cpp(
                prog, /*side_is_v2c=*/side == "v2c", peer, evo_arr, *parH.par,
                rxs, max_iter, max_passes, max_decode_evals, decimals, &stats);
        }
        py::list out_prog = program_to_dict(reduced);
        py::dict st;
        st["passes"] = stats.passes;
        st["fp_evals"] = stats.fp_evals;
        st["size_before"] = stats.size_before;
        st["size_after"] = stats.size_after;
        py::list rp;
        for (const auto& pos : stats.removed_positions) {
            py::list one;
            for (const auto& s : pos) {
                one.append(py::make_tuple(s.idx, s.desc));
            }
            rp.append(std::move(one));
        }
        st["removed_positions"] = rp;
        return py::make_tuple(out_prog, st);
    },
        py::arg("prog"), py::arg("side"), py::arg("peer_prog"),
        py::arg("parity_handle"), py::arg("rx_llrs"), py::arg("evo"),
        py::arg("max_iter") = 8, py::arg("max_passes") = 800,
        py::arg("max_decode_evals") = -1, py::arg("decimals") = 6,
        "C++ port of pushgp.dce.behavioral_reduce_bp (single-threaded).\n"
        "Returns (reduced_prog_dict, stats_dict).");

    // Local helper for tests: program_to_dict from a list-of-dict prog
    // (round-trips via dict_to_program for byte-equality comparison).
    m.def("program_roundtrip", [](py::list prog_dict) {
        Program p = dict_to_program(prog_dict);
        return program_to_dict(p);
    });

    // ============================================================
    // reduce_bp_batch: multi-threaded version of reduce_bp.
    // Jobs is a list of dicts:
    //   {"prog": list, "side": "v2c"|"c2v", "peer_prog": list,
    //    "evo": np.ndarray}
    // All jobs share parity_handle + rx_llrs + max_iter + ...  This is
    // the typical DCE post-seeding pattern.
    // ============================================================
    m.def("reduce_bp_batch", [rx_list_to_cpp](
        py::list jobs,
        const ParityHandle& parH,
        py::list rx_list,
        int max_iter,
        int max_passes,
        int max_decode_evals,
        int decimals,
        int threads,
        py::object progress_cb) {
        if (threads <= 0) threads = 1;
        const int n_jobs = static_cast<int>(jobs.size());

        // ---- Convert all inputs UP FRONT (with GIL).
        struct Job {
            Program prog;
            Program peer;
            bool side_is_v2c;
            std::array<double, N_EVO_CONSTS> evo;
        };
        std::vector<Job> in_jobs(n_jobs);
        for (int i = 0; i < n_jobs; ++i) {
            py::dict d = jobs[i].cast<py::dict>();
            std::string side = d["side"].cast<std::string>();
            if (side != "v2c" && side != "c2v") {
                throw std::invalid_argument("job side must be v2c or c2v");
            }
            in_jobs[i].side_is_v2c = (side == "v2c");
            in_jobs[i].prog = dict_to_program(d["prog"].cast<py::list>());
            in_jobs[i].peer = dict_to_program(d["peer_prog"].cast<py::list>());
            in_jobs[i].evo  = numpy_to_evo(
                d["evo"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
        }
        auto rxs = rx_list_to_cpp(rx_list);

        // ---- Output buffers.
        std::vector<Program>     out_progs(n_jobs);
        std::vector<ReduceStats> out_stats(n_jobs);
        std::atomic<int>         next{0};
        std::atomic<int>         done_count{0};
        std::atomic<bool>        done{false};

        auto t0 = std::chrono::steady_clock::now();
        auto worker_fn = [&]() {
            while (true) {
                int idx = next.fetch_add(1, std::memory_order_relaxed);
                if (idx >= n_jobs) break;
                const Job& J = in_jobs[idx];
                ReduceStats st;
                out_progs[idx] = behavioral_reduce_bp_cpp(
                    J.prog, J.side_is_v2c, J.peer, J.evo, *parH.par,
                    rxs, max_iter, max_passes, max_decode_evals, decimals, &st);
                out_stats[idx] = std::move(st);
                done_count.fetch_add(1, std::memory_order_relaxed);
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(threads);
        {
            py::gil_scoped_release release;
            for (int t = 0; t < threads; ++t) pool.emplace_back(worker_fn);
            // Progress polling (main thread keeps GIL released except
            // when invoking the callback).
            if (!progress_cb.is_none()) {
                while (done_count.load() < n_jobs) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    int dc = done_count.load();
                    double elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
                    {
                        py::gil_scoped_acquire g;
                        try { progress_cb(dc, n_jobs, elapsed); } catch (...) {}
                    }
                    if (dc >= n_jobs) break;
                }
            }
            for (auto& th : pool) th.join();
        }

        // ---- Convert outputs back (with GIL).
        py::list out;
        for (int i = 0; i < n_jobs; ++i) {
            py::dict d;
            d["prog"] = program_to_dict(out_progs[i]);
            py::dict st;
            st["passes"] = out_stats[i].passes;
            st["fp_evals"] = out_stats[i].fp_evals;
            st["size_before"] = out_stats[i].size_before;
            st["size_after"] = out_stats[i].size_after;
            py::list rp;
            for (const auto& pos : out_stats[i].removed_positions) {
                py::list one;
                for (const auto& s : pos) one.append(py::make_tuple(s.idx, s.desc));
                rp.append(std::move(one));
            }
            st["removed_positions"] = rp;
            d["stats"] = st;
            out.append(std::move(d));
        }
        return out;
    },
        py::arg("jobs"), py::arg("parity_handle"), py::arg("rx_llrs"),
        py::arg("max_iter") = 8, py::arg("max_passes") = 800,
        py::arg("max_decode_evals") = -1, py::arg("decimals") = 6,
        py::arg("threads") = 1, py::arg("progress_cb") = py::none(),
        "Multi-threaded reduce_bp.  Each worker thread runs the full\n"
        "behavioral_reduce_bp_cpp on one job at a time using std::thread.\n"
        "Returns a list parallel to `jobs`, each item {'prog':..,'stats':..}.");

    // We also need program_to_dict static helper to be accessible
    // from python so tests can compare structurally.  Already done via
    // reduce_bp output.
}
