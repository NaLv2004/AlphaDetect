// bindings.cpp — pybind11 module exposing the C++ VM/validator/RPG/seeder.
//
// Public Python API:
//   build_program(prog_dict_list)              -> ProgramHandle
//   program_to_dict(ProgramHandle)             -> list[dict]   (round-trip check)
//   run_program(prog, side, evo, incoming, L_v, deg, iter_idx) -> Optional[float]
//   validate_with_panels(prog, side, panels, evo, deg)        -> (ok, reason)
//   validate_random      (prog, side, evo, deg, ncfg, nperm, seed) -> (ok, reason)
//   parallel_seed(side, n_target, max_attempts, threads,
//                 chunk_attempts, min_size, max_size, deg,
//                 num_configs, num_permutations, base_seed,
//                 progress_cb, seen_fingerprints, allowed_op_names)
//                                              -> (list[ProgramHandle], n_attempts, fingerprints)
//   allowed_op_names: list[str], empty = no filter; else intersection with
//                     side's base set (so c2v auto-drops Env_GetChannelLLR).
//   all_op_names() -> list[str]
//
// ProgramHandle is an opaque Python class wrapping shared_ptr<Program>.
#include "common.hpp"
#include "instruction.hpp"
#include "opcodes.hpp"
#include "opcodes_table.hpp"   // include exactly once
#include "rpg.hpp"
#include "validator.hpp"
#include "vm.hpp"
#include "vm_state.hpp"
#include "behav_panel.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using namespace pushgp_cpp;

// ---------------------------------------------------------------- conversions

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
    if (ins.code_block) d["code_block"] = program_to_dict(*ins.code_block);
    if (ins.code_block2) d["code_block2"] = program_to_dict(*ins.code_block2);
    return d;
}
static py::list program_to_dict(const Program& p) {
    py::list out;
    for (const auto& ins : p) out.append(instruction_to_dict(ins));
    return out;
}

// Convert a numpy array (1D float64) to FVec.
static FVec numpy_to_fvec(py::array_t<double, py::array::c_style | py::array::forcecast> a) {
    auto buf = a.unchecked<1>();
    FVec v(static_cast<size_t>(buf.size()));
    for (py::ssize_t i = 0; i < buf.size(); ++i) v[static_cast<size_t>(i)] = buf(i);
    return v;
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

// ------------ ProgramHandle: owning wrapper for use across calls ------------

struct ProgramHandle {
    std::shared_ptr<Program> prog;
};

// ---------- Panel conversion ----------
// Python panel format (one per config):
//   {"L_v": float, "incoming": np.ndarray[float64],
//    "perturb_indices": list[int], "permutations": list[np.ndarray]}
static std::vector<ConfigPanel> dict_to_panels(const py::list& py_panels) {
    std::vector<ConfigPanel> out;
    out.reserve(py_panels.size());
    for (auto&& item : py_panels) {
        py::dict d = item.cast<py::dict>();
        ConfigPanel p;
        if (d.contains("L_v")) p.L_v = d["L_v"].cast<double>();
        p.incoming = numpy_to_fvec(d["incoming"].cast<py::array_t<double>>());
        for (auto&& x : d["perturb_indices"].cast<py::list>()) {
            p.perturb_indices.push_back(x.cast<int>());
        }
        for (auto&& arr : d["permutations"].cast<py::list>()) {
            p.permutations.push_back(numpy_to_fvec(arr.cast<py::array_t<double>>()));
        }
        out.push_back(std::move(p));
    }
    return out;
}

// ============================================================ run_program

static py::object run_program_impl(
    const Program& prog,
    const std::string& side,
    py::array_t<double, py::array::c_style | py::array::forcecast> evo,
    py::array_t<double, py::array::c_style | py::array::forcecast> incoming,
    double L_v,
    int deg,
    int iter_idx)
{
    VM vm;
    auto evo_arr = numpy_to_evo(evo);
    FVec inc = numpy_to_fvec(incoming);
    if (side == "v2c") {
        configure_vm_v2c(vm, inc, L_v, deg, iter_idx, evo_arr);
        double out;
        if (run_v2c(vm, prog, out)) return py::float_(out);
        return py::none();
    } else if (side == "c2v") {
        configure_vm_c2v(vm, inc, deg, iter_idx, evo_arr);
        double out;
        if (run_c2v(vm, prog, out)) return py::float_(out);
        return py::none();
    } else {
        throw std::invalid_argument("side must be 'v2c' or 'c2v'");
    }
}

// ============================================================ validate

static py::tuple validate_with_panels_impl(
    const Program& prog, const std::string& side,
    py::list py_panels,
    py::array_t<double, py::array::c_style | py::array::forcecast> evo,
    int deg)
{
    auto evo_arr = numpy_to_evo(evo);
    auto panels = dict_to_panels(py_panels);
    ValidationResult r = (side == "v2c")
        ? validate_v2c_with_panels(prog, panels, evo_arr, deg)
        : validate_c2v_with_panels(prog, panels, evo_arr, deg);
    return py::make_tuple(r.ok, r.reason);
}

static py::tuple validate_random_impl(
    const Program& prog, const std::string& side,
    py::array_t<double, py::array::c_style | py::array::forcecast> evo,
    int deg, int num_configs, int num_permutations, uint64_t seed)
{
    auto evo_arr = numpy_to_evo(evo);
    ValidationResult r = (side == "v2c")
        ? validate_v2c_random(prog, evo_arr, deg, num_configs, num_permutations, seed)
        : validate_c2v_random(prog, evo_arr, deg, num_configs, num_permutations, seed);
    return py::make_tuple(r.ok, r.reason);
}

// ============================================================ parallel_seed

struct SeedResult {
    std::vector<std::shared_ptr<Program>> programs;
    std::vector<std::string>              fingerprints;  // parallel to programs
    int64_t total_attempts = 0;
};

static SeedResult parallel_seed_impl(
    const std::string& side,
    int n_target,
    int64_t max_attempts,
    int threads,
    int chunk_attempts,
    int min_size,
    int max_size,
    int deg,
    int num_configs,
    int num_permutations,
    uint64_t base_seed,
    py::object progress_cb,
    const std::vector<std::string>& seen_in,
    const std::vector<std::string>& allowed_op_names)
{
    if (threads <= 0) threads = 1;
    std::atomic<int64_t> attempts_total{0};
    std::atomic<int>     valid_count{0};

    std::vector<std::shared_ptr<Program>> valid;
    std::vector<std::string>              valid_fps;
    std::unordered_set<std::string>       seen(seen_in.begin(), seen_in.end());
    std::mutex mu_valid;

    auto t0 = std::chrono::steady_clock::now();

    if (side != "v2c" && side != "c2v") throw std::invalid_argument("side must be v2c/c2v");
    std::vector<Op> instr_set = (side == "v2c")
        ? RandomProgramGenerator::v2c_set()
        : RandomProgramGenerator::c2v_set();

    // ---- Apply op-filter whitelist (intersection with side's base set) ----
    // Empty allowed_op_names means "no filter".  Otherwise translate every
    // name via name_to_op (throwing on unknown), then intersect with the
    // side's base set (so c2v still excludes Env_GetChannelLLR even if
    // the user accidentally whitelists it).
    if (!allowed_op_names.empty()) {
        std::unordered_set<int> base_set;
        for (Op op : instr_set) base_set.insert(static_cast<int>(op));
        std::vector<Op> filtered;
        filtered.reserve(allowed_op_names.size());
        for (const auto& nm : allowed_op_names) {
            Op op;
            if (!name_to_op(nm, op)) {
                throw std::invalid_argument(
                    "parallel_seed: unknown opcode in allowed_op_names: " + nm);
            }
            if (base_set.count(static_cast<int>(op))) {
                filtered.push_back(op);
            }
            // Silently drop ops not in this side's base set (e.g. c2v + Env_GetChannelLLR).
        }
        if (filtered.empty()) {
            throw std::invalid_argument(
                "parallel_seed: allowed_op_names produces empty instruction set for side=" + side);
        }
        instr_set = std::move(filtered);
    }

    std::array<double, N_EVO_CONSTS> evo_default{{1,1,1,1,1,1,1,1}};

    std::atomic<uint64_t> seed_counter{base_seed};
    std::atomic<bool> done{false};

    // Worker -- validates, then computes behavioral fp; the shared
    // `seen` set is the source of truth: a candidate is only accepted
    // if its fingerprint hasn't been claimed yet.
    auto worker_fn = [&](int worker_id) {
        while (!done.load(std::memory_order_relaxed)) {
            int64_t attempts_so_far = attempts_total.load(std::memory_order_relaxed);
            if (attempts_so_far >= max_attempts) break;
            if (valid_count.load(std::memory_order_relaxed) >= n_target) break;

            uint64_t my_seed = seed_counter.fetch_add(1, std::memory_order_relaxed)
                             ^ (static_cast<uint64_t>(worker_id) << 17);
            RandomProgramGenerator rpg(my_seed);

            int my_chunk = chunk_attempts;
            int my_attempts = 0;
            // Collect (program, fingerprint) pairs locally; we lock
            // once at the end of the chunk to commit to the global set.
            std::vector<std::pair<std::shared_ptr<Program>, std::string>> my_valid;
            for (int i = 0; i < my_chunk; ++i) {
                if (valid_count.load(std::memory_order_relaxed) >= n_target) break;
                ++my_attempts;
                Program prog = rpg.random_program(instr_set, min_size, max_size);
                uint64_t sub_seed = (my_seed * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(i));
                ValidationResult r = (side == "v2c")
                    ? validate_v2c_random(prog, evo_default, deg, num_configs, num_permutations, sub_seed)
                    : validate_c2v_random(prog, evo_default, deg, num_configs, num_permutations, sub_seed);
                if (!r.ok) continue;
                std::string fp = compute_behav_fp(side, prog);
                my_valid.emplace_back(
                    std::make_shared<Program>(std::move(prog)), std::move(fp));
            }
            attempts_total.fetch_add(my_attempts, std::memory_order_relaxed);
            if (!my_valid.empty()) {
                std::lock_guard<std::mutex> lk(mu_valid);
                for (auto& pf : my_valid) {
                    if (static_cast<int>(valid.size()) >= n_target) {
                        done.store(true);
                        break;
                    }
                    if (seen.count(pf.second)) continue;       // dedup-as-validation
                    seen.insert(pf.second);
                    valid.push_back(std::move(pf.first));
                    valid_fps.push_back(std::move(pf.second));
                }
                valid_count.store(static_cast<int>(valid.size()), std::memory_order_relaxed);
                if (static_cast<int>(valid.size()) >= n_target) done.store(true);
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back(worker_fn, t);
    }

    // Progress reporting (main thread polls; safe because workers don't touch py).
    if (!progress_cb.is_none()) {
        while (!done.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            int64_t att = attempts_total.load();
            int     val = valid_count.load();
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            try {
                py::gil_scoped_acquire g;
                progress_cb(side, val, att, elapsed);
            } catch (...) { /* ignore */ }
            if (val >= n_target || att >= max_attempts) break;
        }
    }
    for (auto& th : pool) th.join();

    SeedResult res;
    res.programs = std::move(valid);
    res.fingerprints = std::move(valid_fps);
    res.total_attempts = attempts_total.load();
    return res;
}

// =============================================================== module def

PYBIND11_MODULE(pushgp_cpp_seeder, m) {
    m.doc() = "C++ accelerated VM/validator/RPG/seeder for Push-GP";

    py::class_<ProgramHandle>(m, "ProgramHandle")
        .def("__len__", [](const ProgramHandle& h) {
            return h.prog ? h.prog->size() : 0;
        })
        .def("total_length", [](const ProgramHandle& h) {
            return h.prog ? program_length(*h.prog) : 0;
        })
        .def("to_dict", [](const ProgramHandle& h) {
            return h.prog ? program_to_dict(*h.prog) : py::list();
        });

    m.def("all_op_names", []() { return all_op_names(); });

    m.def("build_program", [](py::list lst) {
        ProgramHandle h;
        h.prog = std::make_shared<Program>(dict_to_program(lst));
        return h;
    }, py::arg("prog_dict_list"));

    m.def("program_to_dict", [](const ProgramHandle& h) {
        return h.prog ? program_to_dict(*h.prog) : py::list();
    });

    m.def("run_program",
        [](const ProgramHandle& h, const std::string& side,
           py::array_t<double, py::array::c_style | py::array::forcecast> evo,
           py::array_t<double, py::array::c_style | py::array::forcecast> incoming,
           double L_v, int deg, int iter_idx) {
            if (!h.prog) return py::object(py::none());
            return run_program_impl(*h.prog, side, evo, incoming, L_v, deg, iter_idx);
        },
        py::arg("prog"), py::arg("side"), py::arg("evo"),
        py::arg("incoming"), py::arg("L_v") = 0.0, py::arg("deg") = DEFAULT_DEG,
        py::arg("iter_idx") = 0);

    m.def("validate_with_panels",
        [](const ProgramHandle& h, const std::string& side,
           py::list panels,
           py::array_t<double, py::array::c_style | py::array::forcecast> evo,
           int deg) -> py::tuple {
            if (!h.prog) return py::make_tuple(false, std::string("null program"));
            return validate_with_panels_impl(*h.prog, side, panels, evo, deg);
        },
        py::arg("prog"), py::arg("side"), py::arg("panels"),
        py::arg("evo"), py::arg("deg") = DEFAULT_DEG);

    m.def("validate_random",
        [](const ProgramHandle& h, const std::string& side,
           py::array_t<double, py::array::c_style | py::array::forcecast> evo,
           int deg, int num_configs, int num_permutations, uint64_t seed) -> py::tuple {
            if (!h.prog) return py::make_tuple(false, std::string("null program"));
            return validate_random_impl(*h.prog, side, evo, deg, num_configs, num_permutations, seed);
        },
        py::arg("prog"), py::arg("side"), py::arg("evo"),
        py::arg("deg") = DEFAULT_DEG,
        py::arg("num_configs") = DEFAULT_NUM_CONFIGS,
        py::arg("num_permutations") = DEFAULT_NUM_PERMUTATIONS,
        py::arg("seed") = 0);

    m.def("parallel_seed",
        [](const std::string& side, int n_target, int64_t max_attempts,
           int threads, int chunk_attempts, int min_size, int max_size,
           int deg, int num_configs, int num_permutations,
           uint64_t base_seed, py::object progress_cb,
           std::vector<std::string> seen_in,
           std::vector<std::string> allowed_op_names) {
            SeedResult r;
            {
                py::gil_scoped_release rel;
                r = parallel_seed_impl(side, n_target, max_attempts, threads,
                    chunk_attempts, min_size, max_size, deg,
                    num_configs, num_permutations, base_seed, progress_cb,
                    seen_in, allowed_op_names);
            }
            py::list out;
            for (auto& sp : r.programs) {
                ProgramHandle h; h.prog = sp;
                out.append(py::cast(h));
            }
            // Also return the fingerprints so the Python caller can
            // update its own seen-set without recomputing.
            return py::make_tuple(out, r.total_attempts, r.fingerprints);
        },
        py::arg("side"), py::arg("n_target"),
        py::arg("max_attempts") = 100000000LL,
        py::arg("threads") = 8,
        py::arg("chunk_attempts") = 1000,
        py::arg("min_size") = 4,
        py::arg("max_size") = 30,
        py::arg("deg") = DEFAULT_DEG,
        py::arg("num_configs") = DEFAULT_NUM_CONFIGS,
        py::arg("num_permutations") = DEFAULT_NUM_PERMUTATIONS,
        py::arg("base_seed") = 1234,
        py::arg("progress_cb") = py::none(),
        py::arg("seen_fingerprints") = std::vector<std::string>{},
        py::arg("allowed_op_names") = std::vector<std::string>{});

    m.def("compute_behav_fp",
        [](const std::string& side, const ProgramHandle& h) {
            if (!h.prog) return std::string("NAN");
            return compute_behav_fp(side, *h.prog);
        },
        py::arg("side"), py::arg("prog"));
}
