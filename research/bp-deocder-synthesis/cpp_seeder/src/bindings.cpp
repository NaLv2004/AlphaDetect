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
#include "symbolic_expr.hpp"
#include "symbolic_validator.hpp"
#include "symbolic_vm.hpp"
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
//    "permutations": list[np.ndarray]}
// Input probes are deterministically derived from `incoming` + delta
// (matching pushgp.validators._build_input_probes), so we don't accept
// them from Python anymore.  The legacy "perturb_indices" key is
// silently ignored if present (rev-2 validator doesn't use it).
static std::vector<ConfigPanel> dict_to_panels(const py::list& py_panels,
                                               double delta = DEFAULT_PERTURB_DELTA) {
    std::vector<ConfigPanel> out;
    out.reserve(py_panels.size());
    for (auto&& item : py_panels) {
        py::dict d = item.cast<py::dict>();
        ConfigPanel p;
        if (d.contains("L_v")) p.L_v = d["L_v"].cast<double>();
        p.incoming = numpy_to_fvec(d["incoming"].cast<py::array_t<double>>());
        p.input_probes = build_input_probes(p.incoming, delta);
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
    int deg, int num_configs, int num_permutations, int num_evo_panels, uint64_t seed)
{
    auto evo_arr = numpy_to_evo(evo);
    ValidationResult r = (side == "v2c")
        ? validate_v2c_random(prog, evo_arr, deg, num_configs, num_permutations, num_evo_panels, seed)
        : validate_c2v_random(prog, evo_arr, deg, num_configs, num_permutations, num_evo_panels, seed);
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
    int num_evo_panels,
    uint64_t base_seed,
    py::object progress_cb,
    const std::vector<std::string>& seen_in,
    const std::vector<std::string>& allowed_op_names,
    const std::string& validator_mode = "probe")
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
                Program prog = (side == "v2c")
                    ? rpg.random_program_v2c(instr_set, min_size, max_size)
                    : rpg.random_program_c2v(instr_set, min_size, max_size);
                uint64_t sub_seed = (my_seed * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(i));
                // probe validator: run unless mode is pure "symbolic"
                if (validator_mode != "symbolic") {
                    ValidationResult r = (side == "v2c")
                        ? validate_v2c_random(prog, evo_default, deg, num_configs, num_permutations, num_evo_panels, sub_seed)
                        : validate_c2v_random(prog, evo_default, deg, num_configs, num_permutations, num_evo_panels, sub_seed);
                    if (!r.ok) continue;
                }
                // symbolic validator: run if mode is "symbolic" or "both".
                // Wrap in try/catch since random programs may trigger
                // exceptions — treat any failure as rejection rather than
                // crashing the whole search.
                if (validator_mode == "symbolic" || validator_mode == "both") {
                    bool sym_ok = false;
                    try {
                        CheckResult sr = (side == "v2c")
                            ? validate_symbolic_v2c(prog, deg, 0)
                            : validate_symbolic_c2v(prog, deg, 0);
                        sym_ok = sr.ok;
                    } catch (...) {
                        sym_ok = false;
                    }
                    if (!sym_ok) continue;
                }
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
           int deg, int num_configs, int num_permutations, int num_evo_panels, uint64_t seed) -> py::tuple {
            if (!h.prog) return py::make_tuple(false, std::string("null program"));
            return validate_random_impl(*h.prog, side, evo, deg, num_configs, num_permutations, num_evo_panels, seed);
        },
        py::arg("prog"), py::arg("side"), py::arg("evo"),
        py::arg("deg") = DEFAULT_DEG,
        py::arg("num_configs") = DEFAULT_NUM_CONFIGS,
        py::arg("num_permutations") = DEFAULT_NUM_PERMUTATIONS,
        py::arg("num_evo_panels") = DEFAULT_NUM_EVO_PANELS,
        py::arg("seed") = 0);

    m.def("parallel_seed",
        [](const std::string& side, int n_target, int64_t max_attempts,
           int threads, int chunk_attempts, int min_size, int max_size,
           int deg, int num_configs, int num_permutations, int num_evo_panels,
           uint64_t base_seed, py::object progress_cb,
           std::vector<std::string> seen_in,
           std::vector<std::string> allowed_op_names,
           std::string validator_mode) {
            SeedResult r;
            {
                py::gil_scoped_release rel;
                r = parallel_seed_impl(side, n_target, max_attempts, threads,
                    chunk_attempts, min_size, max_size, deg,
                    num_configs, num_permutations, num_evo_panels, base_seed, progress_cb,
                    seen_in, allowed_op_names, validator_mode);
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
        py::arg("num_evo_panels") = DEFAULT_NUM_EVO_PANELS,
        py::arg("base_seed") = 1234,
        py::arg("progress_cb") = py::none(),
        py::arg("seen_fingerprints") = std::vector<std::string>{},
        py::arg("allowed_op_names") = std::vector<std::string>{},
        py::arg("validator_mode") = std::string("probe"));

    m.def("compute_behav_fp",
        [](const std::string& side, const ProgramHandle& h) {
            if (!h.prog) return std::string("NAN");
            return compute_behav_fp(side, *h.prog);
        },
        py::arg("side"), py::arg("prog"));

    // ============================================================== symbolic

    m.def("symbolic_validate_v2c",
        [](const ProgramHandle& h, int deg, int iter_idx) -> py::tuple {
            if (!h.prog) return py::make_tuple(false, std::string("null program"));
            auto r = validate_symbolic_v2c(*h.prog, deg, iter_idx);
            return py::make_tuple(r.ok, r.reason);
        },
        py::arg("prog"), py::arg("deg") = DEFAULT_DEG, py::arg("iter_idx") = 0);

    m.def("symbolic_validate_c2v",
        [](const ProgramHandle& h, int deg, int iter_idx) -> py::tuple {
            if (!h.prog) return py::make_tuple(false, std::string("null program"));
            auto r = validate_symbolic_c2v(*h.prog, deg, iter_idx);
            return py::make_tuple(r.ok, r.reason);
        },
        py::arg("prog"), py::arg("deg") = DEFAULT_DEG, py::arg("iter_idx") = 0);

    // Generic symbolic check: caller supplies CheckSpec as a dict.
    //
    // spec = {
    //   "deps_required": [{"kind": "X", "indices": []|[0,3,...]}, ...],
    //       # kind in {"X","LV","EVO"}; indices empty = any-of-kind
    //   "sym_groups":    [{"kind": "X", "indices": [0,1,2,3,4,5,6]}, ...],
    //   "require_odd":   true|false,
    //   "odd_negate":    [{"kind":"X","indices":[0,1,...]}, {"kind":"LV","indices":[0]}]
    // }
    // side ∈ {"v2c","c2v"} — selects which atoms are seeded onto the
    // initial stacks; the validator does NOT enforce side-specific spec.
    m.def("symbolic_check",
        [](const ProgramHandle& h, const std::string& side,
           int deg, int iter_idx, py::dict spec_dict) -> py::tuple {
            if (!h.prog) return py::make_tuple(false, std::string("null program"));
            SymbolicVM vm;
            vm.ctx_deg = deg;
            vm.ctx_iter = iter_idx;
            vm.state.clear();
            if (side == "v2c") vm.seed_v2c_atoms();
            else if (side == "c2v") vm.seed_c2v_atoms();
            else return py::make_tuple(false, std::string("unknown side"));
            ExprRef out = vm.run(*h.prog);
            if (vm.opaque) return py::make_tuple(false,
                std::string("opaque: ") + vm.opaque_reason);
            if (!out) return py::make_tuple(false, std::string("empty float stack"));

            auto kind_from_str = [](const std::string& s) -> AtomKind {
                if (s == "X") return AtomKind::X;
                if (s == "LV") return AtomKind::LV;
                return AtomKind::EVO;
            };
            CheckSpec spec;
            if (spec_dict.contains("deps_required")) {
                for (auto item : spec_dict["deps_required"].cast<py::list>()) {
                    py::dict d = item.cast<py::dict>();
                    DepRequirement dr;
                    dr.kind = kind_from_str(d["kind"].cast<std::string>());
                    for (auto v : d["indices"].cast<py::list>())
                        dr.indices.push_back(v.cast<int>());
                    spec.deps_required.push_back(std::move(dr));
                }
            }
            if (spec_dict.contains("sym_groups")) {
                for (auto item : spec_dict["sym_groups"].cast<py::list>()) {
                    py::dict d = item.cast<py::dict>();
                    SymGroup g;
                    g.kind = kind_from_str(d["kind"].cast<std::string>());
                    for (auto v : d["indices"].cast<py::list>())
                        g.indices.push_back(v.cast<int>());
                    spec.sym_groups.push_back(std::move(g));
                }
            }
            if (spec_dict.contains("require_odd"))
                spec.require_odd = spec_dict["require_odd"].cast<bool>();
            if (spec_dict.contains("odd_negate")) {
                for (auto item : spec_dict["odd_negate"].cast<py::list>()) {
                    py::dict d = item.cast<py::dict>();
                    AtomKind k = kind_from_str(d["kind"].cast<std::string>());
                    for (auto v : d["indices"].cast<py::list>())
                        spec.odd_negate_atoms.push_back(encode_atom(k, v.cast<int>()));
                }
            }
            auto r = symbolic_validate(out, spec);
            return py::make_tuple(r.ok, r.reason);
        },
        py::arg("prog"), py::arg("side"), py::arg("deg") = DEFAULT_DEG,
        py::arg("iter_idx") = 0, py::arg("spec") = py::dict());

    m.def("symbolic_expr_table_size",
        []() { return expr_table().size(); });

    m.def("symbolic_expr_table_clear",
        []() { expr_table().clear(); });

    // Debug dump: returns list of per-path {ok, reason, cond, out},
    // plus combined output string and σ-substituted combined output
    // string for a chosen pair-swap of X[a] <-> X[b].
    m.def("symbolic_dump_v2c",
        [](const ProgramHandle& h, int deg, int iter_idx, int swap_a, int swap_b) -> py::dict {
            py::dict r;
            if (!h.prog) { r["error"] = "null program"; return r; }
            SymbolicVM vm;
            vm.ctx_deg = deg;
            vm.ctx_iter = iter_idx;
            vm.state.clear();
            vm.seed_v2c_atoms();
            auto paths = dump_all_paths_(vm, *h.prog);
            py::list lst;
            for (auto& p : paths) {
                py::dict d;
                d["ok"] = p.ok;
                d["reason"] = p.reason;
                d["cond"] = p.cond;
                d["out"] = p.out;
                lst.append(d);
            }
            r["paths"] = lst;
            // combined output:
            SymbolicVM vm2;
            vm2.ctx_deg = deg;
            vm2.ctx_iter = iter_idx;
            vm2.state.clear();
            vm2.seed_v2c_atoms();
            auto raw = run_all_paths_(vm2, *h.prog);
            ExprRef combined;
            bool all_ok = true;
            for (auto& p : raw) if (!p.ok) { all_ok = false; break; }
            if (all_ok) combined = combine_paths_output_(raw);
            r["combined"] = expr_to_string(combined);
            if (combined) {
                AtomMap mp;
                mp.table[encode_atom(AtomKind::X, swap_a)] = encode_atom(AtomKind::X, swap_b);
                mp.table[encode_atom(AtomKind::X, swap_b)] = encode_atom(AtomKind::X, swap_a);
                ExprRef swapped = substitute(combined, mp);
                r["sigma"] = expr_to_string(swapped);
                r["sym_equal"] = (swapped.get() == combined.get());
            } else {
                r["sigma"] = std::string("<null>");
                r["sym_equal"] = false;
            }
            return r;
        },
        py::arg("prog"), py::arg("deg") = DEFAULT_DEG,
        py::arg("iter_idx") = 0, py::arg("swap_a") = 0, py::arg("swap_b") = 1);

    // Per-instruction symbolic trace. Runs the program with a single forced
    // path (no branches expected; opaque if branches needed). After every
    // dispatched instruction (including those nested inside DoRange/While
    // bodies) records a snapshot of all stacks rendered via expr_to_string.
    // Returns:
    //   { "steps": [ {op, float[], int[], bool[], fvec[][]}, ... ],
    //     "opaque": bool, "opaque_reason": str, "step_count": int,
    //     "branches_seen": int }
    m.def("symbolic_trace_v2c",
        [](const ProgramHandle& h, int deg, int iter_idx) -> py::dict {
            py::dict r;
            if (!h.prog) { r["error"] = "null program"; return r; }
            SymbolicVM vm;
            vm.ctx_deg = deg;
            vm.ctx_iter = iter_idx;
            vm.state.clear();
            vm.seed_v2c_atoms();
            py::list steps;
            vm.trace_hook = [&](const Instruction& ins) {
                py::dict d;
                d["op"] = op_to_name(ins.op);
                py::list fs;
                for (auto& e : vm.state.floats) fs.append(expr_to_string(e));
                d["float"] = fs;
                py::list is;
                for (auto& e : vm.state.ints) is.append(expr_to_string(e));
                d["int"] = is;
                py::list bs;
                for (auto& e : vm.state.bools) bs.append(expr_to_string(e));
                d["bool"] = bs;
                py::list fvs;
                for (auto& v : vm.state.fvecs) {
                    py::list one;
                    for (auto& e : v) one.append(expr_to_string(e));
                    fvs.append(one);
                }
                d["fvec"] = fvs;
                d["opaque"] = vm.opaque;
                steps.append(d);
            };
            vm.execute_block(*h.prog);
            r["steps"] = steps;
            r["opaque"] = vm.opaque;
            r["opaque_reason"] = vm.opaque_reason;
            r["step_count"] = vm.step_count;
            r["branches_seen"] = vm.total_branches_seen;
            return r;
        },
        py::arg("prog"), py::arg("deg") = DEFAULT_DEG, py::arg("iter_idx") = 0);
}
