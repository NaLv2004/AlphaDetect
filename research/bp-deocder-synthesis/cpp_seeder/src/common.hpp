// common.hpp — shared constants matching pushgp/types.py
#pragma once

#include <cstdint>
#include <limits>

namespace pushgp_cpp {

// Numeric guards (mirror pushgp/types.py + instructions.py)
constexpr double FLOAT_ABS_MAX = 1e12;
constexpr double NAN_INF_REPLACEMENT = 0.0;
constexpr int64_t INT_ABS_MAX = 1000000000;       // ±1e9
constexpr double FLOAT_EQ_EPSILON = 1e-12;
constexpr double EXP_INPUT_CAP = 80.0;            // a >= 80 → 0.0
constexpr int    MAX_VEC_LEN = 1024;
constexpr int    MEMORY_SIZE = 16;
constexpr int    N_EVO_CONSTS = 8;

// Per-type stack depth caps (DEFAULT_MAX_DEPTHS in types.py)
constexpr int MAX_DEPTH_FLOAT = 200;
constexpr int MAX_DEPTH_INT   = 200;
constexpr int MAX_DEPTH_BOOL  = 200;
constexpr int MAX_DEPTH_FVEC  = 50;
constexpr int MAX_DEPTH_BVEC  = 50;
constexpr int MAX_DEPTH_IVEC  = 50;

// VM resource budgets (vm.py)
constexpr int VM_STEP_MAX  = 2000;
constexpr int VM_FLOP_MAX  = 50000;
constexpr int VM_RECUR_MAX = 32;

// Loop caps (instructions.py)
constexpr int N_MAX_LOOP  = 64;
constexpr int N_MAX_WHILE = 32;

// Validator constants (validators.py)
constexpr int    DEFAULT_DEG = 8;
constexpr int    DEFAULT_NUM_CONFIGS      = 3;
constexpr int    DEFAULT_NUM_PERMUTATIONS = 5;
constexpr double DEFAULT_PERTURB_DELTA = 1.7;
constexpr double EPS_DEPENDENCY  = 1e-9;
constexpr double EPS_INVARIANCE  = 1e-7;

// Genome constants (genome.py)
constexpr double LOG_CONST_MIN = -3.0;
constexpr double LOG_CONST_MAX =  3.0;

}  // namespace pushgp_cpp
