// common.hpp — shared constants mirroring pushgp/types.py + instructions.py.
//
// VM guard policy (matches Python):
//   * DOMAIN errors (log<=0, sqrt<0, atanh|x|>=1, 1/0, x%0, NaN result) ->
//     immediate VM fault.  Caller treats program as rejected.
//   * OVERFLOW (exp(large), large products, ±Inf) -> magnitude clamped to
//     ±FLOAT_CLAMP_DEFAULT (30.0), matching the LLR scale.  May be
//     overridden at compile time by -DPUSHGP_FLOAT_CLAMP=<value>.
#pragma once

#include <cstdint>
#include <limits>

namespace pushgp_cpp {

// ------- Numeric guards ---------------------------------------------------
// Hard ceiling used by data-path sanitisers (incoming vec / constants /
// memory copies).  NOT a math-op clamp.
constexpr double FLOAT_ABS_MAX     = 1e12;
// Replacement value used by data-path sanitiser when an input is non-finite.
constexpr double NAN_INF_REPLACEMENT = 0.0;
// Hard cap on Float.Exp input; above this we treat as overflow (clamp +).
constexpr double EXP_INPUT_HARD_CAP = 700.0;
// Int magnitude clamp (mirrors Python's 10**9 cap in _int_binop).
constexpr int64_t INT_ABS_MAX      = 1000000000;
// Float equality tolerance (Float.EQ).
constexpr double FLOAT_EQ_EPSILON  = 1e-12;
// Vector / memory sizing.
constexpr int    MAX_VEC_LEN = 1024;
constexpr int    MEMORY_SIZE = 16;
constexpr int    N_EVO_CONSTS = 8;

// Default math-op overflow clamp.  Must mirror pushgp/types.py
// `_FLOAT_CLAMP = _read_clamp_env(30.0)`.  Override at compile time with
// -DPUSHGP_FLOAT_CLAMP=<value> if you want to match a different Python
// PUSHGP_FLOAT_CLAMP env setting.
#ifndef PUSHGP_FLOAT_CLAMP
#define PUSHGP_FLOAT_CLAMP 30.0
#endif
constexpr double FLOAT_CLAMP_DEFAULT = PUSHGP_FLOAT_CLAMP;

// Per-type stack depth caps (DEFAULT_MAX_DEPTHS in types.py).
constexpr int MAX_DEPTH_FLOAT = 200;
constexpr int MAX_DEPTH_INT   = 200;
constexpr int MAX_DEPTH_BOOL  = 200;
constexpr int MAX_DEPTH_FVEC  = 50;
constexpr int MAX_DEPTH_BVEC  = 50;
constexpr int MAX_DEPTH_IVEC  = 50;

// VM resource budgets (vm.py).
constexpr int VM_STEP_MAX  = 2000;
constexpr int VM_FLOP_MAX  = 50000;
constexpr int VM_RECUR_MAX = 32;

// Loop caps (instructions.py).
constexpr int N_MAX_LOOP  = 64;
constexpr int N_MAX_WHILE = 32;

// ------- Validator constants ---------------------------------------------
// Mirror pushgp/validators.py rev 2.
constexpr int    DEFAULT_DEG               = 8;
constexpr int    DEFAULT_NUM_CONFIGS       = 5;
constexpr int    DEFAULT_NUM_EVO_PANELS    = 0;   // random evo sampling off
constexpr int    DEFAULT_NUM_PERMUTATIONS  = 5;
constexpr double DEFAULT_PERTURB_DELTA     = 1.7;
constexpr double EPS_DEPENDENCY            = 1e-9;
constexpr double EPS_INVARIANCE            = 1e-7;
constexpr double VAL_EVO_LOG_LO            = -3.0;
constexpr double VAL_EVO_LOG_HI            =  3.0;
constexpr int    VAL_EVO_PANEL_SIZE        = 8;
// Random base sample range for (incoming, L_v); mirrors validators.py.
constexpr double INCOMING_SAMPLE_LO        = -10.0;
constexpr double INCOMING_SAMPLE_HI        =  10.0;
constexpr double L_V_SAMPLE_LO             = -10.0;
constexpr double L_V_SAMPLE_HI             =  10.0;

// ------- Genome constants -------------------------------------------------
// Mirror pushgp/genome.py LOG_CONST_MIN/MAX (widened to allow OMS sentinel).
constexpr double LOG_CONST_MIN = -1.0;
constexpr double LOG_CONST_MAX =  6.0;

}  // namespace pushgp_cpp
