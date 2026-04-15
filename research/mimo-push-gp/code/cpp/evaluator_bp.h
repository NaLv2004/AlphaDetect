/*
 * evaluator_bp.h — C++ Structured BP Stack Decoder Evaluator
 *
 * Extends the base evaluator with 4-program BP genome support:
 *   prog_down, prog_up, prog_belief, prog_halt
 *
 * Compile (MSVC):
 *   cl.exe /EHsc /O2 /openmp /std:c++17 evaluator_bp.cpp /LD /Fe:evaluator_bp.dll
 */
#pragma once
#include <cstdint>

#ifdef _WIN32
  #define BP_EXPORT __declspec(dllexport)
#else
  #define BP_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// Create BP evaluator context
// constellation: interleaved [re0, im0, re1, im1, ...] (2*M doubles)
BP_EXPORT void* bp_eval_create(int Nt, int Nr, int M,
                                const double* constellation,
                                int max_nodes, int flops_max, int step_max,
                                int max_bp_iters);

BP_EXPORT void bp_eval_destroy(void* ctx);

// Evaluate a 4-program BP genome on a single MIMO sample
// Returns BER (fraction of symbol errors)
BP_EXPORT double bp_eval_one(void* ctx,
                              const int* prog_down, int down_len,
                              const int* prog_up, int up_len,
                              const int* prog_belief, int belief_len,
                              const int* prog_halt, int halt_len,
                              const double* H, const double* y,
                              const double* x_true,
                              double noise_var,
                              double* flops_out,
                              int* bp_calls_out);

// Evaluate a 4-program BP genome on a dataset of samples
// OpenMP parallel over samples
// Returns: avg_ber via return value, avg_flops and total_faults via pointers
BP_EXPORT double bp_eval_dataset(void* ctx,
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
                                  int* total_faults_out,
                                  double* avg_bp_calls_out);

// Batch evaluate: multiple genomes on same dataset (OpenMP parallel over genomes)
// Each genome is 4 programs concatenated: [down | up | belief | halt]
// prog_offsets[i*4+0..3] = start offset of each program in prog_all
// prog_lengths[i*4+0..3] = length of each program
BP_EXPORT void bp_eval_batch(void* ctx,
                              int n_genomes,
                              const int* prog_all,
                              const int* prog_offsets,
                              const int* prog_lengths,
                              int n_samples,
                              const double* H_all,
                              const double* y_all,
                              const double* x_true_all,
                              const double* noise_vars,
                              double* ber_out,
                              double* flops_out,
                              int* faults_out,
                              double* bp_calls_out);

// ===========================================================================
// Baseline detectors (LMMSE, K-Best)
// ===========================================================================

// Evaluate LMMSE, K-Best-16, K-Best-32 on a dataset of samples.
// Returns per-detector results via output arrays (length n_samples).
// ber_lmmse_out, ber_kb16_out, ber_kb32_out: per-sample BER (0 or errors/Nt)
// Or pass NULL to skip a detector.
BP_EXPORT void bp_eval_baselines(void* ctx,
                                  int n_samples,
                                  const double* H_all,
                                  const double* y_all,
                                  const double* x_true_all,
                                  const double* noise_vars,
                                  double* ber_lmmse_out,
                                  double* ber_kb16_out,
                                  double* ber_kb32_out);


// ===========================================================================
// MMSE-LB Stack Decoder baseline (no BP, no evolved programs)
// ===========================================================================
BP_EXPORT double bp_eval_mmselb_stack(void* ctx,
                                       int n_samples,
                                       const double* H_all,
                                       const double* y_all,
                                       const double* x_true_all,
                                       const double* noise_vars,
                                       int override_max_nodes);

// Evaluate evolved BP programs at multiple node-count limits
BP_EXPORT void bp_eval_multi_nodes(void* ctx,
                                    const int* prog_down, int down_len,
                                    const int* prog_up, int up_len,
                                    const int* prog_belief, int belief_len,
                                    const int* prog_halt, int halt_len,
                                    int n_samples,
                                    const double* H_all,
                                    const double* y_all,
                                    const double* x_true_all,
                                    const double* noise_vars,
                                    const int* node_limits,
                                    int n_limits,
                                    double* ber_out);

// Evaluate MMSE-LB stack decoder at multiple node-count limits
BP_EXPORT void bp_eval_mmselb_multi_nodes(void* ctx,
                                           int n_samples,
                                           const double* H_all,
                                           const double* y_all,
                                           const double* x_true_all,
                                           const double* noise_vars,
                                           const int* node_limits,
                                           int n_limits,
                                           double* ber_out);

} // extern "C"
