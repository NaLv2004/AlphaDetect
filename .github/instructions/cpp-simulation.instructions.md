---
description: "Use when writing or modifying C++ simulation code for communications systems. Covers Eigen3 matrix patterns, OpenMP parallelism, Monte Carlo simulation structure, parameter configuration headers, and coding conventions from the existing mimo2D codebase."
applyTo: "**/*.cpp, **/*.h"
---

# C++ Communications Simulation Conventions

## Architecture Pattern

Follow the simulation flow from the existing `mimo2D/platform_1/` codebase:

```
main() → system function → SNR sweep loop → frame loop
  ├── generate_info_bits()
  ├── polar_encode()
  ├── interleave() + modulate()
  ├── channel() (AWGN/Rayleigh)
  ├── detect() (MMSE/ZF/SD/K-Best)
  ├── deinterleave() + decode()
  ├── compute_errors() (BER/FER)
  └── early_stop_check()
```

## Parameter Configuration

Use a dedicated `setting.h` header for all configurable parameters:
```cpp
// --- Code Parameters ---
#define CODE_N 1024
#define CODE_K 676
#define NH 32
#define NV 32

// --- MIMO Parameters ---
#define NT 32
#define NR 64
#define MOD_ORDER 4  // 16QAM

// --- Simulation Control ---
#define MIN_SNR 0.0
#define MAX_SNR 10.0
#define SNR_STEP 0.5
#define MAX_FRAMES 100000
#define MIN_ERRORS 100
```

## Eigen3 Usage

- Real matrices: `Eigen::MatrixXd`, `Eigen::VectorXd`
- Complex matrices: `Eigen::MatrixXcd`, `Eigen::VectorXcd`
- Fixed-size for small known dimensions: `Eigen::Matrix2cd` for 2×2 complex
- Include path: `mimo2D/eigen3/`
- Use `.noalias()` for matrix multiplications to avoid temporaries
- Use `.colwise()` / `.rowwise()` for per-column/row operations

## OpenMP Parallelism

```cpp
#pragma omp parallel for schedule(dynamic) reduction(+:total_errors, total_bits)
for (int frame = 0; frame < max_frames; frame++) {
    // Per-frame simulation (thread-safe)
}
```

- Use `schedule(dynamic)` for load balancing across SNR-dependent work
- Use `reduction` for error/bit counters
- Ensure thread-local random number generators (seed with `omp_get_thread_num()`)

## Results Output

Write results to plain text files:
```cpp
// SNR(dB)  BER  FER  Frames  Errors  Time(s)
fprintf(fp, "%.1f  %.6e  %.6e  %d  %d  %.2f\n", snr, ber, fer, frames, errors, time);
```

## Early Stopping

```cpp
if (error_count >= MIN_ERRORS && frame_count >= MIN_FRAMES_PER_SNR) break;
if (fer < TARGET_FER && frame_count >= MIN_FRAMES_PER_SNR) break;
```
