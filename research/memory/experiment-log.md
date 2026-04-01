# Experiment Log

Chronological record of all experiments conducted by the research system.
Each entry includes parameters, results, and key observations.

---

<!-- New entries will be appended below this line -->

## [2026-04-01] Baseline Python MIMO Detection — 32x16, 16QAM, i.i.d. Rayleigh
- **Topic**: research/auto-discover-mimo-det/ (Phase 0 baseline)
- **Parameters**: TxAnt=16, RxAnt=32, 16QAM (ModType=4), uncoded, Eb/N0=6~14dB step 2, seed=114514, delta=0.7, max_frame_errors=50
- **Noise model**: Eb/N0 → σ²=Nr/(10^(SNR/10)·Mc), complex AWGN, real-valued decomposition

### MMSE Detector
| Eb/N0 (dB) | BER       | FER       | Samples |
|------------|-----------|-----------|---------|
| 6          | 7.02E-02  | 9.80E-01  | 51      |
| 8          | 4.89E-02  | 9.43E-01  | 53      |
| 10         | 1.29E-02  | 5.38E-01  | 93      |
| 12         | 5.48E-03  | 2.58E-01  | 194     |
| 14         | 7.89E-04  | 4.85E-02  | 1030    |

### AMP Detector (10 iterations, θ_τs=0.5, θ_τz=0.5)
| Eb/N0 (dB) | BER       | FER       | Samples |
|------------|-----------|-----------|---------|
| 6          | 5.39E-02  | 9.80E-01  | 51      |
| 8          | 2.22E-02  | 6.02E-01  | 83      |
| 10         | 4.44E-03  | 2.22E-01  | 225     |
| 12         | 4.69E-04  | 2.83E-02  | 1764    |
| 14         | 3.44E-05  | 2.20E-03  | 22707   |
- **12dB single-run BER**: 4.69E-04 ✓ (target: 3E-4~5E-4)

### EP Detector (7 iterations, delta=0.7)
| Eb/N0 (dB) | BER       | FER       | Samples |
|------------|-----------|-----------|---------|
| 6          | 5.59E-02  | 9.62E-01  | 52      |
| 8          | 3.27E-02  | 7.81E-01  | 64      |
| 10         | 5.05E-03  | 2.49E-01  | 201     |
| 12         | 4.83E-04  | 3.03E-02  | 1651    |
| 14         | 2.16E-05  | 1.33E-03  | 37592   |
- **12dB single-run BER**: 4.83E-04 ✓ (target: 3E-4~5E-4)

### K-Best Detector (K=128, with QR decomposition)
| Eb/N0 (dB) | BER       | FER       | Samples |
|------------|-----------|-----------|---------|
| 12         | 3.82E-04  | 2.40E-02  | 2087    |

- **Observation**: EP outperforms AMP at high SNR (14dB: EP 2.16E-5 vs AMP 3.44E-5). MMSE shows ~10x worse BER than iterative detectors at 12dB. K-Best (K=128) gives near-ML performance, competitive with AMP/EP at 12dB.
- **Files**: project-AutoDiscoverMimoDetAlg/baseline_alg/{run_sim.py, sim_utils.py, amp_det.py, ep_det.py, kbest_det.py, mmse_det.py}

