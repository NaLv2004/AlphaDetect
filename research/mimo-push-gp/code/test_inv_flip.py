"""
test_inv_flip.py
================
验证 Gen72_BP_fixed (Float.Inv = 1/x) 的 "最差优先" 现象。

假设: min-heap + score=1/cum_dist → 优先扩展 cum_dist 最大的节点 → 找最差路径 → BER≈0.997
翻转: 在 ForEachChild 末尾加 Float.Neg → score=-1/cum_dist → 优先扩展 cum_dist 最小节点 → 近似最优

三种程序:
  Gen72_BP_fixed     :  ...ForEachChild([GetCumDist, Inv,       SetScore])
  Gen72_BP_inverted  :  ...ForEachChild([GetCumDist, Inv, Neg,  SetScore])
  MMSE-LB            :  [Float.GetMMSELB]
  no-corr (pure A*)  :  []

参数: 400 trials, SNR={10,12,14}, max_nodes=200
"""
import sys, time
import numpy as np

CODE_DIR = r"D:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\code"
sys.path.insert(0, CODE_DIR)

from vm import Instruction
from bp_decoder import BPStackDecoder, qam16_constellation

# ------------------------------------------------------------------
# Program definitions
# ------------------------------------------------------------------

# Gen72 active ForEachChild with FIXED Float.Inv  → score = 1/cum_dist  (worst-first)
GEN72_BP_FIXED = [
    Instruction('Float.GetNoiseVar'),
    Instruction('Node.SetScore'),
    Instruction('Node.ForEachSibling', code_block=[
        Instruction('Node.GetScore'),
    ]),
    Instruction('Node.GetParent'),
    Instruction('Node.ForEachChild', code_block=[
        Instruction('Float.GetMMSELB'),      # dead: int_stack top=16 > Nt=8
        Instruction('Node.GetCumDist'),
        Instruction('Float.Inv'),             # 1/cum_dist  (FIXED bug)
        Instruction('Node.SetScore'),
    ]),
]

# Gen72 with Float.Neg appended → score = -1/cum_dist  (best-first / correct A*)
GEN72_BP_INVERTED = [
    Instruction('Float.GetNoiseVar'),
    Instruction('Node.SetScore'),
    Instruction('Node.ForEachSibling', code_block=[
        Instruction('Node.GetScore'),
    ]),
    Instruction('Node.GetParent'),
    Instruction('Node.ForEachChild', code_block=[
        Instruction('Float.GetMMSELB'),      # dead
        Instruction('Node.GetCumDist'),
        Instruction('Float.Inv'),             # 1/cum_dist
        Instruction('Float.Neg'),             # -1/cum_dist  ← inversion
        Instruction('Node.SetScore'),
    ]),
]

MMSE_LB_PROG  = [Instruction('Float.GetMMSELB')]
NOCORR_PROG   = []   # pure A* (score = cum_dist)

# ------------------------------------------------------------------
# Experiment
# ------------------------------------------------------------------

constellation = qam16_constellation()
Nr, Nt = 16, 8

programs = [
    ('Gen72_BP_fixed',    GEN72_BP_FIXED),
    ('Gen72_BP_inverted', GEN72_BP_INVERTED),
    ('MMSE-LB',           MMSE_LB_PROG),
    ('no-corr (A*)',      NOCORR_PROG),
]

N_TRIALS  = 400
SNR_LIST  = [10, 12, 14, 16]
MAX_NODES = 200

print("="*72)
print("TEST: Inverted-score phenomenon for Gen72_BP_fixed")
print(f"Settings: {N_TRIALS} trials, max_nodes={MAX_NODES}")
print("Gen72_BP_fixed    : score = 1/cum_dist  → WORST-first (min-heap)")
print("Gen72_BP_inverted : score = -1/cum_dist → BEST-first  (correct A*)")
print("="*72)

# Warm up vm import
from vm import MIMOPushVM
vm = MIMOPushVM(step_max=2000, flops_max=5_000_000)

results = {name: {} for name, _ in programs}

for snr_db in SNR_LIST:
    snr_lin = 10 ** (snr_db / 10.0)
    counters = {name: 0 for name, _ in programs}
    n_sym = 0
    rng = np.random.RandomState(42 + snr_db * 7)

    t0 = time.time()
    for trial in range(N_TRIALS):
        H = (rng.randn(Nr, Nt) + 1j * rng.randn(Nr, Nt)) / np.sqrt(2)
        x_idx = rng.randint(0, 16, Nt)
        x = constellation[x_idx]
        sig_p = float(np.mean(np.abs(H @ x) ** 2))
        nv = sig_p / snr_lin
        y = H @ x + np.sqrt(nv / 2) * (rng.randn(Nr) + 1j * rng.randn(Nr))

        for name, prog in programs:
            dec = BPStackDecoder(
                Nt=Nt, Nr=Nr, constellation=constellation,
                max_nodes=MAX_NODES, vm=vm,
            )
            xh, _ = dec.detect(H, y, prog, noise_var=nv)
            counters[name] += int(np.sum(xh != x))

        n_sym += Nt

    elapsed = time.time() - t0
    print(f"\n  SNR={snr_db} dB  ({elapsed:.1f}s)")
    for name, _ in programs:
        ber = counters[name] / n_sym
        results[name][snr_db] = ber
        print(f"    {name:25s}: BER = {ber:.5f}")

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
print("\n" + "="*72)
print("SUMMARY")
print(f"{'Program':25s}", end="")
for s in SNR_LIST:
    print(f"  SNR={s:2d}", end="")
print()
print("-"*72)
for name, _ in programs:
    print(f"{name:25s}", end="")
    for s in SNR_LIST:
        print(f"  {results[name][s]:.5f}", end="")
    print()

# ------------------------------------------------------------------
# additional check: BER_fixed + BER(perfect complement)
# ------------------------------------------------------------------
print("\n  For binary analogy:  1 - BER_fixed  vs  BER_inverted")
print(f"  {'SNR':4s}  {'1-BER_fixed':12s}  {'BER_inverted':12s}  {'ratio':8s}")
for s in SNR_LIST:
    b_fix = results['Gen72_BP_fixed'][s]
    b_inv = results['Gen72_BP_inverted'][s]
    complement = 1.0 - b_fix
    ratio = b_inv / complement if complement > 1e-9 else float('inf')
    print(f"  {s:4d}  {complement:.6f}    {b_inv:.6f}    {ratio:.3f}x")
print()
print("If ratio ≈ 1.0 → the inversion is nearly perfect (binary-like).")
print("If ratio >> 1  → QAM-16 M-ary inversion is only partial.")
