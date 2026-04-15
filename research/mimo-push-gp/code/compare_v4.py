"""Per-SNR comparison of discovered v4 program vs distance ordering."""
import numpy as np
from stack_decoder import StackDecoder, qam16_constellation, kbest_detect
from main import generate_mimo_sample, ber_calc
from vm import Instruction

rng = np.random.RandomState(42)
constellation = qam16_constellation()
Nr, Nt = 16, 8
n_trials = 300

# v4 gen9 discovered program
discovered_v4 = [
    Instruction('Node.GetParent'),
    Instruction('Float.FromInt'),
    Instruction('Node.ForEachSibling', code_block=[
        Instruction('Float.Swap'),
        Instruction('Matrix.Pop'),
        Instruction('Int.LT'),
    ]),
    Instruction('Vec.Sub'),
    Instruction('Node.ForEachAncestor', code_block=[
        Instruction('Int.LT'),
        Instruction('Int.Const1'),
        Instruction('Vec.ElementAt'),
    ]),
]
dist_prog = [Instruction('Float.Const0')]

dec60 = StackDecoder(Nt=Nt, Nr=Nr, constellation=constellation, max_nodes=60)
dec_eval = StackDecoder(Nt=Nt, Nr=Nr, constellation=constellation, max_nodes=1500)

print("SNR  | Dist(60)   | V4(60)     | Dist(1500) | V4(1500)   | KB16(100K)")
print("-" * 75)
for snr in [8, 10, 12, 14, 16]:
    d60, v60, d1500, v1500, kb16 = [], [], [], [], []
    for _ in range(n_trials):
        H, x, y, nv = generate_mimo_sample(Nr, Nt, constellation, snr, rng)
        x1, _ = dec60.detect(H, y, dist_prog, noise_var=float(nv))
        x2, _ = dec60.detect(H, y, discovered_v4, noise_var=float(nv))
        x3, _ = dec_eval.detect(H, y, dist_prog, noise_var=float(nv))
        x4, _ = dec_eval.detect(H, y, discovered_v4, noise_var=float(nv))
        xk, _ = kbest_detect(H, y, constellation, K=16)
        d60.append(ber_calc(x, x1))
        v60.append(ber_calc(x, x2))
        d1500.append(ber_calc(x, x3))
        v1500.append(ber_calc(x, x4))
        kb16.append(ber_calc(x, xk))
    print(f"{snr:4d} | {np.mean(d60):10.5f} | {np.mean(v60):10.5f} | {np.mean(d1500):10.5f} | {np.mean(v1500):10.5f} | {np.mean(kb16):10.5f}")
