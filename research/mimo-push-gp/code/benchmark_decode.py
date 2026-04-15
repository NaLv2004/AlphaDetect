"""Benchmark decode time with BP vs without."""
import time
import sys
sys.path.insert(0, r'd:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\code')

from vm import MIMOPushVM, Instruction
from bp_decoder import BPStackDecoder, qam16_constellation
from evolution import random_bp_pattern, random_program
import numpy as np

const = qam16_constellation()
Nt, Nr = 8, 16
rng = np.random.RandomState(42)

H = (np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)) / np.sqrt(2)
x = const[np.random.randint(len(const), size=Nt)]
nv = 0.1
y = H @ x + np.sqrt(nv/2) * (np.random.randn(Nr) + 1j*np.random.randn(Nr))

# Test with BP pattern (ForEachSibling)
prog_bp = random_bp_pattern(rng)

# Test with simple program (no BP)
prog_simple = [Instruction('Node.GetCumDist'), Instruction('Float.Const0')]

# Test with random program (may or may not have BP)
prog_rand = random_program(min_size=8, max_size=15, rng=rng, env_bias=0.4)

def time_decode(prog, max_nodes, step_max, n_reps=5):
    times = []
    for _ in range(n_reps):
        vm = MIMOPushVM(flops_max=4000000, step_max=step_max)
        dec = BPStackDecoder(Nt=Nt, Nr=Nr, constellation=const, max_nodes=max_nodes, vm=vm)
        t0 = time.time()
        _ = dec.detect(H, y, prog, noise_var=nv)
        dt = time.time() - t0
        times.append(dt)
    return np.mean(times), dec.bp_updates

print(f"bp_pattern prog (len={len(prog_bp)}):")
for nn in [100, 200, 500]:
    t, bpu = time_decode(prog_bp, nn, 3000)
    print(f"  max_nodes={nn}: {t:.3f}s, bp_updates={bpu}")

print(f"\nsimple prog (len={len(prog_simple)}):")
for nn in [100, 200, 500]:
    t, bpu = time_decode(prog_simple, nn, 1500)
    print(f"  max_nodes={nn}: {t:.3f}s, bp_updates={bpu}")
