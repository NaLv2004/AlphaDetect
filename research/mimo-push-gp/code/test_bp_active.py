"""Verify fixed BP patterns actually trigger SetScore."""
from vm import MIMOPushVM, Instruction, program_to_oneliner
from bp_decoder import BPStackDecoder, qam16_constellation
from evolution import random_bp_pattern
import numpy as np

rng = np.random.RandomState(42)
const = qam16_constellation()

# Generate a larger MIMO instance for more tree nodes
Nt, Nr = 8, 16
H = (np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)) / np.sqrt(2)
x = const[np.random.randint(len(const), size=Nt)]
nv = 0.1
y = H @ x + np.sqrt(nv/2) * (np.random.randn(Nr) + 1j*np.random.randn(Nr))

n_tested = 0
n_bp_active = 0

for i in range(50):
    prog = random_bp_pattern(rng)
    vm = MIMOPushVM(flops_max=2000000, step_max=3000)
    dec = BPStackDecoder(Nt=Nt, Nr=Nr, constellation=const, max_nodes=200, vm=vm)
    try:
        x_hat, flops = dec.detect(H, y, prog, noise_var=nv)
        n_tested += 1
        if dec.bp_updates > 0:
            n_bp_active += 1
            print(f"Pattern {i}: bp_updates={dec.bp_updates}, prog={program_to_oneliner(prog)[:120]}")
    except Exception as e:
        pass

print(f"\nBP activity: {n_bp_active}/{n_tested} programs triggered SetScore")
