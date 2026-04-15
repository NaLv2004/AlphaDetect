"""Test MMSE-LB combined with k*noise_var correction."""
import numpy as np
from vm import Instruction
from stack_decoder import StackDecoder, qam16_constellation, MIMOPushVM

# Programs to test
mmse_prog = [Instruction('Float.GetMMSELB')]
knoisevar_prog = [Instruction('Node.GetLayer'), Instruction('Float.FromInt'),
                  Instruction('Float.GetNoiseVar'), Instruction('Float.Mul')]
combined_prog = mmse_prog + knoisevar_prog + [Instruction('Float.Min')]
distance_prog = []

constellation = qam16_constellation()
Nr, Nt = 16, 8

progs = [
    ('dist', distance_prog),
    ('mmse', mmse_prog),
    ('k*nv', knoisevar_prog),
    ('min(mmse,k*nv)', combined_prog),
]

print('SNR  | dist     | mmse     | k*nv     | min(mmse,k*nv)')
print('----------------------------------------------------------')

for snr_db in [8, 10, 12, 14, 16]:
    snr_lin = 10**(snr_db/10)
    n_errs = {name: 0 for name, _ in progs}
    n_sym = 0
    rng = np.random.RandomState(42 + snr_db)
    for trial in range(200):
        x_idx = rng.randint(0, 16, Nt)
        x = constellation[x_idx]
        H = (rng.randn(Nr, Nt) + 1j*rng.randn(Nr, Nt)) / np.sqrt(2)
        sig_p = float(np.mean(np.abs(H @ x)**2))
        nv = sig_p / snr_lin
        y = H @ x + np.sqrt(nv/2) * (rng.randn(Nr) + 1j*rng.randn(Nr))
        
        vm = MIMOPushVM(step_max=800, flops_max=500000)
        for name, prog in progs:
            dec = StackDecoder(Nt=Nt, Nr=Nr, constellation=constellation, max_nodes=60, vm=vm)
            xh, _ = dec.detect(H, y, prog, noise_var=nv)
            n_errs[name] += int(np.sum(xh != x))
        n_sym += Nt
    
    bers = [n_errs[name]/n_sym for name, _ in progs]
    print(f'{snr_db:4d} | {bers[0]:.5f} | {bers[1]:.5f} | {bers[2]:.5f} | {bers[3]:.5f}')
