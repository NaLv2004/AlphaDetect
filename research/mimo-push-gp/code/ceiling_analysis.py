"""Ceiling analysis: pure cum_dist at various max_nodes with correct 16-QAM."""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from bp_main_v2 import Genome, Instruction, generate_mimo_sample, N_EVO_CONSTS
from bp_decoder_v2 import qam16_constellation
from cpp_bridge import CppBPEvaluator

constellation = qam16_constellation()
snr = 22.0
n = 200
rng = np.random.RandomState(42)
ds = [generate_mimo_sample(16, 16, constellation, snr, rng) for _ in range(n)]

g = Genome(prog_down=[], prog_up=[],
           prog_belief=[Instruction(name='Node.GetCumDist')],
           prog_halt=[Instruction(name='Bool.True')],
           log_constants=np.zeros(N_EVO_CONSTS))

print(f'Pure cum_dist ceiling (16-QAM, SNR={snr}dB, {n} samples)')
print(f'{"max_nodes":>10}  {"BER":>10}  {"flops":>10}  {"bp_calls":>10}  {"time":>6}')
print('-' * 55)
for mn in [500, 1000, 2000, 4000, 8000]:
    cpp = CppBPEvaluator(Nt=16, Nr=16, mod_order=16, max_nodes=mn,
                          flops_max=20_000_000, step_max=10000, max_bp_iters=2)
    t0 = time.time()
    ber, fl, fa, bp = cpp.evaluate_genome(g, ds)
    dt = time.time() - t0
    print(f'{mn:>10}  {ber:>10.5f}  {fl:>10.0f}  {bp:>10.0f}  {dt:>5.1f}s')

cpp_bl = CppBPEvaluator(Nt=16, Nr=16, mod_order=16, max_nodes=500,
                          flops_max=5_000_000, step_max=2000, max_bp_iters=2)
lm, k16, k32 = cpp_bl.evaluate_baselines(ds)
print(f'\nBaselines on same data: LMMSE={lm:.5f}  KB16={k16:.5f}  KB32={k32:.5f}')
