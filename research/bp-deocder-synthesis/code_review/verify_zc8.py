import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pushgp_ldpc.eval import FitnessConfig, physical_code_rate
from pushgp_ldpc.eval_cpp import evaluate_genome_cpp_ber
from pushgp_ldpc.adapter import oms_seed_genome
from pushgp_ldpc.baselines import uncoded_rate1_baseline, channel_hard_baseline
from ldpc_5g import build_parity

par = build_parity(bgn=2, set_idx=1, zc=8)
print(f"BG2 set1 Zc=8: N={par.cols} M={par.rows} R={physical_code_rate(par)}")
snrs = (2.0, 3.0, 4.0)
cfg = FitnessConfig(par=par, snr_list=snrs, n_frames_per_snr=100, max_iter=20)
print(f"R_eff={cfg.effective_code_rate}")
t0 = time.time()
m = evaluate_genome_cpp_ber(oms_seed_genome(), cfg)
print(f"OMS eval took {time.time()-t0:.2f}s")
print("OMS     BER:", [f"{b:.3e}" for b in m.ber_per_snr])
ch = channel_hard_baseline(cfg)
r1 = uncoded_rate1_baseline(cfg, bits_per_snr=100 * par.cols)
print("ch-hard BER:", [f"{b:.3e}" for b in ch["ber_per_snr"]])
print("R=1     BER:", [f"{b:.3e}" for b in r1["ber_per_snr"]])
