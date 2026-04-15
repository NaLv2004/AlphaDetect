"""Check latest results from evolution runs."""
import json, glob, os, sys

os.chdir(os.path.join(os.path.dirname(__file__), '..', 'results'))

for pattern in ['bp_v3_*', 'bp_v4_*', 'bp_truebp_1_gen340*']:
    files = sorted(glob.glob(pattern))
    if files:
        f = files[-1]
        d = json.load(open(f))
        print(f'=== {f} ===')
        if 'best_fitness' in d:
            bf = d['best_fitness']
            ber = bf.get('ber', -1)
            flops = bf.get('avg_flops', -1)
            faults = bf.get('frac_faults', -1)
            bp = bf.get('bp_updates', -1)
            print(f'  BER={ber:.5f}  flops={flops:.0f}  faults={faults}  bp={bp}')
        if 'best_programs' in d:
            bp = d['best_programs']
            for k in ['F_down', 'F_up', 'F_belief', 'H_halt']:
                print(f'  {k}: {bp.get(k, "?")}')
        if 'eval_results' in d:
            for r in d['eval_results']:
                snr = r['snr_db']
                evo = r['evolved_ber']
                k16 = r.get('kbest16_ber', 0)
                k32 = r.get('kbest32_ber', 0)
                lm = r.get('lmmse_ber', 0)
                print(f'  SNR={snr:5.1f}  Evo={evo:.5f}  LMMSE={lm:.5f}  KB16={k16:.5f}  KB32={k32:.5f}')
        print()
