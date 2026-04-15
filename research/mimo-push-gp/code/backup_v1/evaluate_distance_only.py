import json
from pathlib import Path

from main import evaluate_on_target_system
from vm import Instruction


def main():
    output_path = Path(__file__).resolve().parents[1] / 'results' / 'distance_only_eval.json'
    results = []
    for snr_db in [12.0, 14.0, 16.0]:
        print(f"Evaluating distance_only at {snr_db:.1f} dB...", flush=True)
        snr_result = evaluate_on_target_system(
            [Instruction('Node.GetDistance')],
            eval_nt=16,
            eval_nr=32,
            mod_order=16,
            n_trials=8,
            snr_dbs=[snr_db],
            max_nodes=400,
            flops_max=100000,
        )
        results.extend(snr_result)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(snr_result, indent=2, ensure_ascii=False), flush=True)

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == '__main__':
    main()