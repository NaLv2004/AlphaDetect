import json
from pathlib import Path

from main import evaluate_on_target_system
from vm import Instruction


def main():
    output_path = Path(__file__).resolve().parents[1] / 'results' / 'candidate_eval_compare.json'
    experiments = {
        'evolved_exact': [
            Instruction('Node.GetDistance'),
            Instruction('Node.GetState1'),
            Instruction('Float.Mul'),
            Instruction('Mat.VecMul'),
            Instruction('Float.ConstHalf'),
            Instruction('Float.Add'),
        ],
        'evolved_simplified': [
            Instruction('Node.GetDistance'),
            Instruction('Node.GetState1'),
            Instruction('Float.Mul'),
            Instruction('Float.ConstHalf'),
            Instruction('Float.Add'),
        ],
        'distance_only': [
            Instruction('Node.GetDistance'),
        ],
    }

    results = {}
    for name, program in experiments.items():
        print(f"Evaluating {name}...", flush=True)
        results[name] = evaluate_on_target_system(
            program,
            eval_nt=16,
            eval_nr=32,
            mod_order=16,
            n_trials=12,
            snr_dbs=[12.0, 14.0, 16.0],
            max_nodes=400,
            flops_max=100000,
        )
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({name: results[name]}, indent=2, ensure_ascii=False), flush=True)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == '__main__':
    main()