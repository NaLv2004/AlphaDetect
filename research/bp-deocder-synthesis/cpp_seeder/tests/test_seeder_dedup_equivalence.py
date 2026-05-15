"""Phase B/C check: bit-identical Python vs C++ behavioral fingerprint.

For a batch of randomly generated valid programs, both sides must
produce the same 32-entry fingerprint string. Failure means the
Py/C++ panel constants drifted or the formatting diverged.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # cpp_seeder
sys.path.insert(0, str(ROOT.parent))           # bp-deocder-synthesis (for pushgp)
sys.path.insert(0, str(ROOT))                  # cpp_seeder (for the .pyd)

import pushgp_cpp_seeder as M
from pushgp.evolution import _behav_fingerprint
from pushgp.serialize import dict_to_program


def main(n_per_side: int = 32) -> int:
    n_mismatch = 0
    n_total = 0
    for side in ("v2c", "c2v"):
        # Harvest valid programs (no dedup, no seen set: we want all
        # candidates including potentially duplicates).
        handles, attempts, fps_cpp = M.parallel_seed(
            side=side,
            n_target=n_per_side,
            max_attempts=10_000_000,
            threads=4,
            chunk_attempts=2000,
            min_size=4,
            max_size=20,
            deg=8,
            num_configs=3,
            num_permutations=5,
            base_seed=12345,
            seen_fingerprints=[],
        )
        if len(handles) < n_per_side:
            print(f"[{side}] only got {len(handles)} valid; aborting", flush=True)
            return 2
        for i in range(n_per_side):
            n_total += 1
            fp_cpp = fps_cpp[i]
            fp_cpp2 = M.compute_behav_fp(side, handles[i])
            assert fp_cpp == fp_cpp2, (
                f"[{side}] C++ parallel_seed fp != C++ compute_behav_fp:\n"
                f"  parallel: {fp_cpp[:80]}\n  compute:  {fp_cpp2[:80]}"
            )
            prog_py = dict_to_program(handles[i].to_dict())
            fp_py = _behav_fingerprint(side, prog_py)
            if fp_py != fp_cpp:
                n_mismatch += 1
                print(f"[{side}] MISMATCH idx={i}\n  py : {fp_py[:120]}\n  cpp: {fp_cpp[:120]}",
                      flush=True)
    print(f"[seeder_dedup_eq] checked {n_total} programs, {n_mismatch} mismatches",
          flush=True)
    return 1 if n_mismatch else 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    sys.exit(main(n))
