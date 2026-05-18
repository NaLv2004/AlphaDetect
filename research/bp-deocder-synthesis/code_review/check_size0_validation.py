"""Re-run python+cpp validators against the size-0 individuals'
PRE-DCE programs (loaded from the dump) to see how they pass."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpp_seeder"))

from pushgp.genome import dict_to_instruction, N_EVO_CONSTS
from pushgp.validators import validate_v2c, validate_c2v
import pushgp_cpp_seeder as M

dump = json.loads(
    (ROOT / "code_review" / "_dce_dump.init.json").read_text())
dump_g = json.loads(
    (ROOT / "code_review" / "_dce_dump.gen0.json").read_text())

evo = np.ones(N_EVO_CONSTS, dtype=np.float64)  # k = 10**0 = 1.0
deg = 8

cases = []
for tag, src in [("init", dump), ("gen0", dump_g)]:
    for side_key in ("v", "c"):
        for ent in src[side_key]:
            if ent["size_after"] == 0 and ent["size_before"] > 0:
                cases.append((tag, side_key, ent))

for tag, side_key, ent in cases:
    prog = [dict_to_instruction(d) for d in ent["before"]]
    side = "v2c" if side_key == "v" else "c2v"
    print(f"\n=== {tag} {side} #{ent['i']} size={ent['size_before']} ===")

    # Python validator
    rng = np.random.default_rng(42)
    if side == "v2c":
        ok_py, why_py = validate_v2c(prog, rng=rng, evo_consts=evo,
                                     num_configs=3, num_permutations=5, deg=deg)
    else:
        ok_py, why_py = validate_c2v(prog, rng=rng, evo_consts=evo,
                                     num_configs=3, num_permutations=5, deg=deg)
    print(f"  Python validate_{side}: ok={ok_py}  reason={why_py!r}")

    # C++ validator (same seed seq the seeder uses; just try a few seeds)
    h = M.build_program(ent["before"])
    pass_cnt = fail_reasons = 0
    fails = []
    n_trials = 1000
    for s in range(n_trials):
        ok_cpp, reason = M.validate_random(h, side, evo, deg, 3, 5, s * 7919 + 1)
        if ok_cpp:
            pass_cnt += 1
        else:
            fails.append(reason)
    print(f"  C++  validate_random:    pass {pass_cnt}/{n_trials}")
    if fails:
        from collections import Counter
        for r, c in Counter(fails).most_common(3):
            print(f"    fail reason ({c}): {r}")
