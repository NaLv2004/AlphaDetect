"""1000+ random program parity test: symbolic vs probe validator.

Generates random PushGP programs (including nested Exec.* blocks), runs
both validators on each, reports agreement / disagreement breakdown,
timing, and opaque rate.  Side: c2v (no LV).
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))

import numpy as np
import pushgp_cpp_seeder as S

OP_NAMES = S.all_op_names()
EXEC_OPS = {"Exec.If", "Exec.When", "Exec.DoTimes", "Exec.DoRange", "Exec.While"}
DOUBLE_BLOCK_OPS = {"Exec.If"}

# Filter out Exec ops from the leaf opcode pool; we'll add them with code_blocks.
LEAF_OPS = [n for n in OP_NAMES if n not in EXEC_OPS]


def random_instruction(rng, depth, max_depth):
    use_exec = depth < max_depth and rng.random() < 0.10
    if not use_exec:
        return {"name": rng.choice(LEAF_OPS)}
    op = rng.choice(list(EXEC_OPS))
    body_len = rng.randint(1, 5)
    block = [random_instruction(rng, depth + 1, max_depth) for _ in range(body_len)]
    ins = {"name": op, "code_block": block}
    if op in DOUBLE_BLOCK_OPS:
        body2_len = rng.randint(1, 4)
        ins["code_block2"] = [random_instruction(rng, depth + 1, max_depth) for _ in range(body2_len)]
    return ins


def random_program(rng, min_len=3, max_len=25, max_depth=2):
    n = rng.randint(min_len, max_len)
    return [random_instruction(rng, 0, max_depth) for _ in range(n)]


def main(N=1200, deg=8, seed=42):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed + 1)

    sym_ok = 0
    probe_ok = 0
    agree = 0
    disagree_sym_only = []   # symbolic ok, probe reject
    disagree_probe_only = []  # probe ok, symbolic reject
    opaque_count = 0
    sym_times = []
    probe_times = []

    for i in range(N):
        prog_list = random_program(rng, max_len=20)
        try:
            ph = S.build_program(prog_list)
        except Exception:
            continue

        t0 = time.perf_counter()
        s_ok, s_reason = S.symbolic_validate_c2v(ph, deg, 0)
        sym_times.append(time.perf_counter() - t0)

        # Probe validator with default settings.
        # validate_random(prog, side, evo, deg, num_configs, num_perms, num_evo_panels, seed)
        evo = np_rng.uniform(-1.0, 1.0, size=8).astype(np.float64)
        t0 = time.perf_counter()
        p_ok, p_reason = S.validate_random(ph, "c2v", evo, deg, 32, 8, 4, rng.randint(1, 1 << 30))
        probe_times.append(time.perf_counter() - t0)

        if s_ok:
            sym_ok += 1
        if p_ok:
            probe_ok += 1
        if "opaque" in s_reason:
            opaque_count += 1
        if s_ok == p_ok:
            agree += 1
        elif s_ok and not p_ok:
            if len(disagree_sym_only) < 5:
                disagree_sym_only.append((i, p_reason, s_reason))
        else:
            if len(disagree_probe_only) < 10:
                disagree_probe_only.append((i, p_reason, s_reason, prog_list))

    print(f"\n=== {N} random programs, side=c2v, deg={deg} ===")
    print(f"symbolic accept: {sym_ok}/{N} ({sym_ok/N:.1%})")
    print(f"probe    accept: {probe_ok}/{N} ({probe_ok/N:.1%})")
    print(f"opaque (symbolic): {opaque_count}/{N} ({opaque_count/N:.1%})")
    print(f"agreement:    {agree}/{N} ({agree/N:.1%})")
    print(f"sym-only accepts (probe rejects): {len(disagree_sym_only)}  (showing up to 5)")
    print(f"probe-only accepts (sym rejects): {len(disagree_probe_only)}  (showing up to 10)")
    print(f"sym  time: mean={1e3*np.mean(sym_times):.3f} ms  p99={1e3*np.quantile(sym_times,0.99):.3f} ms")
    print(f"probe time: mean={1e3*np.mean(probe_times):.3f} ms  p99={1e3*np.quantile(probe_times,0.99):.3f} ms")
    print(f"symbolic_expr_table_size: {S.symbolic_expr_table_size()}")

    if disagree_sym_only:
        print("\n--- sym accepts, probe rejects (probe may have false-negatives or sym false-positives) ---")
        for i, p_r, s_r in disagree_sym_only:
            print(f"  #{i}: probe='{p_r}'   sym='{s_r}'")
    if disagree_probe_only:
        print("\n--- probe accepts, sym rejects (more interesting — probe may have false-positives) ---")
        for i, p_r, s_r, prog in disagree_probe_only[:5]:
            ops = [ins["name"] for ins in prog]
            print(f"  #{i}: probe='{p_r}'   sym='{s_r}'   prog={ops}")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    main(N=N)
