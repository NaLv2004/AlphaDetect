"""Verify (a) odd hand-crafted v2c program passes symbolic, (b) measure
acceptance rate of pure symbolic seeding for v2c."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M


def hand_v2c_sum_plus_lv():
    """Compute LV + Sum(x_i). Odd in (X, LV), permutation-invariant in X,
    depends on every X_i and on LV. Should pass symbolic_validate_v2c."""
    prog_dict = [
        {"name": "Float.Const0"},        # Float: [LV, 0]
        {"name": "Env.GetIncomingVec"},  # FVec: [X]
        {"name": "FVec.Dup"},
        {"name": "FVec.Len"},
        {"name": "Exec.DoTimes", "code_block": [
            {"name": "Int.Pop"},
            {"name": "FVec.PopBack"},
            {"name": "Float.Add"},
        ]},
        # Float: [LV, Sum(x_i)]
        {"name": "Float.Add"},           # LV + Sum(x_i)
    ]
    return M.build_program(prog_dict)


def main():
    p = hand_v2c_sum_plus_lv()
    print(f"hand-crafted v2c (LV+Sum(X)) length={len(p)}", flush=True)
    ok, reason = M.symbolic_validate_v2c(p, 8, 0)
    print(f"  v2c symbolic: ok={ok}  reason={reason!r}", flush=True)
    assert ok, f"hand-crafted v2c LV+Sum(X) should pass but got: {reason!r}"

    print("\n[1] timing PROBE-only seed v2c n=50:", flush=True)
    t0 = time.time()
    progs, att, _ = M.parallel_seed(side="v2c", n_target=50, max_attempts=5_000_000,
        threads=8, chunk_attempts=500, min_size=4, max_size=16, deg=8,
        num_configs=3, num_permutations=5, num_evo_panels=2,
        base_seed=11, validator_mode="probe")
    dt = time.time()-t0
    print(f"  got {len(progs)} in {att:,} attempts, wall {dt:.2f}s, rate {att/dt/1000:.1f}k/s", flush=True)

    print("\n[2] timing SYMBOLIC-only seed v2c n=20:", flush=True)
    t0 = time.time()
    progs2, att2, _ = M.parallel_seed(side="v2c", n_target=20, max_attempts=20_000_000,
        threads=8, chunk_attempts=500, min_size=4, max_size=16, deg=8,
        num_configs=3, num_permutations=5, num_evo_panels=2,
        base_seed=12, validator_mode="symbolic")
    dt = time.time()-t0
    print(f"  got {len(progs2)} in {att2:,} attempts, wall {dt:.2f}s, rate {att2/dt/1000:.1f}k/s", flush=True)

    print("\n[3] timing SYMBOLIC-only seed c2v n=20:", flush=True)
    t0 = time.time()
    progs3, att3, _ = M.parallel_seed(side="c2v", n_target=20, max_attempts=20_000_000,
        threads=8, chunk_attempts=500, min_size=4, max_size=16, deg=8,
        num_configs=3, num_permutations=5, num_evo_panels=2,
        base_seed=13, validator_mode="symbolic")
    dt = time.time()-t0
    print(f"  got {len(progs3)} in {att3:,} attempts, wall {dt:.2f}s, rate {att3/dt/1000:.1f}k/s", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
