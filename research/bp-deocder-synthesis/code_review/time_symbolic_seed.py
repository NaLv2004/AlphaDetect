"""Time symbolic-mode seeding to gauge whether pop=20 is reachable."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as S


def run(side, n_target, mode, max_attempts):
    print(f"\n=== side={side}  n_target={n_target}  mode={mode}  max_attempts={max_attempts:,} ===")
    t0 = time.perf_counter()
    try:
        progs, attempts, _ = S.parallel_seed(
            side=side, n_target=n_target, max_attempts=max_attempts,
            threads=8, chunk_attempts=500,
            min_size=4, max_size=16, deg=8,
            num_configs=3, num_permutations=5, num_evo_panels=2,
            base_seed=1, validator_mode=mode)
    except Exception as e:
        print(f"  error: {e}")
        return
    dt = time.perf_counter() - t0
    print(f"  found {len(progs)}/{n_target} in {dt:.1f}s ({attempts:,} attempts; "
          f"{attempts/max(1,dt):,.0f} attempts/s)")
    if progs:
        print("  sample sizes:", sorted([len(h.to_dict()) for h in progs])[:10])


if __name__ == "__main__":
    # probe baseline
    run("c2v", 20, "probe", 5_000_000)
    # symbolic seeding
    run("c2v", 20, "symbolic", 20_000_000)
    # v2c symbolic
    run("v2c", 20, "symbolic", 20_000_000)
