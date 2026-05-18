"""Debug v2c symbolic: build a hand-crafted V2C program that should pass
(depends on all incoming X via sum, depends on LV, is X-permutation invariant,
and is odd under (X->-X, LV->-LV)), and confirm symbolic_validate_v2c accepts it.

Then sample 200 c2v / v2c programs via parallel_seed(probe), run symbolic on
each, and report per-reason rejection counts to see *why* random programs fail
v2c symbolic.
"""
import os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M


def hand_v2c_sum_plus_lv():
    """Program semantics: pop FVec X -> Sum(x), then Float.Mul(LV) -> push.
    Expected: ok=True (odd in X, odd in LV, sym in X, depends on all X+LV)."""
    # Build a minimal program. We need the engine to leave one float on the
    # Float stack at end. Build via build_program from a dict list.
    # Atoms reachable: x_i (initial FVec), LV (initial Float). Operate:
    #   FVec.Sum   -> pops FVec, pushes Float = Sum(x_i)
    #   Float.Mul  -> pops Sum, LV (in some order) -> pushes Sum*LV
    # v2c seed: Float stack already has LV atom (channel LLR).
    # We push 0, get incoming, sum via DoTimes(len), then Mul with LV
    # already at bottom of float stack.
    prog_dict = [
        {"name": "Float.Const0"},        # Float: [LV, 0]
        {"name": "Env.GetIncomingVec"},  # FVec: [X]
        {"name": "FVec.Dup"},            # FVec: [X, X]
        {"name": "FVec.Len"},            # Int += len, FVec: [X, X] (Dup keeps it; Len pops 1 copy)
        {"name": "Exec.DoTimes", "code_block": [
            {"name": "Int.Pop"},
            {"name": "FVec.PopBack"},
            {"name": "Float.Add"},
        ]},
        # After loop: Float: [LV, Sum(x_i)]
        {"name": "Float.Mul"},
    ]
    return M.build_program(prog_dict)


def main():
    p = hand_v2c_sum_plus_lv()
    print(f"hand-crafted v2c program length={len(p)}")
    ok, reason = M.symbolic_validate_v2c(p, 8, 0)
    print(f"  v2c symbolic: ok={ok}  reason={reason!r}")

    print("\nSampling 500 probe-accepted v2c programs and inspecting symbolic verdicts:")
    progs, attempts, _ = M.parallel_seed(
        side="v2c", n_target=500, max_attempts=20_000_000,
        threads=8, chunk_attempts=1000,
        min_size=4, max_size=16, deg=8,
        num_configs=3, num_permutations=5, num_evo_panels=2,
        base_seed=7, validator_mode="probe")
    print(f"  probe found {len(progs)} in {attempts:,} attempts")

    reasons = collections.Counter()
    n_ok = 0
    for h in progs:
        ok, reason = M.symbolic_validate_v2c(h, 8, 0)
        if ok:
            n_ok += 1
            reasons["OK"] += 1
        else:
            # bucketize the reason
            r = reason
            if r.startswith("opaque"):
                # collapse to top-level cause
                key = "opaque:" + r.split(":", 2)[1].split(" ", 1)[0] if ":" in r else "opaque"
            else:
                key = r.split(":")[0].split("(")[0].strip()
            reasons[key] += 1
    print(f"  symbolic accept rate (over probe-accepted): {n_ok}/{len(progs)}")
    print("  reason histogram:")
    for k, v in reasons.most_common():
        print(f"    {v:5d}  {k}")


if __name__ == "__main__":
    main()
