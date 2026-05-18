"""Smoke test: verify symbolic_validate API works on a few hand-crafted programs."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))

import pushgp_cpp_seeder as S


def mk(*ops):
    """Build a flat program from opcode names."""
    return S.build_program([{"name": o} for o in ops])


CASES_C2V = [
    # name, prog, expected_ok, why
    ("exp_abs_sign_x_last",
        mk("Env.GetIncomingVec", "FVec.PopBack", "Float.Sign", "Float.Abs", "Float.Exp"),
        False, "constant after abs∘sign; not symmetric and not odd"),
    ("popback_only",
        mk("Env.GetIncomingVec", "FVec.PopBack"),
        False, "depends on single position only → not sym"),
    ("sum_via_len", S.build_program([
        {"name": "Env.GetIncomingVec"},
        {"name": "Float.Const0"},  # acc on float stack
        {"name": "FVec.Dup"},
        {"name": "FVec.Len"},      # push len onto int
        {"name": "Exec.DoTimes", "code_block": [
            {"name": "Int.Pop"},     # pop loop var
            {"name": "FVec.PopBack"},
            {"name": "Float.Add"},
        ]},
    ]), True, "sum is sym and odd"),
    ("const_only",
        mk("Float.Const1"),
        False, "no dep on incoming"),
    ("single_x_neg",
        mk("Env.GetIncomingVec", "FVec.PopBack", "Float.Neg"),
        False, "only one X → not sym"),
]


def main():
    print("symbolic_expr_table_size at start:", S.symbolic_expr_table_size())
    for name, prog, exp_ok, why in CASES_C2V:
        ok, reason = S.symbolic_validate_c2v(prog, 8, 0)
        status = "PASS" if ok == exp_ok else "FAIL"
        print(f"  [{status}] {name}: ok={ok} expected={exp_ok} reason='{reason}'  # {why}")
    print("symbolic_expr_table_size at end:", S.symbolic_expr_table_size())


if __name__ == "__main__":
    main()
