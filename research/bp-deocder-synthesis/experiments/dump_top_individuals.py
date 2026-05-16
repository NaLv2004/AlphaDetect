import json, sys, pathlib
DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "logged_evolution" / "fromscratch_pop100_dedup"
recs = [json.loads(l) for l in (DIR / "individuals.jsonl").open(encoding="utf-8")]
print("total individuals logged:", len(recs))
gens = sorted({r["gen"] for r in recs})
print("gens present:", gens)
last = max(gens)


def show(tag, sel):
    print("\n===== " + tag + " =====")
    for r in sel:
        print(
            "\n--- gen={g} pair={p} fit={f:+.5f}  v_size={vs} c_size={cs}  v_depth={vd} c_depth={cd}".format(
                g=r["gen"],
                p=r["idx"],
                f=r["fitness"],
                vs=r["v2c_size"],
                cs=r["c2v_size"],
                vd=r["v2c_max_depth"],
                cd=r["c2v_max_depth"],
            )
        )
        print("  log_constants:", r["log_constants"])
        v = r["v2c"]
        c = r["c2v"]
        print("  -- V2C program ({} ops) --".format(len(v)))
        for i, op in enumerate(v):
            print("    {:2d}: {}".format(i, op))
        print("  -- C2V program ({} ops) --".format(len(c)))
        for i, op in enumerate(c):
            print("    {:2d}: {}".format(i, op))


show("BEST OVERALL", sorted(recs, key=lambda r: r["fitness"])[:1])
show("TOP 3 OF GEN {}".format(last), sorted([r for r in recs if r["gen"] == last], key=lambda r: r["fitness"])[:3])
