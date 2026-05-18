"""Inspect the JSON dumps written by the PUSHGP_DCE_DUMP hook in
pushgp/evolution.py._apply_dce_bp.  Prints every individual that
reduced to size 0 with its full pre-DCE program."""
import json
import sys
from pathlib import Path

DUMP_DIR = Path(r"D:\ChannelCoding\RCOM\AlphaDetect\research\bp-deocder-synthesis\code_review")
files = sorted(DUMP_DIR.glob("_dce_dump.*.json"))
if not files:
    print("no dump files found")
    sys.exit(1)

def render_prog(prog):
    lines = []
    for j, ins in enumerate(prog):
        nm = ins.get("name", "?")
        cb = ins.get("code_block")
        if cb is not None:
            inner = " ".join(x.get("name", "?") for x in cb)
            lines.append(f"  {j:2d}: {nm} [ {inner} ]")
        else:
            lines.append(f"  {j:2d}: {nm}")
    return "\n".join(lines)

for f in files:
    rec = json.loads(f.read_text())
    tag = rec["tag"]
    print(f"\n========== {tag}  ({f.name}) ==========")
    for side_key in ("v", "c"):
        for ent in rec[side_key]:
            if ent["size_after"] == 0:
                print(f"\n--- {side_key.upper()}2x #{ent['i']}  size {ent['size_before']} -> 0 "
                      f"passes={ent['passes']} fp_evals={ent['fp_evals']} ---")
                print("BEFORE:")
                print(render_prog(ent["before"]))
                print("AFTER: <empty>")
