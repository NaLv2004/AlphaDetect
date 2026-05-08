"""Parse Phase 3b graft visualisations log: list all entries with verdict / region_size / host / donor,
then print full text of EFFECTIVE/REASONABLE entries with the largest region_size."""
import re
import sys
from pathlib import Path

LOG = Path("results/gnn_training/graft_visualizations.log")

text = LOG.read_text(encoding="utf-8", errors="replace")

# Strip ANSI color codes
ANSI = re.compile(r"\x1b\[[0-9;]*m")
text = ANSI.sub("", text)

# Split entries by "[#N]" markers
entries = re.split(r"^={70,}\n(?=\[#\d+\])", text, flags=re.MULTILINE)

records = []
for e in entries:
    m_head = re.search(r"\[#(\d+)\]\s+child=(\S+)\s+host=(\S+)\s+donor=(\S+)", e)
    if not m_head:
        continue
    m_meta = re.search(
        r"verdict=(\S+)\s+child_score=([-0-9.]+)\s+host_score=([-0-9.]+)\s+child_SER=([-0-9.]+)\s+behavior_change=([-0-9.]+)\s+region_size=(\d+)",
        e,
    )
    if not m_meta:
        continue
    records.append({
        "n": int(m_head.group(1)),
        "child": m_head.group(2),
        "host": m_head.group(3),
        "donor": m_head.group(4),
        "verdict": m_meta.group(1),
        "child_score": float(m_meta.group(2)),
        "host_score": float(m_meta.group(3)),
        "child_SER": float(m_meta.group(4)),
        "beh_change": float(m_meta.group(5)),
        "region_size": int(m_meta.group(6)),
        "text": e,
    })

print(f"Parsed {len(records)} entries.\n")

# Counts by verdict
from collections import Counter
ctr = Counter(r["verdict"] for r in records)
print("Verdict counts:", dict(ctr))

# Region size summary by verdict
import statistics as stats
for v in sorted(set(r["verdict"] for r in records)):
    sizes = [r["region_size"] for r in records if r["verdict"] == v]
    print(f"  {v:18s} n={len(sizes):3d} region_size: min={min(sizes)} max={max(sizes)} mean={stats.mean(sizes):.2f} median={stats.median(sizes)}")

# Table sorted by region_size desc, with verdict filter
print("\n--- All entries sorted by region_size DESC ---")
print(f"{'#':>3}  {'verdict':18s}  {'rsz':>4}  {'SER':>8}  {'beh':>6}  host -> donor")
for r in sorted(records, key=lambda x: -x["region_size"]):
    print(f"{r['n']:>3}  {r['verdict']:18s}  {r['region_size']:>4}  {r['child_SER']:>8.4f}  {r['beh_change']:>6.3f}  {r['host']} -> {r['donor']}")

# Detailed dump: top-3 EFFECTIVE by region_size, top-3 REASONABLE by region_size
def dump_top(verdict_set, k, label):
    sel = [r for r in records if r["verdict"] in verdict_set]
    sel.sort(key=lambda x: -x["region_size"])
    print("\n" + "#" * 80)
    print(f"# TOP-{k} {label} ENTRIES BY region_size")
    print("#" * 80)
    for r in sel[:k]:
        print("\n" + "=" * 80)
        print(f"[#{r['n']}]  {r['verdict']}  region_size={r['region_size']}  host={r['host']}  donor={r['donor']}")
        print(f"  child_SER={r['child_SER']:.4f}  beh={r['beh_change']:.3f}  child_score={r['child_score']:.4f}  host_score={r['host_score']:.4f}")
        print("=" * 80)
        # Print only the region marked lines + a window around them
        body = r["text"]
        # Show body up to a reasonable cap (full IR can be large)
        if len(body) > 40000:
            body = body[:40000] + "\n[...TRUNCATED at 40 KB...]"
        print(body)

dump_top({"EFFECTIVE"}, 3, "EFFECTIVE")
dump_top({"REASONABLE"}, 3, "REASONABLE")
