import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from evolution.ir_pool import build_ir_pool

pool = build_ir_pool()

print("="*70)
print("Type-hint vocabulary across all 91 pool genomes")
print("="*70)

global_th = Counter()
per_genome_unknown = []
for g in pool:
    local = Counter()
    for v in g.ir.values.values():
        th = v.type_hint if v.type_hint is not None else "<None>"
        global_th[th] += 1
        local[th] += 1
    n_unknown = local.get("object", 0) + local.get("<None>", 0) + local.get("Any", 0)
    per_genome_unknown.append((g.algo_id, n_unknown, len(g.ir.values)))

print(f"\nTotal distinct type_hint strings: {len(global_th)}")
print(f"Total values across all genomes: {sum(global_th.values())}")
print(f"\nTop 25 by frequency:")
for th, n in global_th.most_common(25):
    print(f"  {th:<30} {n:>6}")

print(f"\nLattice canonicals (vec_f, vec_cx, mat_f, mat_cx, vec_i, etc.):")
LATTICE = {"vec_f","vec_cx","vec_i","mat_f","mat_cx","tensor3_f","tensor3_cx",
           "bool","int","float","cx","void","any","object","node",
           "candidate_list","open_set","mat_decomp","prob_table"}
in_lattice = sum(global_th[t] for t in LATTICE if t in global_th)
out_lattice = sum(n for th, n in global_th.items() if th not in LATTICE)
print(f"  in lattice:    {in_lattice}")
print(f"  out of lattice:{out_lattice}")

print(f"\nNon-lattice strings (top 20):")
non_lattice = Counter({th: n for th, n in global_th.items() if th not in LATTICE})
for th, n in non_lattice.most_common(20):
    print(f"  {th:<30} {n:>6}")

print(f"\nLmmse genome — sample of values with type_hints:")
g = pool[0]
shown = 0
for vid, v in g.ir.values.items():
    if shown >= 25: break
    print(f"  {vid:<8} hint={(v.name_hint or '-')[:18]:<18} type={v.type_hint}")
    shown += 1

print(f"\nUnknown-type fraction per detector (top 10 worst):")
per_genome_unknown.sort(key=lambda x: -x[1]/max(1,x[2]))
for algo, n_unk, n_tot in per_genome_unknown[:10]:
    print(f"  {algo:<25} {n_unk}/{n_tot} = {100*n_unk/n_tot:.0f}%")
