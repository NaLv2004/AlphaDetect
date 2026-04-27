param([string]$Repo)
Set-Location $Repo
while ($true) {
  $now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  $alive = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*train_gnn.py*' }
  $childCount = @(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq 43080 }).Count
  $py = @"
import json, pathlib
p=pathlib.Path('results/gnn_training/training_log.jsonl')
rows=[]
if p.exists():
    for line in p.read_text().splitlines():
        try: rows.append(json.loads(line))
        except Exception: pass
start=0
for i,r in enumerate(rows):
    if r.get('gen')==1 and r.get('pop_size')==120:
        start=i
cur=rows[start:]
A=V=E=EF=VF=AF=NB=0
last='none'
for r in cur:
    se=((r.get('engine_stats') or {}).get('slot_evo') or {})
    if not se: continue
    a=se.get('n_attempted',0); v=se.get('n_validated',0); e=se.get('n_evaluated',0)
    A+=a; V+=v; E+=e; EF+=se.get('n_eval_failed',0); VF+=se.get('n_validate_failed',0); AF+=se.get('n_apply_failed',0); NB+=se.get('n_noop_behavior',0)
    last=f"gen={r.get('gen')} a={a} v={v} e={e} ef={se.get('n_eval_failed',0)} vf={se.get('n_validate_failed',0)}"
print(f"rows={len(cur)} last={last} total_a={A} total_v={V} total_e={E} eval_failed={EF} noop={NB} val_fail={VF} apply_fail={AF} valid_att={(V/A if A else 0):.3f} eval_valid={(E/V if V else 0):.3f} eval_att={(E/A if A else 0):.3f}")
"@
  $summary = $py | conda run -n AutoGenOld python - 2>$null
  Add-Content -Path 'code_review\train_monitor_20260426_002927\monitor_summary.log' -Value "$now alive=$($alive.Count) childCount=$childCount $summary"
  Start-Sleep -Seconds 60
}
