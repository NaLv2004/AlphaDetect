"""Validate skeleton library pool building."""
import sys
sys.path.insert(0, ".")

from evolution.ir_pool import build_ir_pool, _DETECTOR_SPECS, SLOT_DEFAULTS, compile_detector_template

failed = []
for spec in _DETECTOR_SPECS:
    try:
        compile_detector_template(spec)
    except Exception as e:
        failed.append((spec.algo_id, type(e).__name__, str(e)[:200]))

if failed:
    for f in failed:
        print(f"FAIL: {f[0]}: {f[1]}: {f[2]}")
else:
    print("ALL TEMPLATES COMPILE SUCCESSFULLY")

pool = build_ir_pool()
print(f"Pool size: {len(pool)} / {len(_DETECTOR_SPECS)} specs, slot defaults: {len(SLOT_DEFAULTS)}")
