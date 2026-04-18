"""Quick check that template/slot changes took effect."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolution.ir_pool import BP_TEMPLATE, STACK_TEMPLATE, SLOT_DEFAULTS

print("=== BP_TEMPLATE (first 300 chars) ===")
print(BP_TEMPLATE[:300])
print()
print("=== STACK fallback (last 300 chars) ===")
print(STACK_TEMPLATE[-300:])
print()
print("=== AMP onsager ===")
amp_src = SLOT_DEFAULTS['amp_iterate']
idx = amp_src.find('onsager')
print(amp_src[idx:idx+200])
print()
print("=== expand interf ===")
exp_src = SLOT_DEFAULTS['expand']
idx = exp_src.find('j = ')
print(exp_src[idx:idx+100])
