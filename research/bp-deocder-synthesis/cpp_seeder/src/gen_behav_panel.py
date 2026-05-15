"""Generate the 32-entry behavioral panels used by both Python and C++ dedup.

Run once and paste output into:
  - pushgp/evolution.py   (Python literals)
  - cpp_seeder/src/behav_panel.hpp   (C++ literals)

Determinism: seeded with 0xBE4AC1D so this script always produces the same
panels.  The same byte-identical doubles must appear in both languages.
"""
import numpy as np

rng = np.random.default_rng(0xBE4AC1D)
DEG = 8
N = 32

v_Lv  = rng.uniform(-5.0, 5.0,  size=N).astype(np.float64)
v_inc = rng.uniform(-6.0, 6.0,  size=(N, DEG-1)).astype(np.float64)
c_inc = rng.uniform(-6.0, 6.0,  size=(N, DEG-1)).astype(np.float64)

# Inject 3 fixed extreme rows so degenerate "output 0 / output input"
# programs are revealed by the fingerprint.
v_inc[0]  = np.zeros(DEG-1)
v_inc[1]  = np.array([ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0])
v_inc[2]  = np.array([ 5.0, -5.0,  5.0, -5.0,  5.0, -5.0,  5.0])
v_Lv[0]   = 0.0
v_Lv[1]   = 1.0
v_Lv[2]   = -1.0
c_inc[0]  = np.zeros(DEG-1)
c_inc[1]  = np.array([ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0])
c_inc[2]  = np.array([ 5.0, -5.0,  5.0, -5.0,  5.0, -5.0,  5.0])

def f(x):  return repr(float(x))

print("# ===== Python literals (paste into pushgp/evolution.py) =====")
print("_BEHAV_PANEL_V2C_LV = np.array([")
for v in v_Lv: print(f"    {f(v)},")
print("], dtype=np.float64)")
print("_BEHAV_PANEL_V2C_INC = np.array([")
for row in v_inc: print("    [" + ", ".join(f(x) for x in row) + "],")
print("], dtype=np.float64)")
print("_BEHAV_PANEL_C2V_INC = np.array([")
for row in c_inc: print("    [" + ", ".join(f(x) for x in row) + "],")
print("], dtype=np.float64)")

print()
print("// ===== C++ literals (paste into cpp_seeder/src/behav_panel.hpp) =====")
print("static constexpr int BEHAV_PANEL_N   = 32;")
print("static constexpr int BEHAV_PANEL_DEG = 8;")
print("static constexpr double BEHAV_PANEL_V2C_LV[32] = {")
for v in v_Lv: print(f"    {f(v)},")
print("};")
print("static constexpr double BEHAV_PANEL_V2C_INC[32][7] = {")
for row in v_inc: print("    {" + ", ".join(f(x) for x in row) + "},")
print("};")
print("static constexpr double BEHAV_PANEL_C2V_INC[32][7] = {")
for row in c_inc: print("    {" + ", ".join(f(x) for x in row) + "},")
print("};")
