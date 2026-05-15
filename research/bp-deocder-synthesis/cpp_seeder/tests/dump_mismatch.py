"""Dump first failing program for inspection."""
import struct, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from pushgp.genome import program_to_list
from pushgp.random_program import RandomProgramGenerator, V2C_INSTR, C2V_INSTR
from pushgp.vm import VM
import pushgp_cpp_seeder as M

rng = np.random.default_rng(12345)
rpg = RandomProgramGenerator(np.random.default_rng(12345))
deg = 8

def py_run(prog, side, incoming, L_v, deg, iter_idx, evo):
    vm = VM()
    vm.state.ctx_channel_llr = float(L_v)
    vm.state.ctx_has_channel_llr = (side == "v2c")
    vm.state.ctx_incoming = incoming.astype(np.float64, copy=True)
    vm.state.ctx_deg = int(deg); vm.state.ctx_iter = int(iter_idx)
    vm.state.ctx_max_iter = 25; vm.state.ctx_noise_var = 1.0
    vm.state.ctx_edge_index = 0; vm.state.ctx_code_rate = 0.5
    vm.state.ctx_evo_constants = evo.astype(np.float64, copy=True)
    if side == "v2c":
        vm.state.floats.push(float(L_v))
        vm.state.ints.push(0); vm.state.ints.push(int(deg))
        vm.state.ints.push(int(iter_idx)); vm.state.ints.push(25)
        vm.state.fvecs.push(incoming.astype(np.float64, copy=True))
    else:
        vm.state.ctx_has_channel_llr = False
        vm.state.ints.push(0); vm.state.ints.push(int(deg))
        vm.state.ints.push(int(iter_idx)); vm.state.ints.push(25)
        vm.state.fvecs.push(incoming.astype(np.float64, copy=True))
    return vm.run(prog)

target_k = int(sys.argv[1]) if len(sys.argv) > 1 else 27
target_j = int(sys.argv[2]) if len(sys.argv) > 2 else 1
n_inputs = 5

for k in range(target_k + 1):
    side = "v2c" if (k % 2 == 0) else "c2v"
    instr = V2C_INSTR if side == "v2c" else C2V_INSTR
    size = int(rng.integers(4, 25))
    prog = rpg.random_program(instr, size, size + 5)
    inputs = []
    for j in range(n_inputs):
        inc = rng.uniform(-3.0, 3.0, size=deg-1).astype(np.float64)
        L_v = float(rng.uniform(-2.0, 2.0))
        it = int(rng.integers(0, 8))
        evo = rng.uniform(-3.0, 3.0, size=8).astype(np.float64)
        inputs.append((inc, L_v, it, evo))
    if k != target_k: continue
    pdict = program_to_list(prog)
    print("side =", side, " size =", len(prog))
    print(json.dumps(pdict, indent=2))
    inc, L_v, it, evo = inputs[target_j]
    print("incoming =", inc.tolist())
    print("L_v =", L_v, " iter =", it, " evo =", evo.tolist())
    cpp_h = M.build_program(pdict)
    py_out = py_run(prog, side, inc, L_v, deg, it, evo)
    cpp_out = M.run_program(cpp_h, side, evo, inc, L_v, deg, it)
    print("py_out:", py_out, " bits:", struct.pack("<d", py_out).hex() if py_out is not None else None)
    print("cpp_out:", cpp_out, " bits:", struct.pack("<d", cpp_out).hex() if cpp_out is not None else None)
    break
