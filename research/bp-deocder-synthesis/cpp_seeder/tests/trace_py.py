"""Step-by-step trace of one program through the Python VM."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from pushgp.genome import program_to_list
from pushgp.random_program import RandomProgramGenerator, V2C_INSTR, C2V_INSTR
from pushgp.vm import VM
from pushgp.instructions import HANDLERS

rng = np.random.default_rng(12345)
rpg = RandomProgramGenerator(np.random.default_rng(12345))
deg = 8
target_k = 27
target_j = 1

for k in range(target_k + 1):
    side = "v2c" if (k % 2 == 0) else "c2v"
    instr = V2C_INSTR if side == "v2c" else C2V_INSTR
    size = int(rng.integers(4, 25))
    prog = rpg.random_program(instr, size, size + 5)
    inputs = [(rng.uniform(-3.0, 3.0, size=deg-1).astype(np.float64),
               float(rng.uniform(-2.0, 2.0)),
               int(rng.integers(0, 8)),
               rng.uniform(-3.0, 3.0, size=8).astype(np.float64)) for _ in range(5)]
    if k != target_k: continue
    inc, L_v, it, evo = inputs[target_j]

    vm = VM()
    vm.state.ctx_channel_llr = float(L_v)
    vm.state.ctx_has_channel_llr = (side == "v2c")
    vm.state.ctx_incoming = inc.astype(np.float64, copy=True)
    vm.state.ctx_deg = int(deg)
    vm.state.ctx_iter = int(it)
    vm.state.ctx_max_iter = 25
    vm.state.ctx_evo_constants = evo
    if side == "v2c":
        vm.state.floats.push(float(L_v))
        vm.state.ints.push(0); vm.state.ints.push(int(deg))
        vm.state.ints.push(int(it)); vm.state.ints.push(25)
        vm.state.fvecs.push(inc.astype(np.float64, copy=True))
    else:
        vm.state.ctx_has_channel_llr = False
        vm.state.ints.push(0); vm.state.ints.push(int(deg))
        vm.state.ints.push(int(it)); vm.state.ints.push(25)
        vm.state.fvecs.push(inc.astype(np.float64, copy=True))

    def dump(label):
        f = list(vm.state.floats._data)
        i = list(vm.state.ints._data)
        b = list(vm.state.bools._data)
        fv_top_len = vm.state.fvecs.peek().size if vm.state.fvecs.peek() is not None else None
        print(f"  -> floats={[round(x, 4) for x in f]}  ints={i}  bools={b}  fvec_top_len={fv_top_len}")

    dump("INIT")
    for step, ins in enumerate(prog):
        print(f"step {step}: {ins.name}")
        h = HANDLERS.get(ins.name)
        if h is None: print("  no handler"); continue
        try:
            h(vm, ins)
        except Exception as e:
            print("  EX:", e); break
        dump("after")
    top = vm.state.floats.peek()
    print("FINAL TOP:", top)
    break
