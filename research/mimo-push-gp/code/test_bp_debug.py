"""Debug why BP patterns don't trigger SetScore."""
from vm import MIMOPushVM, Instruction, program_to_oneliner
from bp_decoder import BPStackDecoder, qam16_constellation
from stacks import SearchTreeGraph
from evolution import random_bp_pattern
import numpy as np

rng = np.random.RandomState(42)
const = qam16_constellation()

Nt, Nr = 8, 16
H = (np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)) / np.sqrt(2)
x = const[np.random.randint(len(const), size=Nt)]
nv = 0.1
y = H @ x + np.sqrt(nv/2) * (np.random.randn(Nr) + 1j*np.random.randn(Nr))

# Test a minimal manually-constructed BP program
# This should definitely work:
# 1. Node.GetCumDist pushes a Float
# 2. Node.ForEachSibling([Node.GetCumDist, Node.SetScore]) sets each sibling's score to its cum_dist
manual_prog = [
    Instruction('Node.GetCumDist'),  # push candidate's cum_dist (for correction return)
    Instruction('Node.ForEachSibling', code_block=[
        Instruction('Node.GetCumDist'),  # push this sibling's cum_dist
        Instruction('Node.SetScore'),    # set this sibling's score to cum_dist
    ]),
]

vm = MIMOPushVM(flops_max=2000000, step_max=3000)
dec = BPStackDecoder(Nt=Nt, Nr=Nr, constellation=const, max_nodes=200, vm=vm)
x_hat, flops = dec.detect(H, y, manual_prog, noise_var=nv)
print(f"Manual BP program: bp_updates={dec.bp_updates}, flops={flops}")
print(f"  tree nodes: {dec.search_tree.node_count()}")
print(f"  frontier: {dec.search_tree.frontier_count()}")
print(f"  dirty_nodes at end: {len(dec.search_tree.dirty_nodes)}")

# Now test a simple variant 0 pattern
prog_v0 = [
    Instruction('Node.GetCumDist'),  # preamble: push Float
    Instruction('Node.ForEachSibling', code_block=[
        Instruction('Node.GetCumDist'),
        Instruction('Node.SetScore'),
    ]),
    Instruction('Float.Const0'),  # tail: correction = 0
]
vm2 = MIMOPushVM(flops_max=2000000, step_max=3000)
dec2 = BPStackDecoder(Nt=Nt, Nr=Nr, constellation=const, max_nodes=200, vm=vm2)
x_hat2, flops2 = dec2.detect(H, y, prog_v0, noise_var=nv)
print(f"\nSimple BP v0: bp_updates={dec2.bp_updates}, flops={flops2}")
print(f"  tree nodes: {dec2.search_tree.node_count()}")
