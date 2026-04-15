"""Minimal trace test to see exactly why SetScore isn't firing."""
import sys
sys.path.insert(0, r'd:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\code')

from vm import MIMOPushVM, Instruction, program_to_oneliner
from stacks import SearchTreeGraph, TreeNode
import numpy as np

# Create a minimal fake graph with some nodes
g = SearchTreeGraph()
root = g.create_root(layer=4)
root.cum_dist = 0.0
root.score = 0.0

# Add 3 "children" manually
children = []
for i in range(3):
    c = g.add_child(root, layer=3, symbol=1+0j, local_dist=float(i),
                    cum_dist=float(i), partial_symbols=np.array([1+0j]))
    c.score = float(i) + 100.0  # high initial score
    children.append(c)

print("Before: children scores =", [c.score for c in children])
print("dirty_nodes =", g.dirty_nodes)

# Run a minimal program on children[2] (which has 2 siblings: children[0], children[1])
vm = MIMOPushVM(flops_max=100000, step_max=1000)
R = np.eye(4, dtype=complex)
y = np.zeros(4, dtype=complex)

# Test: inject on children[2], run ForEachSibling([GetCumDist, SetScore])
vm.inject_environment(
    R=R, y_tilde=y,
    x_partial=children[2].partial_symbols,
    graph=g, candidate_node=children[2],
    depth_k=3,
    constellation=np.array([1+0j, -1+0j]),
    noise_var=0.1,
)

prog = [
    Instruction('Node.ForEachSibling', code_block=[
        Instruction('Node.GetCumDist'),
        Instruction('Node.SetScore'),
    ]),
    Instruction('Float.Const0'),  # correction = 0
]

print(f"\nProgram: {program_to_oneliner(prog)}")
print(f"Running on node: id={children[2].node_id}, cum_dist={children[2].cum_dist}")
print(f"Siblings: {[(c.node_id, c.cum_dist) for c in g.siblings(children[2])]}")

result = vm.run(prog)
print(f"\nResult: {result}")
print(f"After: children scores = {[c.score for c in children]}")
print(f"dirty_nodes = {g.dirty_nodes}")
print(f"Expected: children[0].score=0.0, children[1].score=1.0, unchanged children[2].score={children[2].score}")
