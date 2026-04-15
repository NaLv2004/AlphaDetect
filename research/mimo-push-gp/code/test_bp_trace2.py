"""Trace exactly what happens in SetScore."""
import sys
sys.path.insert(0, r'd:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\code')
import traceback

import vm as VM_module
original_execute = VM_module.MIMOPushVM._execute_instruction

def traced_execute(self, ins):
    if ins.name in ('Node.SetScore', 'Node.GetCumDist'):
        print(f"  BEFORE {ins.name}: float_stack={[self.float_stack._items[-3:]]},"
              f" node_stack_depth={self.node_stack.depth()}")
    try:
        original_execute(self, ins)
    except Exception as e:
        print(f"  EXCEPTION in {ins.name}: {e}")
        traceback.print_exc()
        raise
    if ins.name in ('Node.SetScore', 'Node.GetCumDist'):
        nd = self.node_stack.peek()
        g = self.graph_stack.peek()
        print(f"  AFTER {ins.name}: nd={nd.node_id if nd else None} score={nd.score if nd else 'N/A'},"
              f" dirty={[n.node_id for n in g.dirty_nodes] if g else 'no g'}")

VM_module.MIMOPushVM._execute_instruction = traced_execute

from vm import MIMOPushVM, Instruction
from stacks import SearchTreeGraph
import numpy as np

g = SearchTreeGraph()
root = g.create_root(layer=4)
root.cum_dist = 0.0

children = []
for i in range(3):
    c = g.add_child(root, layer=3, symbol=1+0j, local_dist=float(i),
                    cum_dist=float(i), partial_symbols=np.array([1+0j]))
    c.score = float(i) + 100.0
    children.append(c)

vm = MIMOPushVM(flops_max=100000, step_max=1000)
R = np.eye(4, dtype=complex)
y = np.zeros(4, dtype=complex)

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
    Instruction('Float.Const0'),
]

print(f"Running program on node {children[2].node_id}")
print(f"Siblings: {[(c.node_id, c.cum_dist, c.score) for c in g.siblings(children[2])]}")
print()
result = vm.run(prog)
print(f"\nResult: {result}")
print(f"Final scores: {[(c.node_id, c.score) for c in children]}")
print(f"dirty_nodes: {[n.node_id for n in g.dirty_nodes]}")
