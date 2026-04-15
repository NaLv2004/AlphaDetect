"""Trace exactly what happens in _mapreduce with SetScore."""
import sys
sys.path.insert(0, r'd:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\code')

# Monkey-patch SetScore to trace
import vm as VM_module
original_execute = VM_module.MIMOPushVM._execute_instruction

def traced_execute(self, ins):
    if ins.name == 'Node.SetScore':
        val = self.float_stack.peek()
        nd = self.node_stack.peek()
        print(f"  [SetScore] val={val}, nd={nd.node_id if nd else None}, "
              f"node_stack_depth={self.node_stack.depth()}")
    if ins.name == 'Node.GetCumDist':
        nd = self.node_stack.peek()
        print(f"  [GetCumDist] nd={nd.node_id if nd else None}, cum_dist={nd.cum_dist if nd else None}")
    original_execute(self, ins)
    if ins.name == 'Node.SetScore':
        from stacks import TreeNode
        nd = self.node_stack.peek()
        print(f"  [SetScore after] nd={nd.node_id if nd else None}, score={nd.score if nd else None}")
        g = self.graph_stack.peek()
        print(f"  [dirty_nodes] = {[n.node_id for n in g.dirty_nodes] if g else 'no graph'}")

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
print(f"Siblings: {[(c.node_id, c.cum_dist) for c in g.siblings(children[2])]}")
print()
result = vm.run(prog)
print(f"\nResult: {result}")
print(f"Final scores: {[c.score for c in children]}")
print(f"dirty_nodes: {[n.node_id for n in g.dirty_nodes]}")
