"""Trace the discovered program to understand what correction it computes."""
import numpy as np
from vm import MIMOPushVM, Instruction, program_to_string
from stacks import TreeNode, SearchTreeGraph
from stack_decoder import qam16_constellation

# The discovered program (gen 10-26, BER=0.02083, matches KB16)
best_prog = [
    Instruction('Exec.DoTimes', code_block=[
        Instruction('Mat.ElementAt'),
        Instruction('Float.FromInt'),
        Instruction('Float.Min'),
        Instruction('Node.GetLocalDist'),
    ]),
    Instruction('Node.GetSymRe'),
    Instruction('Node.ForEachAncestor', code_block=[
        Instruction('Bool.Dup'),
        Instruction('Matrix.Pop'),
    ]),
    Instruction('Node.ReadMem'),
    Instruction('Vec.Sub'),
    Instruction('Node.ForEachSibling', code_block=[
        Instruction('Node.ForEachSibling', code_block=[
            Instruction('Float.Const2'),
        ]),
        Instruction('Float.Min'),
        Instruction('Node.ChildAt'),
        Instruction('Node.ForEachSibling', code_block=[
            Instruction('Vec.Sub'),
        ]),
    ]),
]

print("Program:")
print(program_to_string(best_prog))
print()

# Create a test scenario
Nt, Nr = 8, 16
constellation = qam16_constellation()
rng = np.random.RandomState(42)

# Generate a channel
H = (rng.randn(Nr, Nt) + 1j * rng.randn(Nr, Nt)) / np.sqrt(2)
x_idx = rng.randint(0, len(constellation), Nt)
x = constellation[x_idx]
snr_db = 14.0
sig_power = float(np.mean(np.abs(H @ x) ** 2))
noise_var = sig_power / (10 ** (snr_db / 10.0))
noise = np.sqrt(noise_var / 2) * (rng.randn(Nr) + 1j * rng.randn(Nr))
y = H @ x + noise

# QR decomposition
Q, R = np.linalg.qr(H, mode='reduced')
y_tilde = Q.conj().T @ y
R_real = np.real(R)
y_tilde_real = np.real(y_tilde)

# Build a small search tree to test
graph = SearchTreeGraph()
root = graph.create_root(layer=7)

# Expand root layer (layer 7) with all symbols
for sym in constellation:
    residual = y_tilde_real[7] - R_real[7, 7] * float(np.real(sym))
    ld = residual ** 2
    child = graph.add_child(
        parent=root, layer=6, symbol=sym,
        local_dist=ld, cum_dist=ld,
        partial_symbols=np.array([np.real(sym)]),
    )

# Now expand one child at layer 6
parent = root.children[0]
for sym in constellation:
    residual = y_tilde_real[6] - R_real[6, 6] * float(np.real(sym)) - R_real[6, 7] * float(np.real(parent.symbol))
    ld = residual ** 2
    child = graph.add_child(
        parent=parent, layer=5, symbol=sym,
        local_dist=ld, cum_dist=parent.cum_dist + ld,
        partial_symbols=np.append(parent.partial_symbols, np.real(sym)),
    )

# Test the program on nodes at different layers
vm = MIMOPushVM(flops_max=100000, step_max=5000)

print("=== Testing on children of root (layer 6) ===")
for i, node in enumerate(root.children[:4]):
    vm.inject_environment(
        R=R_real, y_tilde=y_tilde_real,
        x_partial=node.partial_symbols,
        graph=graph, candidate_node=node,
        depth_k=node.layer,
    )
    correction = vm.run(best_prog)
    score = node.cum_dist + correction
    print(f"  child {i}: sym={np.real(node.symbol):.1f} "
          f"local_dist={node.local_dist:.4f} cum_dist={node.cum_dist:.4f} "
          f"correction={correction:.4f} score={score:.4f} "
          f"(ld+symre={node.local_dist + float(np.real(node.symbol)):.4f})")

print()
print("=== Testing on children at layer 5 ===")
parent = root.children[0]
for i, node in enumerate(parent.children[:4]):
    vm.inject_environment(
        R=R_real, y_tilde=y_tilde_real,
        x_partial=node.partial_symbols,
        graph=graph, candidate_node=node,
        depth_k=node.layer,
    )
    correction = vm.run(best_prog)
    score = node.cum_dist + correction
    print(f"  child {i}: sym={np.real(node.symbol):.1f} "
          f"local_dist={node.local_dist:.4f} cum_dist={node.cum_dist:.4f} "
          f"correction={correction:.4f} score={score:.4f} "
          f"(ld+symre={node.local_dist + float(np.real(node.symbol)):.4f})")

# Compare with simplified program: just local_dist + sym_re
print("\n=== Simplified correction: local_dist + Re(symbol) ===")
for i, node in enumerate(root.children[:4]):
    simple_corr = node.local_dist + float(np.real(node.symbol))
    score = node.cum_dist + simple_corr
    print(f"  child {i}: sym={np.real(node.symbol):.1f} "
          f"correction={simple_corr:.4f} score={score:.4f}")

# Test if this equals a static value for nodes with no siblings
print("\n=== Nodes with no siblings (only child) ===")
# A node with no siblings in the search tree
lone_node = graph.add_child(
    parent=root, layer=5, symbol=constellation[0],
    local_dist=0.5, cum_dist=1.0,
    partial_symbols=np.array([1.0, 1.0]),
)
vm.inject_environment(
    R=R_real, y_tilde=y_tilde_real,
    x_partial=lone_node.partial_symbols,
    graph=graph, candidate_node=lone_node,
    depth_k=lone_node.layer,
)
correction = vm.run(best_prog)
print(f"  lone node: correction={correction:.4f}")
