from cpp_bridge import CppBPEvaluator, OPCODE_MAP
cpp = CppBPEvaluator(Nt=16, Nr=16, mod_order=16, max_nodes=500, flops_max=1000000, step_max=1000, max_bp_iters=2)
print('DLL loaded OK')
if 'Node.ForEachChildMin' in OPCODE_MAP:
    print(f'ForEachChildMin opcode: {OPCODE_MAP["Node.ForEachChildMin"]}')
else:
    print('ERROR: ForEachChildMin NOT in OPCODE_MAP!')
print(f'Total opcodes: {len(OPCODE_MAP)}')
