"""Test pseudocode generator on genomes from sbp0414-1.log."""
import numpy as np
from vm import Instruction
from bp_main_v2 import (
    program_to_formula_trace, program_to_formula, program_to_pseudocode,
    _format_pseudocode_block, Genome
)

# ─── Helper ────────────────────────────────────────────────────────────────
def show(g, label):
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    print(f'  F_down  = {program_to_formula_trace(g.prog_down,  "down",   g)}')
    print(f'  F_up    = {program_to_formula_trace(g.prog_up,    "up",     g)}')
    print(f'  F_belief= {program_to_formula_trace(g.prog_belief,"belief", g)}')
    print(f'  H_halt  = {program_to_formula_trace(g.prog_halt,  "halt",   g)}')
    ec = g.evo_constants
    print(f'  EC: {ec[0]:.4f} {ec[1]:.4f} {ec[2]:.4f} {ec[3]:.4f}')
    print()
    for pt_label, pt_type, pt_prog in [
        ('F_down ', 'down',    g.prog_down),
        ('F_up   ', 'up',      g.prog_up),
        ('F_belief', 'belief', g.prog_belief),
        ('H_halt ', 'halt',    g.prog_halt),
    ]:
        plines = _format_pseudocode_block(pt_prog, pt_type, g)
        print(f'  [{pt_label}]:  ({len(pt_prog)} instrs)')
        for pl in plines:
            print(f'    {pl}')
    print()

# ─── Genome #1 from sbp0414-1, Gen 1 (BER=0.05414) ────────────────────────
g1 = Genome(
    prog_down=[Instruction('Float.Neg'), Instruction('Float.Sub'), Instruction('Float.Square')],
    prog_up=[
        Instruction('Float.Sub'), Instruction('Float.Div'), Instruction('Int.Pop'),
        Instruction('Float.Add'),
        Instruction('Node.ForEachChild', code_block=[
            Instruction('Node.GetMUp'), Instruction('Float.Dup'), Instruction('Float.EvoConst0')]),
        Instruction('Float.Const1'),
        Instruction('Exec.DoTimes', code_block=[
            Instruction('Float.Mul'), Instruction('Int.Dup'), Instruction('Float.Max')]),
        Instruction('Float.Neg'), Instruction('Graph.NodeCount'), Instruction('Float.Exp'),
        Instruction('Exec.DoTimes', code_block=[Instruction('Float.Min'), Instruction('Float.Rot')]),
    ],
    prog_belief=[Instruction('Float.Swap'), Instruction('Node.GetLocalDist'),
                 Instruction('Mat.Row'), Instruction('Float.Swap')],
    prog_halt=[Instruction('Float.Abs'), Instruction('Float.GT'),
               Instruction('Int.LT'), Instruction('Bool.Not')],
    log_constants=np.array([1.4731, 1.0414, 0.9708, -0.2980])
)
show(g1, 'Genome #1 (Gen 1, BER=0.05414)')

# ─── Genome #8 from Gen 1 (F_up = -Sum_children(M_up)) ───────────────────
g8 = Genome(
    prog_down=[Instruction('Float.GetMMSELB')],
    prog_up=[
        Instruction('Node.ForEachChild', code_block=[
            Instruction('Node.GetMUp'), Instruction('Float.Add')]),
        Instruction('Float.Neg'),
    ],
    prog_belief=[Instruction('Float.Abs'), Instruction('Node.GetMDown')],
    prog_halt=[Instruction('Float.Const0_1'), Instruction('Float.GT')],
    log_constants=np.zeros(4)
)
show(g8, 'Genome #8 (Gen 1, F_up = -Sum_children(M_up))')

# ─── Genome #4 from Gen 2 (F_up = D_i + DoTimes(GetMUp,Mul)) ─────────────
g4 = Genome(
    prog_down=[Instruction('Node.SetMUp'), Instruction('Float.Abs')],
    prog_up=[
        Instruction('Node.ForEachChild', code_block=[Instruction('Float.Add'), Instruction('Int.Pop')]),
        Instruction('Node.ForEachChild', code_block=[Instruction('Float.ConstHalf'), Instruction('Int.Dup')]),
        Instruction('Float.Sqrt'), Instruction('Float.Square'),
        Instruction('Float.Rot'), Instruction('Float.Swap'),
        Instruction('Exec.DoTimes', code_block=[
            Instruction('Int.Dup'), Instruction('Float.Abs'), Instruction('Node.GetMUp')]),
        Instruction('Node.GetCumDist'),
        Instruction('Exec.DoTimes', code_block=[Instruction('Node.GetMUp'), Instruction('Float.Mul')]),
        Instruction('Float.Add'),
    ],
    prog_belief=[Instruction('Node.GetCumDist'), Instruction('Node.GetCumDist'),
                 Instruction('Int.Const2'), Instruction('Node.GetMUp')],
    prog_halt=[Instruction('Bool.And'), Instruction('Float.LT'),
               Instruction('Float.EvoConst2'), Instruction('Float.EvoConst1'), Instruction('Float.Abs')],
    log_constants=np.array([1.4731, 0.9087, 0.9708, 0.1219])
)
show(g4, 'Genome #4 (Gen 2, best, BER=0.06647)')

import numpy as np
from vm import Instruction
from bp_main_v2 import program_to_formula_trace, program_to_formula, Genome

# Reproduce genome #1 from sbp0414-1.log (Gen 1 best, BER=0.05414)
g1 = Genome(
    prog_down=[Instruction('Float.Neg'), Instruction('Float.Sub'), Instruction('Float.Square')],
    prog_up=[
        Instruction('Float.Sub'), Instruction('Float.Div'), Instruction('Int.Pop'), Instruction('Float.Add'),
        Instruction('Node.ForEachChild', code_block=[Instruction('Node.GetMUp'), Instruction('Float.Dup'), Instruction('Float.EvoConst0')]),
        Instruction('Float.Const1'),
        Instruction('Exec.DoTimes', code_block=[Instruction('Float.Mul'), Instruction('Int.Dup'), Instruction('Float.Max')]),
        Instruction('Float.Neg'), Instruction('Graph.NodeCount'), Instruction('Float.Exp'),
        Instruction('Exec.DoTimes', code_block=[Instruction('Float.Min'), Instruction('Float.Rot')]),
    ],
    prog_belief=[Instruction('Float.Swap'), Instruction('Node.GetLocalDist'), Instruction('Mat.Row'), Instruction('Float.Swap')],
    prog_halt=[Instruction('Float.Abs'), Instruction('Float.GT'), Instruction('Int.LT'), Instruction('Bool.Not')],
    log_constants=np.array([1.4731, 1.0414, 0.9708, -0.2980])
)

print('=== Genome #1 from log (was: val=FAULT, deps=CONST/noop) ===')
print('F_down:', program_to_formula_trace(g1.prog_down, 'down', g1))
print('F_up:  ', program_to_formula_trace(g1.prog_up, 'up', g1))
print('F_bel: ', program_to_formula_trace(g1.prog_belief, 'belief', g1))
print('H_halt:', program_to_formula_trace(g1.prog_halt, 'halt', g1))
print()

print('=== program_to_formula display tests ===')
# ForEachChild([GetMUp, Dup, EC0]): ec0 is top each iter -> Sum_children(EC0)
print('ForEachChild([GetMUp,Dup,EC0]):',
    program_to_formula([Instruction('Node.ForEachChild', code_block=[
        Instruction('Node.GetMUp'), Instruction('Float.Dup'), Instruction('Float.EvoConst0')])]))

# ForEachChild([GetMUp, Add]): accumulates m_up from children
print('ForEachChild([GetMUp,Add]) with init:',
    program_to_formula([Instruction('Node.ForEachChild', code_block=[
        Instruction('Node.GetMUp'), Instruction('Float.Add')])], ['init']))

# ForEachChild([Float.Add]): consumes outer stack items
print('ForEachChild([Float.Add]) with [a,b,c,d]:',
    program_to_formula([Instruction('Node.ForEachChild', code_block=[Instruction('Float.Add')])],
        ['a', 'b', 'c', 'd']))

# DoTimes with context from existing stack items
print('DoTimes([Min,Rot]) after push 1 + ForEachChild:',
    program_to_formula([
        Instruction('Float.Const1'),
        Instruction('Node.ForEachChild', code_block=[Instruction('Node.GetMUp'), Instruction('Float.Dup'), Instruction('Float.EvoConst0')]),
        Instruction('Exec.DoTimes', code_block=[Instruction('Float.Min'), Instruction('Float.Rot')]),
    ], ['a', 'b']))

# Genome #4 type F_up: (D_i + repeat(nx: EMPTY)) - should now show more context
print()
print('=== Genome #4 style F_up with context ===')
g4_up = [
    Instruction('Node.GetCumDist'),  # push D_i
    Instruction('Int.Pop'),
    Instruction('Exec.DoTimes', code_block=[Instruction('Node.GetMUp'), Instruction('Float.Add')]),
]
# With 16-pair input names for context
flat_names = [f'x{i}' for i in range(32)]
print('Genome4-like F_up:', program_to_formula(g4_up, flat_names))
print()
print("Expected:")
print("  F_down   = (M_par_down + C_i)   [cumulative distance — A*]")
print("  F_belief = max(M_down, EC1)      [= max(cum_dist, 0.348) — clamped A*]")
print("  H_halt   should involve float LT comparison")
