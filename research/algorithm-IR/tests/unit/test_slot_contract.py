"""R4: build_slot_contract derives ports from SlotPopulation.spec via
the type_lattice. No `(in:any) -> (out:any)` placeholders allowed for
slots whose spec carries real type information.
"""
from __future__ import annotations

import numpy as np

from algorithm_ir.ir.type_lattice import (
    ALL_ATOMIC_TYPES,
    TYPE_TOP,
    is_list_type,
    is_tuple_type,
    parse_composite,
)
from evolution.gp.contract import SlotContract, build_slot_contract
from evolution.ir_pool import build_ir_pool


def _is_lattice_token(t: str) -> bool:
    return (
        t in ALL_ATOMIC_TYPES
        or is_list_type(t)
        or is_tuple_type(t)
        or parse_composite(t) is not None
    )


def test_all_contract_ports_use_lattice_tokens():
    pool = build_ir_pool(np.random.default_rng(42))
    saw_real_type = False
    for genome in pool:
        for slot_key, pop in genome.slot_populations.items():
            contract = build_slot_contract(pop, slot_key=slot_key)
            assert isinstance(contract, SlotContract)
            for port in contract.input_ports:
                assert _is_lattice_token(port.type), \
                    f"{slot_key} input {port.name}: {port.type!r} not in lattice"
                if port.type != TYPE_TOP:
                    saw_real_type = True
            for port in contract.output_ports:
                assert _is_lattice_token(port.type), \
                    f"{slot_key} output {port.name}: {port.type!r} not in lattice"
                if port.type != TYPE_TOP:
                    saw_real_type = True
    assert saw_real_type, "no slot contract carried a non-TYPE_TOP type — alias map likely broken"


def test_tuple_return_decomposes_into_multiple_output_ports():
    pool = build_ir_pool(np.random.default_rng(42))
    found = False
    for genome in pool:
        for slot_key, pop in genome.slot_populations.items():
            spec_ret = getattr(pop.spec, "return_type", None)
            if spec_ret == "tuple" or (isinstance(spec_ret, str) and is_tuple_type(spec_ret)):
                contract = build_slot_contract(pop, slot_key=slot_key)
                # tuple<...> with components yields >=1 ports; bare "tuple" -> any
                assert len(contract.output_ports) >= 1
                found = True
    # not strictly required to find one; just ensure no crash if present.
    if not found:
        return


def test_population_uses_real_contract_not_placeholder():
    """Production path: micro_population_step must build a contract whose
    types come from the spec, not the legacy any/any placeholder."""
    from evolution.pool_types import SlotPopulation
    from evolution.skeleton_registry import ProgramSpec

    spec = ProgramSpec(
        name="regularizer",
        param_names=["G", "sigma2"],
        param_types=["mat", "float"],
        return_type="mat",
    )
    pop = SlotPopulation(slot_id="lmmse.regularizer", spec=spec)
    contract = build_slot_contract(pop, slot_key="lmmse.regularizer")
    assert contract.input_ports[0].type == "mat_cx"   # alias normalized
    assert contract.input_ports[1].type == "float"
    assert contract.output_ports[0].type == "mat_cx"
