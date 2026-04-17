# Algorithm IR Architecture Guide

This package implements a structure-neutral algorithm IR for algorithm discovery and skeleton transplantation.
It is written for readers who do not have a compiler background, so this document explains the system from first principles.

The most important architectural fact is this:

`xDSL is now the real low-level substrate of the system.`

That means:

- `compile_function_to_ir(...)` first lowers a restricted Python function into a legacy neutral op/block/value description.
- That description is immediately lowered into an xDSL `ModuleOp`.
- The public `FunctionIR` object you work with is reconstructed from that xDSL module.
- When we graft a donor skeleton, we mutate xDSL blocks and xDSL operations directly.
- After mutation, we rebuild `FunctionIR` from xDSL again.

## Quick Demo

To see the system step through BP injection into a stack decoder, run:

```powershell
conda run --no-capture-output -n AutoGenOld python research/algorithm-IR/demo.py
```

The demo shows two levels of transplantation:

- local score-region injection with `bp_summary_update`
- runtime-nested injection where `bp_tree_runtime_update` runs after each stack expansion

So the dictionaries `func_ir.values`, `func_ir.ops`, and `func_ir.blocks` are no longer the hidden source of truth.
They are a convenient analysis and interpreter view rebuilt from xDSL.

## 1. What Problem This System Solves

The system is designed for this workflow:

1. Take a Python algorithm and lower it into a uniform executable IR.
2. Run that IR and collect runtime facts.
3. Select a local computation region to override.
4. Infer the boundary of that region.
5. Insert a donor algorithm skeleton into that region.
6. Produce a new executable IR and run it again.

The IR is intentionally structure-neutral.
It does not hardcode that some region "is a search tree" or "is BP".
It only records executable facts such as:

- read an item from a dictionary
- call a function
- append to a list
- compare two values
- branch to another block

Higher-level interpretations are optional and live above the core IR.

## 2. Why We Do Not Rewrite Python Source Directly

If we tried to graft donor skeletons directly on Python source code, several things would be fragile:

- It would be hard to define the exact region being replaced.
- It would be hard to state the inputs and outputs of that region precisely.
- It would be hard to preserve control flow and dataflow after rewriting.
- It would be hard to compare host and donor algorithms in one common representation.

The IR solves this by turning a Python function into:

- explicit values
- explicit operations
- explicit blocks
- explicit control-flow edges

This gives us a representation that is:

- executable
- sliceable
- rewriteable
- regenerable

## 3. The Current Architecture in One Picture

The main pipeline is:

`Python function -> AST -> neutral op/block/value description -> xDSL ModuleOp -> FunctionIR view -> execution/analysis/rewrite`

More concretely:

1. `frontend/ast_parser.py`
   Parses Python source into a Python AST and records source spans.
2. `frontend/ir_builder.py`
   Lowers the restricted Python AST into neutral values, ops, and blocks.
3. `ir/xdsl_bridge.py`
   Converts that neutral representation into xDSL `ModuleOp`.
4. `ir/model.py`
   Rebuilds a `FunctionIR` view from xDSL so analysis and the interpreter can work on a simple API.
5. `runtime/interpreter.py`
   Executes the rebuilt `FunctionIR` and records runtime events and runtime values.
6. `factgraph/`
   Builds a graph that aligns static IR and dynamic execution.
7. `region/`
   Defines the rewrite region and infers its boundary contract.
8. `grafting/`
   Matches a donor skeleton and rewrites the xDSL program.
9. `regeneration/`
   Packs the rewritten result as an `AlgorithmArtifact`.

## 4. Core Objects You Need to Understand

### 4.1 `Value`

A `Value` is a static value slot in the IR.
It is not a runtime Python object.

You can think of it as:

- a wire in a circuit
- a variable version
- the output slot of an operation

Important fields:

- `id`: internal id such as `v_64`
- `name_hint`: readable hint such as `score_0`
- `type_hint`: coarse static type such as `float`, `list`, `dict`, `tuple`, `complex`
- `def_op`: which op defines it
- `use_ops`: which ops consume it
- `attrs`: extra metadata such as source variable name and type info

### 4.2 `Op`

An `Op` is a single low-level action.

Examples:

- `const`
- `assign`
- `binary`
- `compare`
- `call`
- `get_item`
- `set_item`
- `append`
- `pop`
- `branch`
- `jump`
- `return`

Important fields:

- `opcode`
- `inputs`
- `outputs`
- `block_id`
- `attrs`

### 4.3 `Block`

A `Block` is a straight-line sequence of operations with no internal branch target.

It has:

- `op_ids`
- `preds`
- `succs`

Blocks are what make control flow explicit.
This is especially important for:

- `if/else`
- `while`
- loop backedges
- returns

### 4.4 `FunctionIR`

`FunctionIR` is the public function-level object used by the rest of the system.

Today it has two roles:

- it exposes a friendly analysis/execution view
- it keeps the xDSL-backed representation attached

Important fields:

- `values`
- `ops`
- `blocks`
- `arg_values`
- `return_values`
- `entry_block`
- `xdsl_module`
- `xdsl_func`
- `xdsl_op_map`
- `xdsl_block_map`

The critical point is:

`FunctionIR.clone()` clones the xDSL module first, then rebuilds the public view from the clone.

That is the mechanism that makes xDSL the actual substrate instead of a mirror.

## 5. xDSL Is the Source of Truth

This section is the architectural change that matters most.

### Before

Earlier, the project used a handwritten op/block/value graph as the real internal state.
xDSL was only attached as a side mirror.
That was not a real migration.

### Now

Now the workflow is:

1. Build a neutral temporary representation.
2. Lower it to xDSL.
3. Rebuild `FunctionIR` from xDSL.
4. Perform skeleton grafting by mutating xDSL blocks and operations.
5. Rebuild `FunctionIR` from xDSL again.

This means that if an edit is not reflected in xDSL, it is not part of the real rewritten program.

### Why This Matters

Because it gives us:

- stable cloning
- a proper IR host framework
- structured blocks and regions
- operation insertion and erasure APIs
- a foundation for richer types and future passes

## 6. How Python Is Lowered

The frontend currently supports a restricted but useful subset of Python:

- assignments
- `if/else`
- `while`
- `for`
- comparisons
- unary and binary arithmetic
- function calls
- attribute access
- item access
- list, tuple, and dict literals
- list mutation via `append` and `pop`

The lowering is intentionally low-level.

For example, a source statement like:

```python
score = candidate["metric"] + costs[candidate["depth"]]
```

does not remain a single high-level node.
It becomes several explicit IR ops:

- load key `"metric"`
- load `candidate["metric"]`
- load key `"depth"`
- load `candidate["depth"]`
- load `costs[...]`
- do the `Add`
- assign to `score`

This is what makes partial replacement possible.

## 7. What the xDSL Layer Stores

The xDSL layer currently uses unregistered operations with payloads.
This is deliberate.

We are not yet trying to encode algorithm meaning into a high-level typed dialect.
Instead, we use xDSL as a robust IR host:

- blocks are real xDSL blocks
- the function is a real xDSL `FuncOp`
- the program is a real xDSL `ModuleOp`
- neutral op metadata is stored in op payload attributes

The block label operation stores:

- block id
- block attrs
- predecessor order
- successor order

The predecessor order matters because `phi` selection depends on it.

## 8. Why the Public View Still Exists

A natural question is:

If xDSL is the substrate, why keep `Value`, `Op`, and `Block` dataclasses at all?

Because they are still convenient for:

- runtime interpretation
- region slicing
- boundary inference
- factgraph construction
- debugging
- readable tests

But these views are now regenerated from xDSL.
They are not the hidden program state that rewriting mutates directly.

## 9. Runtime Layer

The runtime system executes `FunctionIR` and records dynamic evidence.

The key runtime objects are:

### `RuntimeValue`

A concrete runtime instance of a static IR value.

### `RuntimeEvent`

A concrete execution of one static op.
It records:

- which static op ran
- which runtime values entered
- which runtime values exited
- control context

### `ShadowStore`

Tracks mutable object behavior across runtime execution:

- attribute writes
- item writes
- list membership
- object version history

This matters because many algorithms use mutable Python containers and dictionaries.

## 10. FactGraph

`factgraph/` merges:

- static IR facts
- dynamic execution facts
- alignment edges between static and dynamic objects

This is useful for:

- runtime-aware region analysis
- later structure mining
- future NN guidance

The current system does not require automatic structure discovery to do rewriting.
But FactGraph preserves the evidence needed for later work.

## 11. RewriteRegion and BoundaryContract

These are the real rewrite-centered abstractions of the system.

### `RewriteRegion`

A region says:

- which ops belong to the region
- which blocks belong to the region
- which values enter from outside
- which values leave to outside
- which state carriers are involved
- which schedule anchors matter

It is the answer to:

`What exact part of the host algorithm are we going to peel off or override?`

### `BoundaryContract`

A contract says:

- what the input ports are
- what the output ports are
- what can be read
- what can be written
- whether new state is allowed
- where the rewritten outputs reconnect
- which invariants must be preserved

It is the answer to:

`What shape must a donor satisfy to fit here safely?`

## 12. Projection Is Optional

Projection is no longer the primary rewrite object.

A projection is only an optional interpretation layer.
It can describe a region as:

- scheduling-like
- local-interaction-like
- candidate-pool-like

But rewriting does not require projection.
The central rewrite objects are still:

- `RewriteRegion`
- `BoundaryContract`
- donor skeleton

## 13. Skeleton Grafting

The current package implements two donor styles.

### 13.1 `bp_summary_update`

This donor replaces a scalar score computation.

In the stack-decoder host, the original region is roughly:

```python
score = candidate["metric"] + costs[candidate["depth"]]
```

The donor-based version computes a BP-like summary and then reconnects the result back into `score`.

### 13.2 `bp_tree_runtime_update`

This donor is the more important one.

It is a runtime-nested graft:

- host algorithm expands one layer of the stack decoder
- donor runs on the current explored tree
- donor updates every explored node
- donor writes back runtime state such as `bp_bias` and `metric`
- host continues search

This is the current demonstration that the system can support:

`host schedule contains donor schedule`

which is the beginning of true nested skeleton transplantation.

## 14. How Grafting Works Now

The implementation in `grafting/rewriter.py` now works like this:

1. Clone the host `FunctionIR`.
   This clones the xDSL module, not just a Python dict.
2. Resolve the region and contract.
3. Build donor lowering ops as new xDSL operations.
4. Insert those xDSL ops into the chosen xDSL block.
5. Erase removed xDSL ops if needed.
6. Rebuild a fresh `FunctionIR` view from xDSL.
7. Validate and execute the rewritten result.

This is the exact point where the migration became real.

## 15. Richer Type Support

The system now supports richer type metadata than the earliest MVP.

Examples already covered in tests:

- `tuple`
- `complex`
- `list`
- `dict`
- scalar types such as `int`, `float`, `bool`

This is still a lightweight type system, not a full theorem-proving type discipline.
But it is enough to make the IR and grafting pipeline much less brittle.

## 16. Current Tests and What They Prove

The test suite currently has 10 tests.

### Frontend tests

- compile a branch-and-loop example
- compile a stack-decoder-like host
- compile a tuple and complex example

These prove:

- control flow lowering works
- container-heavy host code lowers correctly
- richer value kinds are preserved

### Runtime and FactGraph tests

- execute the branch-and-loop example
- execute the stack decoder and build factgraph
- execute the complex tuple example

These prove:

- the xDSL-backed rebuild is executable
- the interpreter still matches Python behavior
- dynamic fact collection still works

### Region and Projection tests

- define a region
- infer a boundary contract
- optionally annotate projections

These prove:

- the rewrite-centered abstractions still work on the xDSL-backed IR

### Integration tests

- graft `bp_summary_update` into `stack_decoder_host`
- graft `bp_tree_runtime_update` into `stack_decoder_runtime_host`

These prove:

- xDSL-native rewriting works
- rewritten IR is executable
- runtime donor nesting works
- `clone()` preserves the grafted program

The second integration test is especially important.
It checks that after each expansion, the runtime BP donor runs over the current explored tree and records:

```text
[3, 5, 7]
```

That means the donor really ran over all explored nodes at each step, not just a trivial scalar replacement.

## 17. What Changed in This Refactor

The major changes in this migration are:

- `compile_function_to_ir(...)` now produces an xDSL-backed `FunctionIR`.
- `FunctionIR.clone()` clones xDSL and rebuilds the public view.
- block predecessor order is preserved in xDSL payload so `phi` works correctly.
- payload round-tripping now restores callables correctly after xDSL clone.
- `grafting/rewriter.py` now mutates xDSL blocks and xDSL ops directly.
- integration tests now require grafted IR to survive `clone()` and still execute correctly.
- runtime tree-BP grafting remains supported and tested.

## 18. Current Limitations

The system is much stronger now, but still intentionally limited.

Current limitations include:

- restricted Python frontend rather than full Python support
- neutral payload-based xDSL ops rather than a dedicated dialect
- lightweight static types instead of a stronger typed effect system
- no automatic structure discovery yet
- no NN-guided region ranking yet

These are acceptable for the current research phase because the immediate goal is:

`robust region-based executable skeleton transplantation`

and that goal is now supported by an xDSL-backed core.

## 19. Recommended Mental Model

If you want one sentence to remember the whole system, use this:

`We compile Python algorithms into a structure-neutral xDSL-backed executable IR, slice out a local computation region, attach a contract to it, graft a donor skeleton there, and then rebuild and execute the new algorithm.`

That is the present architecture.
