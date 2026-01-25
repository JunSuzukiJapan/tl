# Design: Liveness Analysis & Memory Slot Assignment

## 1. Structure

We will introduce a new module `src/compiler/liveness.rs`.

```rust
pub struct LivenessAnalyzer {
    // Variable name -> Interval (start_stmt_idx, end_stmt_idx)
    intervals: HashMap<String, Interval>,
    // Current statement index during traversal
    current_idx: usize,
}

#[derive(Debug, Clone)]
pub struct Interval {
    start: usize, // Index of first definition
    end: usize,   // Index of last use
}
```

## 2. Algorithm (Linear Scan Simplified)

1.  **Flatten Control Flow (Virtual Instructions)**:
    Since we are working on AST, we can treat each `Stmt` in a block as an instruction index.
    Nested blocks (If/Loop) complicate this.
    *Simple approach*: Assign a unique monotonically increasing index to every `Expr` and `Stmt` in execution order (Pre-order traversal).
    
2.  **Collect Intervals**:
    - **Def**: When `Let` or `Assign` is encountered, set `start` of interval.
    - **Use**: When `Variable` expr is encountered, update `end` of interval to `current_idx`.
    - **Loop Handling**: Variables used inside a loop must be considered live for the entire loop duration (simplified conservative approach) or we implementation iterative dataflow analysis.
    *Conservatively*: If a var is used in a loop, extend its end to the end of the loop.

3.  **Slot Assignment (Graph Coloring / Greedy)**:
    - Sort intervals by start time.
    - Iterate and assign to "Slots".
    - `Slot` = Reusable memory buffer ID.
    - Two variables can share a slot if their intervals do not overlap.

## 3. Integration with Codegen

- `CodeGenerator` will run `LivenessAnalyzer::analyze(&function)` before generating code.
- It produces a `SlotMap: HashMap<String, usize>` (VarName -> SlotID).
- **Codegen**:
    - When `Let x = Tensor::new(...)` is compiled:
        - Check `SlotMap` for `x`.
        - If assigned Slot `S`:
             - Emit `ptr = runtime::get_buffer(S, size)` (Reuse)
        - Else:
             - Emit `malloc` (Fallback)

## 4. Destination Passing Style (DPS)

For functions returning tensors:
- Modify `compile_function_def` to add `dest: *mut Tensor` as 1st arg.
- Modify `Return` stmt to store to `dest`.
- Modify `Call` site to allocate a slot and pass it.

## 5. Structs

```rust
pub struct MemorySlot {
    pub id: usize,
    pub size_hint: Option<usize>, // If statically known
}

pub struct FunctionAnalysis {
    pub slots: HashMap<String, usize>, // Var -> SlotID
    pub max_slots: usize,
}
```
