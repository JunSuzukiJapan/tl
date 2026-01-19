# Logic Programming in TensorLanguage (TL)

TL integrates a powerful Prolog-like logic programming engine directly into its tensor-based runtime. This allows you to define knowledge bases, perform logical inference, and seamlessly mix symbolic reasoning with neural networks or numerical computations.

## 1. Syntax Overview

Logic statements in TL are first-class citizens. You can define **Facts**, **Rules**, and **Queries**.

### Facts
Facts declare static knowledge. They consist of a predicate and arguments (entities or values).

```rust
// Syntax Sugar (Recommended)
father(alice, bob).       // "alice is the father of bob"
is_student(charlie).      // Unary predicate

// Optional @ prefix (Legacy/Explicit)
@father(bob, diana).
```

### Rules
Rules define how to deduce new facts from existing ones. A rule holds if the **body** (right side) is true, implying the **head** (left side) is true.

```rust
// "x is a grandparent of z IF x is father of y AND y is father of z"
grandparent(x, z) :- father(x, y), father(y, z).

// Recursive rules are supported
ancestor(x, y) :- father(x, y).
ancestor(x, y) :- father(x, z), ancestor(z, y).
```

- Variables start with a lowercase letter in rules (conventionally `x`, `y`, `z` or `$x` in queries).
- `,` represents logical AND.
- `.` ends the statement.

### Queries
You can query the Knowledge Base (KB) using the `?` suffix. Queries return Tensors.

```rust
// 1. Boolean Query (True/False)
// Returns a 0-dimensional tensor: [1.] (True) or [0.] (False)
let is_father = ?father(alice, bob); 
println("Is alice father of bob? {}", is_father);

// 2. Variable Query (Search)
// Use $variable to ask "who?". Returns a list of matches.
// Result: Tensor of shape [N, 1] containing entity IDs (or names when printed).
let children = ?father(alice, $child);
println("Children of alice: {}", children);
```

## 2. Symbolic Output

TL automatically maps entity names (symbols) to unique integer IDs internally. When printing logic results involving entities, the runtime automatically resolves these IDs back to their names.

```rust
father(alice, bob).

fn main() {
    println("{}", ?father(alice, $x));
    // Output:
    // [[bob]]
}
```

## 3. Scope and File Organization

Facts and rules must be defined at the **global scope**. They cannot be defined inside functions.

### Single File (Script Style)
You can define facts, rules, and the `main` function in the same file.

```rust
// main.tl
father(alice, bob).
grandparent(x, z) :- father(x, y), father(y, z).

fn main() {
    let res = ?grandparent(alice, $x);
    println("{}", res);
}
```

### External Files (Module Style)
You can organize your logic in separate files. The compiler automatically collects facts and rules from all imported modules.

**facts.tl**:
```rust
father(alice, bob).
father(bob, charlie).
```

**logic.tl**:
```rust
// Rules can also be in separate files
grandparent(x, z) :- father(x, y), father(y, z).
```

**main.tl**:
```rust
mod facts;
mod logic;

// Import relations/rules into the current scope
use facts::*;
use logic::*;

fn main() {
    // Facts from 'facts.tl' and rules from 'logic.tl' are automatically loaded.
    let res = ?grandparent(alice, $x);
    println("{}", res);
}
```

## 4. Integration with Tensors

Since query results are standard TL Tensors, you can use them in mathematical operations.

- **Boolean**: `0.0` or `1.0` (float). Useful for masking or conditional logic.
- **Search**: `Int64` tensor of entity IDs. Can be used as indices for embeddings.

Example: Neuro-Symbolic Integration
```rust
// Logic: Find all ancestors
let ancestors = ?ancestor(alice, $x);

// Neural: Get embeddings for these ancestors
let embeds = embedding(ancestors, weights);

// Compute average embedding
let query_vec = mean(embeds, 0);
```

## 4. Advanced Features

### Transitive/Recursive Logic
TL supports full Datalog-style recursion. You can define complex relations like graph reachability or transitive closures.

```rust
path(x, y) :- edge(x, y).
path(x, y) :- edge(x, z), path(z, y).
```

### Multiple Dependencies
Rules can have multiple conditions. All must be satisfied.

```rust
compatible(x, y) :- is_friend(x, y), has_same_hobby(x, y).
```

### Automatic KB Initialization
There is no need to manually initialize the Knowledge Base. The compiler automatically aggregates all facts and rules from all modules (including imports) and initializes the inference engine at the start of `main()`.
