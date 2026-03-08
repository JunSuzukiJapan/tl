# Logic Programming in TensorLanguage (TL)

TL integrates a powerful Prolog-style logic inference engine directly into its runtime (tensor computation engine). This enables you to define knowledge bases, perform logical reasoning, and seamlessly combine the results (symbolic reasoning) with neural networks and numerical computations.

## 1. Syntax Overview

In TL, logic statements are treated as first-class citizens. You can define **Facts**, **Rules**, and **Queries**.

### Facts
Facts declare static knowledge. They consist of a predicate and arguments (entities or values).

```rust
// Syntactic sugar (recommended)
father(alice, bob).       // "alice is the father of bob"
is_student(charlie).      // Unary predicate
```

### Relation Declarations
You can explicitly declare the argument types of relations. If not declared, they are automatically inferred from rules and facts.

```rust
relation parent(entity, entity);
relation age(entity, i64);
```

### Rules
Rules define how to derive new facts from existing ones. If the **body** (right side) is true, the **head** (left side) is also inferred to be true.

```rust
// "If x is the father of y, and y is the father of z, then x is the grandparent of z"
grandparent(x, z) :- father(x, y), father(y, z).

// Recursive rules are also supported
ancestor(x, y) :- father(x, y).
ancestor(x, y) :- father(x, z), ancestor(z, y).
```

- Variables in rules start with lowercase letters (conventionally `x`, `y`, `z`).
- `,` represents logical conjunction (AND).
- `.` terminates a statement.

### Queries
You can query the knowledge base (KB) using the `?` suffix. Query results are returned as tensors.

```rust
// 1. True/False query
// Returns a 0-dimensional tensor: [1.] (true) or [0.] (false)
let is_father = ?father(alice, bob); 
println("Is alice father of bob? {}", is_father);

// 2. Variable query (search)
// Use $variable_name to ask "who?" or "what?". Returns a list of matches.
// Result: Tensor of shape [N, 1]. Contains entity IDs (displayed as names).
let children = ?father(alice, $child);
println("Children of alice: {}", children);
```

## 2. Symbolic Output

TL automatically maps entity names (symbols) to unique integer IDs internally. When displaying logic query results (tensors containing entity IDs), the runtime automatically resolves these IDs back to their original names.

```rust
father(alice, bob).

fn main() {
    println("{}", ?father(alice, $x));
    // Output:
    // [[bob]]
}
```

## 3. Scope and File Organization

Facts and rules must be defined at **global scope** (outside of functions). They cannot be defined inside functions.

### Single File (Script Style)
Facts, rules, and the `main` function can all be in the same file.

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
Logic can be organized into separate files. The compiler automatically collects facts and rules from all imported modules.

**facts.tl**:
```rust
father(alice, bob).
father(bob, charlie).
```

**logic.tl**:
```rust
// Rules can also be placed in separate files
grandparent(x, z) :- father(x, y), father(y, z).
```

**main.tl**:
```rust
mod facts;
mod logic;

// Import relations and rules into current scope
use facts::*;
use logic::*;

fn main() {
    // Facts from 'facts.tl' and rules from 'logic.tl' are automatically loaded
    let res = ?grandparent(alice, $x);
    println("{}", res);
}
```

## 4. Integration with Tensors

Query results are standard TL tensors, so they can be used directly in mathematical and neural network operations.

- **Boolean values**: `0.0` or `1.0` (float). Useful for masking and conditional logic.
- **Search results**: `Int64` tensors of entity IDs. Can be used as indices for Embedding layers.

Example: Neuro-Symbolic Integration
```rust
// Logic: search all ancestors
let ancestors = ?ancestor(alice, $x);

// Neural: get embedding vectors for these ancestors
let embeds = embedding(ancestors, weights);

// Compute mean vector
let query_vec = mean(embeds, 0);
```

## 4. Advanced Features

### Transitive and Recursive Logic
TL supports full Datalog-style recursion. You can define complex relationships such as graph reachability and transitive closures.

```rust
path(x, y) :- edge(x, y).
path(x, y) :- edge(x, z), path(z, y).
```

### Multiple Dependencies
Rules can have multiple conditions. The rule holds only when all conditions are satisfied.

```rust
compatible(x, y) :- is_friend(x, y), has_same_hobby(x, y).
```

### Automatic KB Initialization
There is no need to manually initialize the knowledge base. The compiler automatically aggregates facts and rules from all modules (including imported ones) and initializes the inference engine at the start of `main()`.

## 5. Negation and Arithmetic Comparisons

### Negation
You can use `not()` in rule bodies to express negation of facts. TL supports stratified negation.

```rust
// Those without a direct boss are managers
manager(X) :- employee(X), not(has_boss(X)).
```

Negation can be combined with recursion, but negative cycles are detected and result in a compile error.

```rust
// This is valid (negation is in a different stratum)
reachable(X, Y) :- edge(X, Y).
reachable(X, Y) :- edge(X, Z), reachable(Z, Y).
unreachable(X, Y) :- node(X), node(Y), not(reachable(X, Y)).
```

### Arithmetic Comparisons
Arithmetic comparisons can be used as conditions in rule bodies.

```rust
// Those aged 18 or older are adults
adult(X) :- person(X, Age), Age >= 18.

// Computing salary differences
earns_more(X, Y) :- salary(X, Sx), salary(Y, Sy), Sx > Sy.
```

Supported comparison operators: `>`, `<`, `>=`, `<=`, `==`, `!=`
