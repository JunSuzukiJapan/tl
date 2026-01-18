# Family Tree with Syntax Sugar

This example demonstrates TensorLogic's Prolog-like logic programming syntax sugar.

## Files

- `facts.tl`: Contains facts (father relations) and rules (grandparent, ancestor).
- `main.tl`: Main program that imports facts and executes logic queries.

## Syntax Highlights

### Facts
```
father(alice, bob).
father(bob, charlie).
```

### Rules
```
grandparent(g, c) :- father(g, f), father(f, c).
ancestor(a, d) :- father(a, d).
ancestor(a, d) :- father(a, x), ancestor(x, d).
```

### Queries
```rust
let res = @father(alice, bob)?;       // Boolean query
let res = @ancestor(alice, $x)?;      // Variable query ($x is unbound)
```

### Module System
```rust
mod facts;        // Load facts.tl as submodule
use facts::*;     // Import all relations
```

## Running

```bash
cd examples/tasks/logic/family_tree_with_syntax_sugar
tl main.tl
```
