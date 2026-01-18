# Shortest Path with Syntax Sugar

This example demonstrates TensorLogic's Prolog-like syntax combined with tropical matrix logic.

## Files

- `facts.tl`: Graph edge declarations using logic facts.
- `main.tl`: Shortest path computation using tropical matrix operations, with logic query demonstration.

## Approach

Graph structure is declared using logic facts:
```
edge(a, b).
edge(b, c).
```

Shortest path is computed using tropical matrix operations (min-plus semiring):
```rust
let new_dist = [i, j, k | ... { dist[i, k] + dist[k, j] }].min(2);
```

## Running

```bash
cd examples/tasks/logic/shortest_path_with_syntax_sugar
tl main.tl
```
