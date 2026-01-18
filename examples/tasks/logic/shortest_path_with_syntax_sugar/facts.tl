// Edge facts: edge(from, to, weight)
// Note: The new logic syntax is primarily for symbolic reasoning.
// For numeric edge weights, we still use tensor operations for computation,
// but we demonstrate the syntax here for defining the graph structure.

// Graph edges (directed)
// A(0) -> B(1), weight 1.0
// B(1) -> C(2), weight 2.0
// A(0) -> C(2), weight 5.0 (slower direct path)
// C(2) -> D(3), weight 1.0

edge(a, b).
edge(b, c).
edge(a, c).
edge(c, d).

// Shortest path is computed using tropical matrix operations in main.tl
// This file demonstrates how graph structure can be declared using facts.
