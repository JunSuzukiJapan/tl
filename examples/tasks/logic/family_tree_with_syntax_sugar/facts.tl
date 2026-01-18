// 1. Entities & Facts
// Implicitly declares 'father' relation
father(alice, bob).
father(bob, charlie).
father(charlie, diana).

// 2. Rules
// Grandparent: P -> F -> GP
grandparent(g, c) :- father(g, f), father(f, c).

// Ancestor (Recursive)
ancestor(a, d) :- father(a, d).
ancestor(a, d) :- father(a, x), ancestor(x, d).
