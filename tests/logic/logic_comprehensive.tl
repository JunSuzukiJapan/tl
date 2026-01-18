
// ==========================================
// Comprehensive Logic Inference Test
// ==========================================

// 1. Simple Facts
// p(a).
p(a).
p(b).

// 2. Simple Rules
// q(x) :- p(x).
q(x) :- p(x).

// 3. Transitive Rules / Recursion
// edge(1, 2). edge(2, 3). edge(3, 4).
edge(1, 2).
edge(2, 3).
edge(3, 4).
edge(4, 5).

// path(x, y) :- edge(x, y).
// path(x, y) :- edge(x, z), path(z, y).
path(x, y) :- edge(x, y).
path(x, y) :- edge(x, z), path(z, y).

// 4. Multiple Dependencies
// r(x) :- p(x), q(x). (should be true for a, b)
r(x) :- p(x), q(x).

// 5. Arity > 2
// triple(1, 2, 3).
triple(1, 2, 3).
triple(3, 4, 5).

// tri_rule(x, z) :- triple(x, y, z).
tri_rule(x, z) :- triple(x, y, z).

// ==========================================
// Queries
// ==========================================

fn main() {
    println("--- Test 1: Simple Facts ---");
    // Expect: true
    println("p(a)?");
    println("{}", @p(a)?);

    // Expect: false
    println("p(c)?");
    println("{}", @p(c)?);

    println("\n--- Test 2: Simple Rules ---");
    // Expect: true (derived from p(a))
    println("q(a)?");
    println("{}", @q(a)?);

    println("\n--- Test 3: Transitive/Recursive ---");
    // Expect: true (direct)
    println("path(1, 2)?");
    println("{}", @path(1, 2)?);

    // Expect: true (transitive 1->2->3)
    println("path(1, 3)?");
    println("{}", @path(1, 3)?);

    // Expect: true (deep 1->...->5)
    println("path(1, 5)?");
    println("{}", @path(1, 5)?);

    // Expect: false (reverse)
    println("path(2, 1)?");
    println("{}", @path(2, 1)?);

     println("\n--- Test 4: Multiple Dependencies ---");
    // Expect: true
    println("r(a)?");
    println("{}", @r(a)?);

    println("\n--- Test 5: Arity > 2 ---");
    // Expect: true
    println("triple(1, 2, 3)?");
    println("{}", @triple(1, 2, 3)?);

    // Expect: true (derived 1, 3)
    println("tri_rule(1, 3)?");
    println("{}", @tri_rule(1, 3)?);
    
    // Expect: true (derived 3, 5)
    println("tri_rule(3, 5)?");
    println("{}", @tri_rule(3, 5)?);

    println("\n--- Test 6: Variable Query ---");
    // Expect: 2, 3, 4, 5
    println("path(1, $x)?");
    println("{}", @path(1, $x)?);
}
