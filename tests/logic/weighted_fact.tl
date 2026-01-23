
// Weighted Logic Facts Test

// 1. Facts with different types
edge(a, b, 1.5).
edge(b, c, 2.7).
is_active(a, true).
is_active(b, false).
label(a, "Start").
label(c, "End").

// 2. Rules using these facts
reachable($X, $Y, $W) :- edge($X, $Y, $W).
reachable($X, $Z, $W) :- edge($X, $Y, $W), reachable($Y, $Z, $W2).

fn main() {
    println("--- Test: Float in Facts ---");
    // Query: edge(a, b, $w)?
    let res1 = ?edge(a, b, $w);
    println("edge(a, b, $w) matches:");
    println(res1);
    
    println("\n--- Test: Bool in Facts ---");
    let res2 = ?is_active(a, $v);
    println("is_active(a, $v) matches (1 for true):");
    println(res2);
    
    let res3 = ?is_active(b, $v);
    println("is_active(b, $v) matches (0 for false):");
    println(res3);

    println("\n--- Test: Rule with Weight (after infer) ---");
    let res4 = ?reachable(a, b, $w);
    println("reachable(a, b, $w) matches (should be 1.5):");
    println(res4);

    println("\n--- Test: Float extraction from derived fact (Recursive) ---");
    let res5 = ?reachable(a, c, $w2);
    println("reachable(a, c, $w2) matches (should be 1.5 because of rule structure):");
    println(res5);

    println("\n--- Test: Entity and Global Consistency ---");
    let res6 = ?label(a, $s);
    println("label(a, $s) matches:");
    println(res6);
}
