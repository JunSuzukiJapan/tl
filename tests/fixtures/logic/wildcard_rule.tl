// Define some simple relations
edge(1, 2).
edge(2, 3).
edge(3, 4).

// Rule using wildcard: matching any destination
has_outgoing(x) :- edge(x, _).

// Rule using wildcard: matching any source
has_incoming(y) :- edge(_, y).

// Rule using multiple wildcards (path of length 2 exists)
has_path_2(start, end) :- edge(start, mid), edge(mid, end).
// Check ignoring the middle node completely
path_exists_ignoring_mid(start, end) :- edge(start, _), edge(_, end). 
// Note: edge(start, _) and edge(_, end) are independent, so this just checks if start has outgoing AND end has incoming, 
// NOT that they are connected.
// Actually, let's test specifically the unification behavior.

// Query using wildcard in query itself is not supported in syntax like `?edge(1, _)`.
// But we can define a rule that effectively does that.

// Testing simple match
match_any(_) :- true.

fn main() {
    println("--- Wildcard Logic Test ---");

    // 1. Test has_outgoing
    // Should be true for 1, 2, 3. False for 4.
    let out1 = (has_outgoing(1))[0];
    let out4 = (has_outgoing(4))[0];
    
    if out1 > 0.5 {
        println("has_outgoing(1): OK");
    } else {
        println("has_outgoing(1): FAILED");
    }

    if out4 < 0.5 {
         println("has_outgoing(4): OK");
    } else {
         println("has_outgoing(4): FAILED (expected false)");
    }

    // 2. Test has_incoming
    // Should be true for 2, 3, 4. False for 1.
    let in2 = (has_incoming(2))[0];
    let in1 = (has_incoming(1))[0];

    if in2 > 0.5 {
        println("has_incoming(2): OK");
    } else {
        println("has_incoming(2): FAILED");
    }

    if in1 < 0.5 {
         println("has_incoming(1): OK");
    } else {
         println("has_incoming(1): FAILED (expected false)");
    }
    
    // 3. Test wildcard in head (should probably not compile or mean unused arg, but let's test if parser accepts regular wildcards in body)
    // We used `edge(x, _)` above.

    println("WILDCARD_TEST_OK");
}
