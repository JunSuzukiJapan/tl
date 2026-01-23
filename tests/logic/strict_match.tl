
edge(a, b, 1.5).

fn main() {
    println("--- Strict Matching Test ---");
    // 'a' has ID 0, 'b' has ID 1.
    // But querying with literals 0 and 1 (Int) should NOT match Entity(0) and Entity(1).
    let res = ?edge(0, 1, $w);
    println("edge(0, 1, $w) matches (should be empty):");
    println(res);
    
    let res2 = ?edge(a, b, $w);
    println("edge(a, b, $w) matches (should be 1.5):");
    println(res2);
}
