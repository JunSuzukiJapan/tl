fn main() {
    println("--- Testing Char Comprehensive ---");

    let s = "ABC";
    let c_a = s.char_at(0);
    let c_b = s.char_at(1);
    let c_c = s.char_at(2);

    // 1. Printing
    print("c_a: "); // print without newline
    println(c_a);   // println Char
    println("c_b: {}", c_b); // Formatted print

    // 2. Comparison
    // Assuming Char supports equality check
    if c_a == c_a {
        println("c_a == c_a: OK");
    } else {
        println("c_a == c_a: FAIL");
    }

    if c_a != c_b {
        println("c_a != c_b: OK");
    } else {
        println("c_a != c_b: FAIL");
    }

    // 3. Logic
    // Can we compare char with int? Probably not directly without cast, but let's stick to Char-Char for now
    let s2 = "A";
    let c_a2 = s2.char_at(0);
    if c_a == c_a2 {
        println("Char 'A' equality from different strings: OK");
    } else {
        println("Char 'A' equality from different strings: FAIL");
    }

    println("--- Char Tests Complete ---");
}
