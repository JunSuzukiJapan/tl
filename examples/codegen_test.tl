fn main() {
    let a = [1.0, 2.0];
    let b = [-1.0, -2.0];
    let c = a + b;
    print(c); // Should be [0, 0]

    let d = -a; // Unary Neg matches F32 tensor types
    
    // Test logical not on bool
    let val = true;
    let not_val = !val;
    if not_val {
        print([0.0]);
    } else {
        print([1.0]);
    }

    // Let's test scalar control flow which prints a tensor.
    if true {
        print([100.0]);
    }

    // Test Loop
    let t = [10.0, 20.0, 30.0];
    for x in t {
        print([x]); // Print each element as tensor (assuming cast or wrapping works, or just print array literal)
    }
}
