fn main() {
    println("--- Tuple Test ---");

    // 1. Literal & Access
    let t = (10, 20);
    print(t.0); // 10
    print(t.1); // 20

    // 2. Mixed Types
    let t2 = (1, 3.14, true);
    print(t2.0); // 1
    print(t2.1); // 3.14
    if t2.2 {
        println("True");
    }

    // 3. String (Heap)
    let s = "Hello";
    let t3 = (s, "World");
    println(t3.0);
    println(t3.1);

    // 4. Nested
    let nested = ((1, 2), 3);
    print(nested.0.0); // 1
    print(nested.1); // 3

    println("--- Done ---");
}
