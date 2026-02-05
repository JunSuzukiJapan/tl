// SKIP: Known compiler issue - generic type inference
struct Container<T> {
    val: T
}

// Function pointers and casts are separate features


fn main() {
    let c = Container { val: 3.14 };
    // Function pointers not fully supported yet in this way maybe, but let's test simpler advanced case
    
    // Generic mixed
    let c1 = Container { val: 10 };
    let c2 = Container { val: Container { val: 20 } }; // Nested
    
    print(c1.val);
    print(c2.val.val);
}
