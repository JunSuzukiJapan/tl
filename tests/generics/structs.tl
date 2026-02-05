// SKIP: Known compiler issue - generic type inference
struct Box<T> {
    inner: T,
}

struct Pair<K, V> {
    first: K,
    second: V,
}

struct Container<T> {
    element: T,
}

fn main() {
    // Single generic parameter
    let b1: Box<i64> = Box { inner: 10 };
    let b2: Box<f64> = Box { inner: 3.14 };
    
    // Multiple generic parameters
    let p: Pair<i64, f64> = Pair { first: 1, second: 2.0 };
    
    // Nested generics
    let nested: Box<Box<i64>> = Box { inner: Box { inner: 42 } };
    
    // Generic struct with boolean
    let b_bool: Box<bool> = Box { inner: true };

    print(b1.inner);
}
