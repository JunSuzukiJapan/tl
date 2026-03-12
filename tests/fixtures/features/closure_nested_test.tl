// Test: nested closures
fn main() {
    let x: i64 = 10;
    let outer = |a: i64| {
        let inner = |b: i64| a + b + x;
        inner(20)
    };
    println("{}", outer(5));  // → 35 (5 + 20 + 10)
}
