// Test: mutable closure capture
fn main() {
    // Immutable capture (existing - regression)
    let x: i64 = 10;
    let add_x = |a: i64| a + x;
    println("{}", add_x(5));           // → 15

    // Mutable capture
    let mut count: i64 = 0;
    let inc = |n: i64| { count = count + n; };
    inc(5);
    inc(3);
    println("{}", count);              // → 8
}
