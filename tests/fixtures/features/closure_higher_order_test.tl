// Test: higher-order functions
fn apply(f: Fn(i64) -> i64, x: i64) -> i64 {
    f(x)
}

fn main() {
    let double = |x: i64| x * 2;
    println("{}", apply(double, 5));  // → 10

    let offset: i64 = 100;
    let add_offset = |x: i64| x + offset;
    println("{}", apply(add_offset, 5));  // → 105
}
