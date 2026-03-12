// Test: closure capturing outer variables
fn main() {
    let x: i64 = 10;
    let y: i64 = 20;

    // Single capture
    let add_x = |a: i64| a + x;
    println("{}", add_x(5));           // → 15

    // Multiple captures
    let add_xy = |a: i64| a + x + y;
    println("{}", add_xy(3));          // → 33
}
