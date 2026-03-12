// Numeric methods integration test
// Tests: to_string, to_f64/to_i64/to_f32, min, max, clamp for i64

// STDOUT: [i64_to_string] 42
// STDOUT: [i64_min] 3
// STDOUT: [i64_max] 10
// STDOUT: [i64_clamp] 5

fn main() {
    // i64 to_string
    let x: i64 = 42;
    println("[i64_to_string] {}", x.to_string());

    // i64 min/max/clamp
    println("[i64_min] {}", 10.min(3));
    println("[i64_max] {}", 10.max(3));
    println("[i64_clamp] {}", 1.clamp(5, 15));
}
