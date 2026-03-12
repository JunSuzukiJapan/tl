// Builtins extra test: String methods, i32 methods, assert
// Tests new builtin methods added in MISSING_BUILTINS_ANALYSIS

// STDOUT: [upper] HELLO WORLD
// STDOUT: [lower] hello world
// STDOUT: [index_of] 6
// STDOUT: [index_of_neg] -1
// STDOUT: [split_len] 3
// STDOUT: [split_0] hello
// STDOUT: [split_1] world
// STDOUT: [split_2] foo
// STDOUT: [i32_to_string] 42
// STDOUT: [i32_min] 3
// STDOUT: [i32_max] 10
// STDOUT: [i32_clamp] 5
// STDOUT: [assert_pass] ok

fn main() {
    // String.to_uppercase / to_lowercase
    let s = "Hello World";
    println("[upper] {}", s.to_uppercase());
    println("[lower] {}", s.to_lowercase());

    // String.index_of
    let s2 = "hello world";
    println("[index_of] {}", s2.index_of("world"));
    println("[index_of_neg] {}", s2.index_of("xyz"));

    // String.split
    let csv = "hello,world,foo";
    let parts = csv.split(",");
    println("[split_len] {}", parts.len());
    println("[split_0] {}", parts.get(0).unwrap());
    println("[split_1] {}", parts.get(1).unwrap());
    println("[split_2] {}", parts.get(2).unwrap());

    // i32 methods (i32 values are just i64 in TL)
    let x: i64 = 42;
    println("[i32_to_string] {}", x.to_string());
    println("[i32_min] {}", 3.min(10));
    println("[i32_max] {}", 3.max(10));
    println("[i32_clamp] {}", 15.clamp(0, 5));

    // assert
    assert(true, "this should not fail");
    println("[assert_pass] ok");
}
