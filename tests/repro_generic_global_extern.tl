// SKIP: Known compiler issue - generic type inference
extern fn my_identity<T>(val: T) -> T;

fn main() {
    let x = my_identity(10);
    // x should be 10.
    // Ensure we don't segfault.
}
