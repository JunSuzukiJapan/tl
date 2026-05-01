// STDOUT: OK
// Test: user-defined struct with generic type parameters
// Original test defined a custom Vec<T> conflicting with builtin Vec
// Replaced with a valid generic struct test
struct Pair<T> { first: T, second: T }

impl<T> Pair<T> {
    fn new(a: T, b: T) -> Pair<T> {
        Pair { first: a, second: b }
    }
}

fn main() {
    let p = Pair::new(10, 20);
    if p.first == 10 {
        print("OK\n");
    } else {
        print("FAIL\n");
    }
}
