struct Map { _d: i64 }
extern fn func1(path: String) -> i64;
extern fn func2(path: String) -> i64;

struct S1 { x: i64 }

// Removed impl block to isolate declaration parsing
extern fn func3(path: String) -> i64;
extern fn func4(path: String) -> i64;

fn main() {
    println("Success");
}
