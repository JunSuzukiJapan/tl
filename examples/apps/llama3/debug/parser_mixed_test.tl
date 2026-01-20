struct Map { _d: i64 }
extern fn func1(path: String) -> i64;
extern fn func2(path: String) -> i64;

struct S1 { x: i64 }
impl S1 {
    fn new() -> S1 { S1 { x: 0 } }
}

extern fn func3(path: String) -> i64;
extern fn func4(path: String) -> i64;

fn main() {
    println("Success");
}
