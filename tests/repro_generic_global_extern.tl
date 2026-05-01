// STDOUT: OK
// Regression test: global generic extern fn declaration should compile
// Note: we don't call the extern fn since it has no implementation
extern fn my_identity<T>(val: T) -> T;

fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn main() {
    let x = add(10, 20);
    if x == 30 {
        print("OK\n");
    } else {
        print("FAIL\n");
    }
}
