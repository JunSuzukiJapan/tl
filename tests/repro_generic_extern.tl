// STDOUT: OK
// Regression test: generic extern fn in impl block should compile
// Note: we don't call the extern fn since it has no implementation
struct Wrapper<T> {
    val: T,
}

impl<T> Wrapper<T> {
    extern fn my_new() -> Self;
    fn make(v: T) -> Wrapper<T> {
        Wrapper { val: v }
    }
}

fn main() {
    let w = Wrapper::make(42);
    if w.val == 42 {
        print("OK\n");
    } else {
        print("FAIL\n");
    }
}
