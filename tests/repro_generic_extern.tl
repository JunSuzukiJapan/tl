struct Wrapper<T> {
    val: T,
}

impl<T> Wrapper<T> {
    extern fn my_new() -> Self;
}

fn test() {
    let w: Wrapper<i64> = Wrapper::my_new();
}

fn main() {
    test();
}
