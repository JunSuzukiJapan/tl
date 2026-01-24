struct Wrapper<T> {
    val: T
}

impl<T> Wrapper<T> {
    fn new(v: T) -> Self {
        Wrapper { val: v }
    }
}

fn main() {
    let w = Wrapper<i64>::new(42);
    print(w.val);
}
