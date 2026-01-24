struct Box<T> {
    inner: T,
}

fn main() {
    let b = Box { inner: 10 };
}
