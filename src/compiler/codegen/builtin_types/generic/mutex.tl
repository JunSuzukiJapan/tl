struct Mutex<T> {
    id: i64,
}

impl<T> Mutex<T> {
    // new, read, modify, release are implicitly implemented via semantics and codegen
}
