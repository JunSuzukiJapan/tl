struct Box<T> {
    inner: T,
}

impl Box<T> {
    fn new(val: T) -> Box<T> {
        Box { inner: val }
    }

    fn get_inner(self) -> T {
        self.inner
    }

    fn set_inner(self, val: T) {
        self.inner = val;
    }
}

fn main() {
    let b = Box::new(42);
    let val = b.get_inner();
    print(val);
    
    let b2 = Box::new(3.14);
    print(b2.get_inner());
}
