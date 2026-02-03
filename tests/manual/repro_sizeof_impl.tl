struct S<T> { ptr: ptr<T> }

impl<T> S<T> {
    fn test() {
        let x = T::sizeof();
    }
}

fn main() {
    print("Run\n");
}
