// Minimal reproduction of nested struct sret crash
struct Inner { x: i64 }
struct Outer { inner: Inner }

impl Inner {
    fn new(val: i64) -> Inner {
        print("Inner::new called");
        return Inner(val);
    }
}

impl Outer {
    fn new(val: i64) -> Outer {
        print("Outer::new start");
        let i = Inner::new(val);
        print("Inner created, value:");
        print(i.x);
        print("Creating Outer");
        let o = Outer(i);
        print("Outer created");
        return o;
    }
}

fn main() {
    print("Test nested sret");
    
    let o1 = Outer::new(42);
    print("o1 created");
    print("o1.inner.x:");
    print(o1.inner.x);
    
    print("Success!");
}
