// Ultra-minimal test to isolate sret nested struct issue
struct Inner { x: i64 }
struct Outer { inner: Inner }

impl Inner {
    fn new(val: i64) -> Inner {
        return Inner(val);
    }
}

impl Outer {
    fn new(val: i64) -> Outer {
        let i = Inner::new(val);
        print("Before Outer: i.x =");
        print(i.x);
        return Outer(i);
    }
}

fn main() {
    print("Test");
    let o = Outer::new(42);
    print("After return");
    print(o.inner.x);
}
