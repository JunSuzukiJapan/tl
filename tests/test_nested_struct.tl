
struct Inner {
    val: i64,
}

impl Inner {
    fn new(v: i64) -> Inner {
        return Inner { val: v };
    }
}

struct Outer {
    a: Inner,
    b: Inner,
}

impl Outer {
    fn new(v1: i64, v2: i64) -> Outer {
        let i1 = Inner::new(v1);
        let i2 = Inner::new(v2);
        return Outer { a: i1, b: i2 };
    }
}

fn main() {
    print("Testing nested struct...");
    let o = Outer::new(10, 20);
    print("Nested struct created!");
}
