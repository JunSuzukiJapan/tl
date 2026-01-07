// Exact nested struct + tensor crash reproduction
struct Inner { x: Tensor<f32, 1> }
struct Outer { inner: Inner }

impl Inner {
    fn new(val: f32) -> Inner {
        print("Inner::new");
        let t = Tensor::randn([4], false) * 0.0 + val;
        return Inner(t);
    }
}

impl Outer {
    fn new(val: f32) -> Outer {
        print("Outer::new start");
        let i = Inner::new(val);
        print("i.x before Outer:");
        print(i.x);
        print("Creating Outer with i");
        let o = Outer(i);
        print("Outer created, checking inner.x");
        print(o.inner.x);
        print("Returning Outer");
        return o;
    }
}

fn main() {
    print("Test nested Outer creation");
    let o = Outer::new(1.0);
    print("Got Outer in main");
    print("o.inner.x:");
    print(o.inner.x);
    print("Done!");
}
