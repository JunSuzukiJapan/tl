// Isolate tensor in nested struct issue step by step
struct Inner { x: Tensor<f32, 1> }
struct Outer { inner: Inner }

impl Inner {
    fn new(val: f32) -> Inner {
        print("Inner::new - creating tensor");
        let t = Tensor::randn([4], false) * 0.0 + val;
        print("Inner::new - tensor created, constructing Inner");
        let i = Inner(t);
        print("Inner::new - Inner constructed, returning");
        return i;
    }
}

fn main() {
    print("Test 1: Simple Inner return");
    let i1 = Inner::new(1.0);
    print("i1 returned to main");
    print(i1.x);
    
    print("Test 2: Inner stored in outer manually");
    let i2 = Inner::new(2.0);
    print("i2 returned to main");
    print(i2.x);
    let o = Outer(i2);
    print("Outer created in main");
    print(o.inner.x);
    
    print("All done!");
}
