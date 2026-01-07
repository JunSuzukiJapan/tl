// Test for sret with nested structs containing tensors
struct Inner { x: Tensor<f32, 1> }
struct Outer { inner: Inner }

impl Inner {
    fn new(val: f32) -> Inner {
        let t = Tensor::randn([4], false) * 0.0 + val;
        return Inner(t);
    }
    
    fn step(self, lr: f32) -> Inner {
        let new_x = self.x + lr;
        return Inner(new_x);
    }
}

impl Outer {
    fn new(val: f32) -> Outer {
        let i = Inner::new(val);
        return Outer(i);
    }
    
    fn step(self, lr: f32) -> Outer {
        let new_inner = self.inner.step(lr);
        return Outer(new_inner);
    }
}

fn main() {
    print("Test sret with nested structs");
    
    let o1 = Outer::new(1.0);
    print("o1.inner.x:");
    print(o1.inner.x);
    
    let o2 = o1.step(0.5);
    print("o2.inner.x (should be ~1.5):");
    print(o2.inner.x);
    
    print("Success!");
}
