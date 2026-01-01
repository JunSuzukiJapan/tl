struct S { x: Tensor<f32, 1> }

impl S {
    fn new() -> S {
        return S(randn([1], true));
    }
    
    fn step(self) -> S {
        let s = self;
        let v = randn([1], true);
        s.x = v;
        return s;
    }
}

fn main() {
    let obj = S::new();
    obj = obj.step();
    print("Done");
}
