// Test: struct return from impl method (like Linear::new)

struct Container {
    t: Tensor<f32, 2>,
    s: Tensor<f32, 1>,
}

impl Container {
    fn create(r: i64, c: i64) -> Container {
        print("Container::create: creating tensors with mul only, no detach...");
        let t = Tensor::randn([r, c], false) * 0.001;  // mul but no detach
        print("Container::create: t =");
        print(t);
        
        let s = Tensor::zeros([c], false);  // no operation
        print("Container::create: s =");
        print(s);
        
        print("Container::create: constructing Container...");
        let result = Container(t, s);
        print("Container::create: result.t =");
        print(result.t);
        print("Container::create: returning...");
        result
    }
}

fn main() {
    print("=== Impl Static Method Return Test ===");
    let c = Container::create(4, 3);
    print("main: After return, c.t =");
    print(c.t);
    print("main: c.s =");
    print(c.s);
}
