// Two tensor (same rank) fields test

struct TwoTensors {
    a: Tensor<f32, 2>,
    b: Tensor<f32, 2>,
}

impl TwoTensors {
    fn new_tensors(x: i64, y: i64) -> TwoTensors {
        print("Creating tensor a (WITH MUL)...");
        let ta = Tensor::randn([x, y], false) * 0.001;  // Added multiplication
        print("Tensor a created:");
        print(ta);
        print("Creating tensor b (WITH MUL)...");
        let tb = Tensor::randn([y, x], false) * 0.002;  // Added multiplication
        print("Tensor b created:");
        print(tb);
        print("Constructing TwoTensors...");
        let result = TwoTensors(ta, tb);
        print("TwoTensors constructed.");
        result
    }
    
    fn get_a(self) -> Tensor<f32, 2> {
        print("get_a: accessing self.a...");
        let a = self.a;
        print("get_a: self.a =");
        print(a);
        a
    }
    
    fn get_b(self) -> Tensor<f32, 2> {
        print("get_b: accessing self.b...");
        let b = self.b;
        print("get_b: self.b =");
        print(b);
        b
    }
}

fn main() {
    print("=== Two Tensors Test ===");
    let t = TwoTensors::new_tensors(2, 3);
    print("TwoTensors created, calling get_a...");
    let a = t.get_a();
    print("Final a:");
    print(a);
    print("Calling get_b...");
    let b = t.get_b();
    print("Final b:");
    print(b);
}
