
struct S { t: Tensor<f32, 1> }

impl S {
    fn new() -> S {
        let t = (Tensor::randn([5], true) * 0.0 + 1.0).detach();
        S { t: t }
    }
}

fn make_s() -> S {
    let s = S::new();
    s
}

fn main() {
    print("Testing UAF fix...");
    let s = make_s();
    print("Struct returned.");
    print("Accessing field t...");
    let t = s.t;
    print("Field accessed.");
    print(t);
    print("Success!");
}
