// Isolate tensor crash in nested struct
struct Inner { x: Tensor<f32, 1> }

impl Inner {
    fn new(val: f32) -> Inner {
        print("Inner::new start");
        let t = Tensor::randn([4], false) * 0.0 + val;
        print("Tensor created");
        let i = Inner(t);
        print("Inner constructed");
        return i;
    }
}

fn main() {
    print("Test tensor in struct sret");
    
    let i = Inner::new(1.0);
    print("Inner returned to main");
    print("i.x:");
    print(i.x);
    
    print("Success!");
}
