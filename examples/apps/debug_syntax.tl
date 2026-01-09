
struct Tokenizer {}

extern fn test_extern(s: String) -> Tokenizer;
extern fn test_tensor(t: Tensor<f32, 2>);

struct RMSNorm {
    weight: Tensor<f32, 1>,
    eps: f32,
}

impl RMSNorm {
    fn new(w: Tensor<f32, 1>) -> RMSNorm {
        return RMSNorm(w, 0.00001);
    }
}

fn main() {
    let s = "hello";
    print(s);
    if !false {
        print("Not False is True");
    }
}
