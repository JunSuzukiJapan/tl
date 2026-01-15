// Test with Tensor field in struct

struct Linear {
    weight: Tensor<f32, 2>,
}

impl Linear {
    fn new(w: Tensor<f32, 2>) -> Linear {
        return Linear { weight: w };
    }
}

fn main() {
    print("Testing tensor struct...");
    print("Done!");
}
