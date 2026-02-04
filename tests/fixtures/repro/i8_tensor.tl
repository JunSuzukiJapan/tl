// Minimal reproduction for matmul_quantized

struct Linear { w: Tensor<i8, 2> }

impl Linear {
    fn from_map(m: Map, k: String) -> Linear { 
        Linear { w: m.get_quantized(k) } 
    }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { 
        x.matmul_quantized(self.w) 
    }
}

fn main() {
    println("Testing matmul_quantized");
}
