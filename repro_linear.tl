
struct Linear { w: Tensor<i8, 2> }

// --- Standard Library Wrapper Implementations ---

fn string_concat(a: String, b: String) -> String { a.concat(b) }
fn string_from_int(i: i64) -> String { String::from_int(i) }

// --- App Impls ---
impl Linear {
    fn from_map(m: Map, k: String) -> Linear { Linear { w: m.get_quantized(k) } }
    // tensor.matmul_quantized
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { 
        x.matmul_quantized(self.w) 
    }
}

fn main() {
    println("Linear impl defined. Start.");
}
