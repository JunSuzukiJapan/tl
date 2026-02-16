
struct Linear { w: Tensor<i8, 2> }
struct MyTokenizer { _h: i64 }
struct MyKVCache { _h: i64 }
struct MyFile {} 

struct RMSNorm { w: Tensor<f32, 1>, e: f32 }
struct MLP { g: Linear, u: Linear, d: Linear }

fn string_concat(a: String, b: String) -> String { a.concat(b) }
fn string_from_int(i: i64) -> String { String::from_int(i) }

impl Linear {
    fn from_map(m: Map, k: String) -> Linear { Linear { w: m.get_quantized(k) } }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { x.matmul_quantized(self.w) }
}

impl RMSNorm {
    fn new(m: Map, k: String) -> RMSNorm { 
        RMSNorm { w: m.get_1d(k), e: 0.00001 } 
    }
    // Tensor::rms_norm
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { 
        x.rms_norm(self.w, self.e)
    }
}

impl MLP {
    fn from_map(m: Map, p: String) -> MLP {
        MLP {
            g: Linear::from_map(m, string_concat(p, "_gate.weight")),
            u: Linear::from_map(m, string_concat(p, "_up.weight")),
            d: Linear::from_map(m, string_concat(p, "_down.weight"))
        }
    }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let gx = self.g.forward(x.clone());
        let sx = gx.silu();
        let ux = self.u.forward(x.clone());
        let mu = sx.mul(ux);
        self.d.forward(mu)
    }
}

fn main() {
    println("MLP impl defined. Start.");
}
