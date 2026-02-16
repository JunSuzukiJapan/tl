
struct Linear { w: Tensor<i8, 2> }
struct MyTokenizer { _h: i64 }
struct MyKVCache { _h: i64 }
struct MyFile {} 

struct RMSNorm { w: Tensor<f32, 1>, e: f32 }
struct MLP { g: Linear, u: Linear, d: Linear }
struct Attention { q: Linear, k: Linear, v: Linear, o: Linear }

fn string_concat(a: String, b: String) -> String { a.concat(b) }
fn string_from_int(i: i64) -> String { String::from_int(i) }

impl Linear {
    fn from_map(m: Map, k: String) -> Linear { Linear { w: m.get_quantized(k) } }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { x.matmul_quantized(self.w) }
}

impl Attention {
    fn from_map(m: Map, p: String) -> Attention {
        Attention {
            q: Linear::from_map(m, string_concat(p, "_q.weight")),
            k: Linear::from_map(m, string_concat(p, "_k.weight")),
            v: Linear::from_map(m, string_concat(p, "_v.weight")),
            o: Linear::from_map(m, string_concat(p, "_output.weight"))
        }
    }
}

fn main() {
    println("Attention from_map defined. Start.");
}
