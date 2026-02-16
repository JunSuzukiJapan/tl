
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
    // body 3
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        let q_out = self.q.forward(x.clone());
        let q_4d = q_out.reshape([1, sl, 32, 128]);
        
        let cs = cos.clone().narrow(0, sp, sl);
        let ss = sin.clone().narrow(0, sp, sl);
        
        // RoPE
        let qr = q_4d.clone().apply_rope(cs.clone(), ss.clone());

        // Test KVCache
        let k_cache = ch.get_k(li); 
        ch.update(li, x.clone(), x.clone());

        x
    }
}

fn main() {
    println("Attention body bisect 3. Start.");
}
