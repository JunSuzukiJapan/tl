
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
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        // Llama 3: 32 heads, 128 dim/head = 4096 hidden.
        // KV Heads: 8.
        let q_4d = self.q.forward(x.clone()).reshape([1, sl, 32, 128]);
        let k_4d = self.k.forward(x.clone()).reshape([1, sl, 8, 128]);
        let v_4d = self.v.forward(x.clone()).reshape([1, sl, 8, 128]);
        
        let cs = cos.clone().narrow(0, sp, sl);
        let ss = sin.clone().narrow(0, sp, sl);
        
        // RoPE
        let qr = q_4d.clone().apply_rope(cs.clone(), ss.clone()); 
        let kr = k_4d.clone().apply_rope(cs.clone(), ss.clone());
        
        let kt = kr.transpose(1, 2);
        let vt = v_4d.transpose(1, 2);
        
        let k_tot = if sp == 0 { kt } else { ch.get_k(li).cat_4d(kt.clone(), 2) };
        let v_tot = if sp == 0 { vt } else { ch.get_v(li).cat_4d(vt.clone(), 2) };
        
        ch.update(li, k_tot.clone(), v_tot.clone());
        
        let qt = qr.transpose(1, 2);
        
        let kw = k_tot.clone().repeat_interleave(4, 1);
        let vw = v_tot.clone().repeat_interleave(4, 1);
        
        let sm = if sl > 1 { 
            let mm = qt.clone().matmul_4d(kw.clone().transpose(2, 3));
            let sc = mm.scale(0.08839);
            let mask = Tensor::new_causal_mask(sl).reshape([1, 1, sl, sl]);
            sc.add_4d(mask)
        } else { 
            let mm = qt.clone().matmul_4d(kw.clone().transpose(2, 3));
            mm.scale(0.08839)
        };
        
        let sm_soft = sm.softmax(3);
        let ao = sm_soft.matmul_4d(vw).transpose(1, 2);
        
        self.o.forward(ao.reshape([sl, 4096]))
    }
}

fn main() {
    println("Attention impl defined. Start.");
}
