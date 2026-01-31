
// --- Struct Definitions ---
struct Linear { w: Tensor<f32, 2> }
struct RMSNorm { w: Tensor<f32, 1>, e: f32 }
struct MLP { g: Linear, u: Linear, d: Linear }
struct Attention { q: Linear, k: Linear, v: Linear, o: Linear }
struct Block { an: RMSNorm, ffn: RMSNorm, at: Attention, mlp: MLP }

// --- App Impls ---
impl Linear {
    fn from_map(m: Map, k: String) -> Linear { 
        let t: Tensor<f32, 2> = m.get(k);
        Linear { w: t.transpose_2d(0, 1) } 
    }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { 
        x.matmul(self.w) 
    }
}

impl RMSNorm {
    fn new(m: Map, k: String) -> RMSNorm { RMSNorm { w: m.get_1d(k), e: 0.00001 } }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { x.rms_norm(self.w, self.e) }
}

impl MLP {
    fn from_map(m: Map, p: String) -> MLP {
        let g = Linear::from_map(m, p.concat("_gate.weight"));
        let u = Linear::from_map(m, p.concat("_up.weight"));
        let d = Linear::from_map(m, p.concat("_down.weight"));
        MLP {
            g: g,
            u: u,
            d: d
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

impl Attention {
    fn from_map(m: Map, p: String) -> Attention {
        let q = Linear::from_map(m, p.concat("_q.weight"));
        let k = Linear::from_map(m, p.concat("_k.weight"));
        let v = Linear::from_map(m, p.concat("_v.weight"));
        let o = Linear::from_map(m, p.concat("_output.weight"));
        Attention { q: q, k: k, v: v, o: o }
    }
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        let q_4d = self.q.forward(x.clone()).reshape([1, sl, 32, 64]);
        let k_4d = self.k.forward(x.clone()).reshape([1, sl, 4, 64]);
        let v_4d = self.v.forward(x.clone()).reshape([1, sl, 4, 64]);
        
        let cs = cos.narrow(0, sp, sl);
        let ss = sin.narrow(0, sp, sl);
        
        let qr = q_4d.apply_rope(cs, ss);
        let kr = k_4d.apply_rope(cs, ss);
        
        let kt = kr.transpose(1, 2);
        let vt = v_4d.transpose(1, 2);
        
        let k_tot = if sp == 0 { kt } else { ch.get_k(li).cat_4d(kt, 2) };
        let v_tot = if sp == 0 { vt } else { ch.get_v(li).cat_4d(vt, 2) };
        
        ch.update(li, k_tot.clone(), v_tot.clone());
        
        let qt = qr.transpose(1, 2);
        
        let kw = k_tot.repeat_interleave(8, 1);
        let vw = v_tot.repeat_interleave(8, 1);
        
        let sm = if sl > 1 { 
            let mm = qt.matmul_4d(kw.transpose(2, 3));
            let sc = mm.scale(0.125);
            let mask = Tensor::new_causal_mask(sl).reshape([1, 1, sl, sl]);
            sc.add_4d(mask)
        } else { 
            let mm = qt.matmul_4d(kw.transpose(2, 3));
            mm.scale(0.125)
        };
        
        let sm_soft = sm.softmax(3);
        let ao = sm_soft.matmul_4d(vw).transpose(1, 2);
        
        self.o.forward(ao.reshape([sl, 2048]))
    }
}

impl Block {
    fn from_map(m: Map, li: i64) -> Block {
        let p = "blk.".concat(String::from_int(li));
        let an = RMSNorm::new(m, p.concat(".attn_norm.weight"));
        let ffn = RMSNorm::new(m, p.concat(".ffn_norm.weight"));
        let at = Attention::from_map(m, p.concat(".attn"));
        let mlp = MLP::from_map(m, p.concat(".ffn"));
        Block {
            an: an,
            ffn: ffn,
            at: at,
            mlp: mlp
        }
    }
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        let attn_out = self.at.forward(self.an.forward(x.clone()), cos, sin, sl, ch, li, sp);
        let x2 = x.add(attn_out);
        let ffn_out = self.mlp.forward(self.ffn.forward(x2.clone()));
        x2.add(ffn_out)
    }
}

fn main() {
    println("Full Impl Check");
    /*
    Param::set_device(Device::Auto);
    ...
    */
}
