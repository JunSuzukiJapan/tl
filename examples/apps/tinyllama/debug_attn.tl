
// Debug: Run 1 prefill + 2 decode steps, print intermediate values
fn main() {
    Param::set_device(Device::Auto);

    let model_path = "~/.llm/models/tinyllama/tinyllama-1.1b-chat-q4_0.gguf";
    let tokenizer_path = "~/.llm/models/tinyllama/tokenizer.json";

    let weights = Map::load(model_path);
    let tok = Tokenizer::new(tokenizer_path);
    let cos = Tensor::rope_new_cos(64, 2048, 10000.0);
    let sin = Tensor::rope_new_sin(64, 2048, 10000.0);
    let on = RMSNorm::new(weights, "output_norm.weight");
    let emb: Tensor<f32, 2> = weights.get("token_embd.weight");
    let oh_w: Tensor<f32, 2> = weights.get("output.weight");
    let oh = Linear { w: oh_w.transpose_2d(0, 1) };

    let b0 = Block::from_map(weights, 0);
    let b1 = Block::from_map(weights, 1);

    let prompt = "Hello";
    let tokens = tok.encode(prompt);
    let cache = KVCache::new(2);
    let batch_len = tokens.len();
    
    println("=== Prefill ===");
    println("batch_len:");
    println(batch_len);

    let xi = tokens.embedding(emb);
    let mut x = xi.clone();
    x = b0.forward(x, cos, sin, batch_len, cache, 0, 0);
    x = b1.forward(x, cos, sin, batch_len, cache, 1, 0);

    let lo = oh.forward(on.forward(x));
    let last_logits = lo.narrow(0, batch_len - 1, 1);
    
    let nxt = last_logits.sample(0.01, 0.9);
    let tid = nxt.item_i64();
    println("Predicted token after prefill:");
    println(tid);
    let decoded = tok.decode(nxt);
    println(decoded);

    // Decode step 1
    println("=== Decode Step 1 ===");
    let sp1 = batch_len;
    println("start_pos:");
    println(sp1);
    
    let xi2 = nxt.embedding(emb);
    let mut x2 = xi2.clone();
    x2 = b0.forward_debug(x2, cos, sin, 1, cache, 0, sp1);
    x2 = b1.forward(x2, cos, sin, 1, cache, 1, sp1);

    let lo2 = oh.forward(on.forward(x2));
    let nxt2 = lo2.narrow(0, 0, 1).sample(0.01, 0.9);
    let tid2 = nxt2.item_i64();
    println("Predicted token decode 1:");
    println(tid2);
    let d2 = tok.decode(nxt2);
    println(d2);

    // Decode step 2
    println("=== Decode Step 2 ===");
    let sp2 = batch_len + 1;
    println("start_pos:");
    println(sp2);
    
    let xi3 = nxt2.embedding(emb);
    let mut x3 = xi3.clone();
    x3 = b0.forward_debug(x3, cos, sin, 1, cache, 0, sp2);
    x3 = b1.forward(x3, cos, sin, 1, cache, 1, sp2);

    let lo3 = oh.forward(on.forward(x3));
    let nxt3 = lo3.narrow(0, 0, 1).sample(0.01, 0.9);
    let tid3 = nxt3.item_i64();
    println("Predicted token decode 2:");
    println(tid3);
    let d3 = tok.decode(nxt3);
    println(d3);

    println("=== Done ===");
    cache.free();
}

// --- Struct Definitions ---
struct Linear { w: Tensor<f32, 2> }
struct RMSNorm { w: Tensor<f32, 1>, e: f32 }
struct MLP { g: Linear, u: Linear, d: Linear }
struct Attention { q: Linear, k: Linear, v: Linear, o: Linear }
struct Block { an: RMSNorm, ffn: RMSNorm, at: Attention, mlp: MLP }

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
        MLP { g: g, u: u, d: d }
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
    
    fn forward_debug(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        let q_proj = self.q.forward(x.clone());
        let k_proj = self.k.forward(x.clone());
        let v_proj = self.v.forward(x.clone());
        
        let q_4d = q_proj.reshape([1, sl, 32, 64]);
        let k_4d = k_proj.reshape([1, sl, 4, 64]);
        let v_4d = v_proj.reshape([1, sl, 4, 64]);
        
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
        
        // Single token decode, no mask needed
        let mm = qt.matmul_4d(kw.transpose(2, 3));
        println("  scores (QK^T) first vals:");
        mm.display();
        
        let sc = mm.scale(0.125);
        let sm_soft = sc.softmax(3);
        
        println("  softmax probs first vals:");
        sm_soft.display();
        
        let ao = sm_soft.matmul_4d(vw).transpose(1, 2);
        
        self.o.forward(ao.reshape([sl, 2048]))
    }
}

impl Block {
    fn from_map(m: Map, i: i64) -> Block {
        let prefix = "blk.".concat(String::from_int(i));
        let an = RMSNorm::new(m, prefix.concat(".attn_norm.weight"));
        let ffn = RMSNorm::new(m, prefix.concat(".ffn_norm.weight"));
        let at = Attention::from_map(m, prefix.concat(".attn"));
        let mlp = MLP::from_map(m, prefix.concat(".ffn"));
        Block { an: an, ffn: ffn, at: at, mlp: mlp }
    }
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        let attn_out = self.at.forward(self.an.forward(x.clone()), cos, sin, sl, ch, li, sp);
        let x2 = x.add(attn_out);
        let ffn_out = self.mlp.forward(self.ffn.forward(x2.clone()));
        x2.add(ffn_out)
    }
    fn forward_debug(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        let normed = self.an.forward(x.clone());
        let attn_out = self.at.forward_debug(normed, cos, sin, sl, ch, li, sp);
        let x2 = x.add(attn_out);
        let ffn_out = self.mlp.forward(self.ffn.forward(x2.clone()));
        x2.add(ffn_out)
    }
}
