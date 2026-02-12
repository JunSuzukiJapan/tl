
// Full 22-layer forward pass with logits analysis
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
    let b2 = Block::from_map(weights, 2);
    let b3 = Block::from_map(weights, 3);
    let b4 = Block::from_map(weights, 4);
    let b5 = Block::from_map(weights, 5);
    let b6 = Block::from_map(weights, 6);
    let b7 = Block::from_map(weights, 7);
    let b8 = Block::from_map(weights, 8);
    let b9 = Block::from_map(weights, 9);
    let b10 = Block::from_map(weights, 10);
    let b11 = Block::from_map(weights, 11);
    let b12 = Block::from_map(weights, 12);
    let b13 = Block::from_map(weights, 13);
    let b14 = Block::from_map(weights, 14);
    let b15 = Block::from_map(weights, 15);
    let b16 = Block::from_map(weights, 16);
    let b17 = Block::from_map(weights, 17);
    let b18 = Block::from_map(weights, 18);
    let b19 = Block::from_map(weights, 19);
    let b20 = Block::from_map(weights, 20);
    let b21 = Block::from_map(weights, 21);

    let prompt = "A chat between a curious user and an assistant.\n\nUSER: What is the capital of France?\nASSISTANT:";
    let tokens = tok.encode(prompt);
    let cache = KVCache::new(22);
    let batch_len = tokens.len();

    println("Input tokens count:");
    println(batch_len);

    let xi = tokens.embedding(emb);
    println("Embedded, running 22 layers...");

    let mut x = xi.clone();
    x = b0.forward(x, cos, sin, batch_len, cache, 0, 0);
    x = b1.forward(x, cos, sin, batch_len, cache, 1, 0);
    x = b2.forward(x, cos, sin, batch_len, cache, 2, 0);
    x = b3.forward(x, cos, sin, batch_len, cache, 3, 0);
    x = b4.forward(x, cos, sin, batch_len, cache, 4, 0);
    x = b5.forward(x, cos, sin, batch_len, cache, 5, 0);
    x = b6.forward(x, cos, sin, batch_len, cache, 6, 0);
    x = b7.forward(x, cos, sin, batch_len, cache, 7, 0);
    x = b8.forward(x, cos, sin, batch_len, cache, 8, 0);
    x = b9.forward(x, cos, sin, batch_len, cache, 9, 0);
    x = b10.forward(x, cos, sin, batch_len, cache, 10, 0);
    x = b11.forward(x, cos, sin, batch_len, cache, 11, 0);
    x = b12.forward(x, cos, sin, batch_len, cache, 12, 0);
    x = b13.forward(x, cos, sin, batch_len, cache, 13, 0);
    x = b14.forward(x, cos, sin, batch_len, cache, 14, 0);
    x = b15.forward(x, cos, sin, batch_len, cache, 15, 0);
    x = b16.forward(x, cos, sin, batch_len, cache, 16, 0);
    x = b17.forward(x, cos, sin, batch_len, cache, 17, 0);
    x = b18.forward(x, cos, sin, batch_len, cache, 18, 0);
    x = b19.forward(x, cos, sin, batch_len, cache, 19, 0);
    x = b20.forward(x, cos, sin, batch_len, cache, 20, 0);
    x = b21.forward(x, cos, sin, batch_len, cache, 21, 0);

    println("All layers done. Running RMSNorm + LM head...");
    let normed = on.forward(x);
    println("RMSNorm output (first 10 vals):");
    normed.display();

    let lo = oh.forward(normed);
    let last_logits = lo.narrow(0, batch_len - 1, 1);
    
    println("Last token logits (first 20 vals of vocab):");
    last_logits.display();

    // Sample with temp=0.01 (near greedy)
    let nxt = last_logits.sample(0.01, 0.9);
    let tid = nxt.item_i64();
    println("Greedy token id:");
    println(tid);
    let decoded = tok.decode(nxt);
    println("Decoded:");
    println(decoded);

    // Generate 30 tokens with temp=0.6
    println("=== Generating 30 tokens ===");
    let mut total_len = batch_len;
    let mut cur = nxt.clone();
    let mut gi = 0;
    while gi < 30 {
        total_len = total_len + 1;
        let sp = total_len - 1;
        let xi2 = cur.embedding(emb);
        let mut x2 = xi2.clone();
        x2 = b0.forward(x2, cos, sin, 1, cache, 0, sp);
        x2 = b1.forward(x2, cos, sin, 1, cache, 1, sp);
        x2 = b2.forward(x2, cos, sin, 1, cache, 2, sp);
        x2 = b3.forward(x2, cos, sin, 1, cache, 3, sp);
        x2 = b4.forward(x2, cos, sin, 1, cache, 4, sp);
        x2 = b5.forward(x2, cos, sin, 1, cache, 5, sp);
        x2 = b6.forward(x2, cos, sin, 1, cache, 6, sp);
        x2 = b7.forward(x2, cos, sin, 1, cache, 7, sp);
        x2 = b8.forward(x2, cos, sin, 1, cache, 8, sp);
        x2 = b9.forward(x2, cos, sin, 1, cache, 9, sp);
        x2 = b10.forward(x2, cos, sin, 1, cache, 10, sp);
        x2 = b11.forward(x2, cos, sin, 1, cache, 11, sp);
        x2 = b12.forward(x2, cos, sin, 1, cache, 12, sp);
        x2 = b13.forward(x2, cos, sin, 1, cache, 13, sp);
        x2 = b14.forward(x2, cos, sin, 1, cache, 14, sp);
        x2 = b15.forward(x2, cos, sin, 1, cache, 15, sp);
        x2 = b16.forward(x2, cos, sin, 1, cache, 16, sp);
        x2 = b17.forward(x2, cos, sin, 1, cache, 17, sp);
        x2 = b18.forward(x2, cos, sin, 1, cache, 18, sp);
        x2 = b19.forward(x2, cos, sin, 1, cache, 19, sp);
        x2 = b20.forward(x2, cos, sin, 1, cache, 20, sp);
        x2 = b21.forward(x2, cos, sin, 1, cache, 21, sp);
        let lo2 = oh.forward(on.forward(x2));
        let nxt2 = lo2.narrow(0, 0, 1).sample(0.01, 0.9);
        let s2 = tok.decode(nxt2);
        print(s2);
        cur = nxt2.clone();
        gi = gi + 1;
    }
    println("");
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
}
