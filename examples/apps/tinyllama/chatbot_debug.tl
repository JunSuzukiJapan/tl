
// Debug: Full Attention + KVCache + Loop (simplified to 1 block)
struct Linear { w: Tensor<f32, 2> }
struct RMSNorm { w: Tensor<f32, 1>, e: f32 }
struct Attention { q: Linear, k: Linear, v: Linear, o: Linear }

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

impl Attention {
    fn from_map(m: Map, p: String) -> Attention {
        let q = Linear::from_map(m, p.concat("_q.weight"));
        let k = Linear::from_map(m, p.concat("_k.weight"));
        let v = Linear::from_map(m, p.concat("_v.weight"));
        let o = Linear::from_map(m, p.concat("_output.weight"));
        Attention { q: q, k: k, v: v, o: o }
    }
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        println("  Attn: accessing self.q...");
        let lq = self.q;
        println("  Attn: self.q accessed, calling forward...");
        let q_4d = lq.forward(x.clone()).reshape([1, sl, 32, 64]);
        println("  Attn: K forward...");
        let k_4d = self.k.forward(x.clone()).reshape([1, sl, 4, 64]);
        println("  Attn: V forward...");
        let v_4d = self.v.forward(x.clone()).reshape([1, sl, 4, 64]);
        
        println("  Attn: narrow cos/sin...");
        let cs = cos.narrow(0, sp, sl);
        let ss = sin.narrow(0, sp, sl);
        
        println("  Attn: apply_rope...");
        let qr = q_4d.apply_rope(cs, ss);
        let kr = k_4d.apply_rope(cs, ss);
        
        println("  Attn: transpose...");
        let kt = kr.transpose(1, 2);
        let vt = v_4d.transpose(1, 2);
        
        println("  Attn: KV cache...");
        let k_tot = if sp == 0 { kt } else { ch.get_k(li).cat_4d(kt, 2) };
        let v_tot = if sp == 0 { vt } else { ch.get_v(li).cat_4d(vt, 2) };
        
        ch.update(li, k_tot.clone(), v_tot.clone());
        
        println("  Attn: attention computation...");
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
        
        println("  Attn: softmax...");
        let sm_soft = sm.softmax(3);
        println("  Attn: final matmul...");
        let ao = sm_soft.matmul_4d(vw).transpose(1, 2);
        
        println("  Attn: output projection...");
        self.o.forward(ao.reshape([sl, 2048]))
    }
}

fn main() {
    println("Debug: Starting");
    Param::set_device(Device::Auto);

    let model_path = "~/.llm/models/tinyllama/tinyllama-1.1b-chat-q4_0.gguf";
    let tokenizer_path = "~/.llm/models/tinyllama/tokenizer.json";

    let weights = Map::load(model_path);
    let tok = Tokenizer::new(tokenizer_path);
    println("Debug: Model loaded");

    let cos = Tensor::rope_new_cos(64, 2048, 10000.0);
    let sin = Tensor::rope_new_sin(64, 2048, 10000.0);
    let on = RMSNorm::new(weights, "output_norm.weight");
    let emb: Tensor<f32, 2> = weights.get("token_embd.weight");
    let oh_w: Tensor<f32, 2> = weights.get("output.weight");
    let oh = Linear { w: oh_w.transpose_2d(0, 1) };
    let at0 = Attention::from_map(weights, "blk.0.attn");
    let an0 = RMSNorm::new(weights, "blk.0.attn_norm.weight");
    println("Debug: Weights loaded");

    let prompt = "A chat between a curious user and an assistant.\n\nUSER: Hello\nASSISTANT:";
    let mut tokens = tok.encode(prompt);
    let cache = KVCache::new(22);
    let mut cur_tokens = tokens.clone();
    println("Debug: Tokens encoded, starting generation loop");

    // Iteration 1: Prompt (multi-token)
    println("=== Iteration 1 (prompt) ===");
    let total_len = tokens.len();
    let batch_len = cur_tokens.len();
    let start_pos = total_len - batch_len;
    println("  batch_len = "); println(String::from_int(batch_len));
    
    let xi = cur_tokens.embedding(emb);
    println("  embedding done");
    
    let normed = an0.forward(xi);
    println("  RMSNorm done");

    let attn_out = at0.forward(normed, cos, sin, batch_len, cache, 0, start_pos);
    println("  Attention block 0 done");
    
    let lo = oh.forward(on.forward(attn_out));
    println("  Output forward done");
    
    let nxt = lo.narrow(0, batch_len - 1, 1).sample(0.5, 0.9);
    let tid = nxt.item_i64();
    println("  Token: "); println(String::from_int(tid));

    // Iteration 2: Single-token (this is where segfault likely happens)
    println("=== Iteration 2 (single token) ===");
    tokens = tokens.cat_i64(nxt, 0);
    cur_tokens = nxt;
    
    let total_len2 = tokens.len();
    let batch_len2 = cur_tokens.len();
    let start_pos2 = total_len2 - batch_len2;
    println("  batch_len2 = "); println(String::from_int(batch_len2));
    println("  start_pos2 = "); println(String::from_int(start_pos2));
    
    let xi2 = cur_tokens.embedding(emb);
    println("  embedding done");
    
    let normed2 = an0.forward(xi2);
    println("  RMSNorm done");

    let attn_out2 = at0.forward(normed2, cos, sin, batch_len2, cache, 0, start_pos2);
    println("  Attention block 0 done");
    
    let lo2 = oh.forward(on.forward(attn_out2));
    println("  Output forward done");
    
    let nxt2 = lo2.narrow(0, batch_len2 - 1, 1).sample(0.5, 0.9);
    let tid2 = nxt2.item_i64();
    println("  Token: "); println(String::from_int(tid2));
    
    cache.free();
    println("=== ALL DONE ===");
}
