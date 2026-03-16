
// --- Struct Definitions ---
struct Linear { w: Tensor<i8, 2> }
struct RMSNorm { w: Tensor<f32, 1>, e: f32 }
struct MLP { g: Linear, u: Linear, d: Linear }
struct Attention { q: Linear, k: Linear, v: Linear, o: Linear }
struct Block { an: RMSNorm, ffn: RMSNorm, at: Attention, mlp: MLP }

// --- App Impls ---
impl Linear {
    fn from_map(m: Map, k: String) -> Linear { Linear { w: m.get_quantized(k) } }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { 
        x.matmul_quantized(self.w) 
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
        // TinyLlama: 32 heads, 64 dim/head = 2048 hidden. KV Heads: 4.
        let q_4d = self.q.forward(x.clone()).reshape([1, sl, 32, 64]);
        let k_4d = self.k.forward(x.clone()).reshape([1, sl, 4, 64]);
        let v_4d = self.v.forward(x.clone()).reshape([1, sl, 4, 64]);
        
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
        
        let kw = k_tot.clone().repeat_interleave(8, 1);
        let vw = v_tot.clone().repeat_interleave(8, 1);
        
        let sm = if sl > 1 { 
            let mm = qt.clone().matmul_4d(kw.clone().transpose(2, 3));
            let sc = mm.scale(0.125);
            let mask = Tensor::new_causal_mask(sl).reshape([1, 1, sl, sl]);
            sc.add_4d(mask)
        } else { 
            let mm = qt.clone().matmul_4d(kw.clone().transpose(2, 3));
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
    println("TinyLlama Chatbot");
    Param::set_device(Device::Auto);

    let mut model_path = "~/.llm/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let tokenizer_path = "~/.llm/models/tinyllama/tokenizer.json";

    if !File::exists(model_path) {
        println("Model file not found at:");
        println(model_path);
        println("Downloading model...");
        
        let url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
        if File::download(url, model_path) {
            println("Download complete.");
        } else {
            println("Download failed.");
            println("Please try manual download from:");
            println(url);
            return;
        }
    }

    let weights = Map::load(model_path);

    if !File::exists(tokenizer_path) {
        println("Tokenizer not found. Downloading...");
        let tok_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json";
        if !File::download(tok_url, tokenizer_path) {
            println("Failed to download tokenizer.");
            return;
        }
        println("Tokenizer downloaded.");
    }

    let tok = Tokenizer::new(tokenizer_path);
    let cos = Tensor::rope_new_cos(64, 2048, 10000.0);
    let sin = Tensor::rope_new_sin(64, 2048, 10000.0);
    let on = RMSNorm::new(weights, "output_norm.weight");
    let oh = Linear::from_map(weights, "output.weight");
    let emb: Tensor<f32, 2> = weights.get("token_embd.weight");

    println("Loaded output norm");
    println("Loaded output head");
    println("Loaded embedding");
    
    while true {
        let input_str = read_line("User> ");
        if (input_str == "exit" || input_str == "quit") {
            return;
        }
        
        let mut tokens = tok.encode_chat_zephyr(input_str);
        let cache = KVCache::new(22);
        let mut cur_tokens = tokens.clone();
        let mut gen_idx = 0;
        let mut gen_str = "";
        let mut newline_count = 0;
        print("Assistant> ");
        while gen_idx < 100 {
            let total_len = tokens.len();
            let batch_len = cur_tokens.len();
            let start_pos = total_len - batch_len;
            let xi = cur_tokens.embedding(emb);
            let mut x = xi.clone();
            let mut l = 0;
            while l < 22 {
                let blk = Block::from_map(weights, l);
                x = blk.forward(x, cos.clone(), sin.clone(), batch_len, cache, l, start_pos);
                l = l + 1;
            }
            let lo = oh.forward(on.forward(x));
            // Apply repetition penalty to discourage template tag regeneration
            let lo_penalized = lo.repetition_penalty(tokens, 1.2 as f64);
            let nxt = lo_penalized.narrow(0, batch_len - 1, 1).sample(0.6, 0.9);
            let tid = nxt.item_i64();
            let s_nxt = tok.decode(nxt);
            
            // EOS token (</s> = 2)
            if tid == 2 { gen_idx = 200; }

            // 改行トークン (token_id=13) の連続検出
            if tid == 13 {
                newline_count = newline_count + 1;
                if newline_count >= 2 {
                    gen_idx = 200;
                }
            } else {
                newline_count = 0;
            }

            // 停止チェック（出力前）
            gen_str = gen_str.concat(s_nxt);
            if gen_str.contains("USER:") {
                gen_idx = 200;
            }
            if gen_str.contains("ASSISTANT:") {
                gen_idx = 200;
            }

            // 停止条件に該当しなければ出力
            if gen_idx < 200 {
                print(s_nxt);
            }
            tokens = tokens.cat_i64(nxt, 0);
            cur_tokens = nxt;
            gen_idx = gen_idx + 1;
        }
        cache.free();
        println("");
    }
}
