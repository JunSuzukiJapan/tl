extern fn tl_gguf_load(path: String) -> i64;
extern fn tl_tokenizer_new(path: String) -> i64;
extern fn tl_tokenizer_encode(tok: i64, prompt: String) -> Tensor<i64, 1>;
extern fn tl_tokenizer_decode(tok: i64, ids: Tensor<i64, 1>) -> String;
extern fn tl_tensor_map_get(map: i64, key: String) -> Tensor<f32, 2>;
extern fn tl_tensor_map_get_1d(map: i64, key: String) -> Tensor<f32, 1>;
extern fn tl_tensor_embedding(idx: Tensor<i64, 1>, table: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_rms_norm(x: Tensor<f32, 2>, w: Tensor<f32, 1>, e: f32) -> Tensor<f32, 2>;
extern fn tl_tensor_matmul(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_add(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_silu(t: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_mul(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_scale(a: Tensor<f32, 4>, s: f32) -> Tensor<f32, 4>;
extern fn tl_tensor_transpose_2d(t: Tensor<f32, 2>, d1: i64, d2: i64) -> Tensor<f32, 2>;
extern fn tl_tensor_transpose(t: Tensor<f32, 4>, d1: i64, d2: i64) -> Tensor<f32, 4>;

extern fn tl_tensor_apply_rope(x: Tensor<f32, 4>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>) -> Tensor<f32, 4>;
extern fn tl_tensor_repeat_interleave(t: Tensor<f32, 4>, repeats: i64, dim: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_new_causal_mask(dim: i64) -> Tensor<f32, 2>;
extern fn tl_tensor_narrow(t: Tensor<f32, 2>, dim: i64, start: i64, len: i64) -> Tensor<f32, 2>;
extern fn tl_kv_cache_new(layers: i64) -> i64;
extern fn tl_kv_cache_free(cache: i64);
extern fn tl_kv_cache_get_k(cache: i64, layer: i64) -> Tensor<f32, 4>;
extern fn tl_kv_cache_get_v(cache: i64, layer: i64) -> Tensor<f32, 4>;
extern fn tl_kv_cache_update(cache: i64, layer: i64, k: Tensor<f32, 4>, v: Tensor<f32, 4>);
extern fn tl_tensor_cat_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>, d: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_rope_new_cos(d: i64, l: i64, t: f32) -> Tensor<f32, 2>;
extern fn tl_tensor_rope_new_sin(d: i64, l: i64, t: f32) -> Tensor<f32, 2>;
extern fn tl_tensor_argmax(t: Tensor<f32, 2>, d: i64, k: bool) -> Tensor<i64, 1>;
extern fn tl_tensor_len(t: Tensor<i64, 1>) -> i64;
extern fn tl_tensor_item_i64(t: Tensor<i64, 1>) -> i64;
extern fn tl_tensor_cat_i64(a: Tensor<i64, 1>, b: Tensor<i64, 1>, d: i64) -> Tensor<i64, 1>;
extern fn tl_tensor_sample(t: Tensor<f32, 2>, temp: f32, top_p: f32) -> Tensor<i64, 1>;
extern fn tl_string_contains(h: String, n: String) -> bool;
extern fn tl_read_line(prompt: String) -> String;
extern fn println(s: String);
extern fn print(s: String);
extern fn tl_string_concat(a: String, b: String) -> String;
extern fn tl_string_from_int(i: i64) -> String;
extern fn tl_tensor_matmul_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>) -> Tensor<f32, 4>;
extern fn tl_tensor_add_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>) -> Tensor<f32, 4>;
extern fn tl_tensor_softmax(t: Tensor<f32, 4>, d: i64) -> Tensor<f32, 4>;

extern fn tl_tensor_map_get_quantized(map: i64, key: String) -> i64;
extern fn tl_qtensor_matmul(input: Tensor<f32, 2>, weight: i64) -> Tensor<f32, 2>;

struct Linear { w: i64 }
impl Linear {
    fn from_map(m: i64, k: String) -> Linear { return Linear { w: tl_tensor_map_get_quantized(m, k) }; }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { 
        return tl_qtensor_matmul(x, self.w); 
    }
}
struct RMSNorm { w: Tensor<f32, 1>, e: f32 }
impl RMSNorm {
    fn new(m: i64, k: String) -> RMSNorm { return RMSNorm { w: tl_tensor_map_get_1d(m, k), e: 0.00001 }; }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { return tl_tensor_rms_norm(x, self.w, self.e); }
}
struct MLP { g: Linear, u: Linear, d: Linear }
impl MLP {
    fn from_map(m: i64, p: String) -> MLP {
        return MLP {
            g: Linear::from_map(m, tl_string_concat(p, "_gate.weight")),
            u: Linear::from_map(m, tl_string_concat(p, "_up.weight")),
            d: Linear::from_map(m, tl_string_concat(p, "_down.weight"))
        };
    }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        return self.d.forward(tl_tensor_mul(tl_tensor_silu(self.g.forward(x.clone())), self.u.forward(x.clone())));
    }
}
struct Attention { q: Linear, k: Linear, v: Linear, o: Linear }
impl Attention {
    fn from_map(m: i64, p: String) -> Attention {
        return Attention {
            q: Linear::from_map(m, tl_string_concat(p, "_q.weight")),
            k: Linear::from_map(m, tl_string_concat(p, "_k.weight")),
            v: Linear::from_map(m, tl_string_concat(p, "_v.weight")),
            o: Linear::from_map(m, tl_string_concat(p, "_output.weight"))
        };
    }
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: i64, li: i64, sp: i64) -> Tensor<f32, 2> {
        // Llama 3: 32 heads, 128 dim/head = 4096 hidden.
        // KV Heads: 8.
        let q_4d = self.q.forward(x.clone()).reshape([1, sl, 32, 128]);
        let k_4d = self.k.forward(x.clone()).reshape([1, sl, 8, 128]);
        let v_4d = self.v.forward(x.clone()).reshape([1, sl, 8, 128]);
        
        let cs = tl_tensor_narrow(cos, 0, sp, sl);
        let ss = tl_tensor_narrow(sin, 0, sp, sl);
        
        // RoPE dim is 128
        let qr = tl_tensor_apply_rope(q_4d, cs, ss);
        let kr = tl_tensor_apply_rope(k_4d, cs, ss);
        
        let kt = tl_tensor_transpose(kr, 1, 2);
        let vt = tl_tensor_transpose(v_4d, 1, 2);
        
        let k_tot = if sp == 0 { kt } else { tl_tensor_cat_4d(tl_kv_cache_get_k(ch, li), kt, 2) };
        let v_tot = if sp == 0 { vt } else { tl_tensor_cat_4d(tl_kv_cache_get_v(ch, li), vt, 2) };
        tl_kv_cache_update(ch, li, k_tot.clone(), v_tot.clone());
        
        let qt = tl_tensor_transpose(qr, 1, 2);
        
        // GQA: Repeat KV 4 times (32 heads / 8 kv heads = 4)
        let kw = tl_tensor_repeat_interleave(k_tot, 4, 1);
        let vw = tl_tensor_repeat_interleave(v_tot, 4, 1);
        
        // Scaling: 1 / sqrt(128) approx 0.08839
        let sm = if sl > 1 { tl_tensor_add_4d(tl_tensor_scale(tl_tensor_matmul_4d(qt, tl_tensor_transpose(kw, 2, 3)), 0.08839), tl_tensor_new_causal_mask(sl).reshape([1, 1, sl, sl])) } else { tl_tensor_scale(tl_tensor_matmul_4d(qt, tl_tensor_transpose(kw, 2, 3)), 0.08839) };
        let ao = tl_tensor_transpose(tl_tensor_matmul_4d(tl_tensor_softmax(sm, 3), vw), 1, 2);
        
        // 4096 hidden size
        return self.o.forward(ao.reshape([sl, 4096]));
    }
}
struct Block { an: RMSNorm, ffn: RMSNorm, at: Attention, mlp: MLP }
impl Block {
    fn from_map(m: i64, li: i64) -> Block {
        let p = tl_string_concat("blk.", tl_string_from_int(li));
        return Block {
            an: RMSNorm::new(m, tl_string_concat(p, ".attn_norm.weight")),
            ffn: RMSNorm::new(m, tl_string_concat(p, ".ffn_norm.weight")),
            at: Attention::from_map(m, tl_string_concat(p, ".attn")),
            mlp: MLP::from_map(m, tl_string_concat(p, ".ffn"))
        };
    }
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, batch_len: i64, ch: i64, li: i64, sp: i64) -> Tensor<f32, 2> {
        let h = tl_tensor_add(x.clone(), self.at.forward(self.an.forward(x.clone()), cos, sin, batch_len, ch, li, sp));
        return tl_tensor_add(h.clone(), self.mlp.forward(self.ffn.forward(h)));
    }
}
fn main() {
    println("Llama 3.1 8B Chatbot");
    let model_path = "~/.llm/models/models--lmstudio-community--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/8601e6db71269a2b12255ebdf09ab75becf22cc8/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
    let tokenizer_path = "~/.llm/models/llama3_tokenizer.json";
    
    let weights = tl_gguf_load(model_path);
    let tok = tl_tokenizer_new(tokenizer_path);
    
    // RoPE: Dim 128, Theta 500000.0
    let cos = tl_tensor_rope_new_cos(128, 8192, 500000.0);
    let sin = tl_tensor_rope_new_sin(128, 8192, 500000.0);
    
    let on = RMSNorm::new(weights, "output_norm.weight");
    println("Loaded output norm");
    let oh = Linear::from_map(weights, "output.weight");
    println("Loaded output head");
    let emb = tl_tensor_map_get(weights, "token_embd.weight");
    println("Loaded embedding");
    
    while true {
        let input_str = tl_read_line("User> ");
        if (input_str == "exit" || input_str == "quit") {
            return;
        }
        
        // Llama 3 Prompt Format
        // <|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        let p1 = "<|start_header_id|>user<|end_header_id|>\n\n";
        let p2 = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        let prompt = tl_string_concat(p1, tl_string_concat(input_str, p2));
        
        let tokens = tl_tokenizer_encode(tok, prompt);
        let cache = tl_kv_cache_new(32); // 32 Layers
        let cur_tokens = tokens.clone();
        let gen_idx = 0;
        let gen_str = "";
        print("Assistant> ");
        while gen_idx < 100 {
            let total_len = tl_tensor_len(tokens);
            let batch_len = tl_tensor_len(cur_tokens);
            let start_pos = total_len - batch_len;
            let xi = tl_tensor_embedding(cur_tokens, emb);
            let x = xi;
            let l = 0;
            // 32 Layers
            while l < 32 {
                let blk = Block::from_map(weights, l);
                x = blk.forward(x, cos, sin, batch_len, cache, l, start_pos);
                l = l + 1;
            }
            let lo = oh.forward(on.forward(x));
            // Sampling t=0.7, p=0.9
            let nxt = tl_tensor_sample(tl_tensor_narrow(lo, 0, batch_len - 1, 1), 0.7, 0.9);
            let s_nxt = tl_tokenizer_decode(tok, nxt);
            print(s_nxt);
            
            gen_str = tl_string_concat(gen_str, s_nxt);
            if tl_string_contains(gen_str, "User") {
                gen_idx = 200;
            }
            if tl_string_contains(gen_str, "Assistant") {
                gen_idx = 200;
            }
            // Llama 3 sometimes outputs <|eot_id|>
            if tl_string_contains(gen_str, "<|eot_id|>") {
                gen_idx = 200;
            }

            tokens = tl_tensor_cat_i64(tokens, nxt, 0);
            cur_tokens = nxt;
            gen_idx = gen_idx + 1;
        }
        println("");
        tl_kv_cache_free(cache);
    }
}
