extern fn tl_read_line(prompt: String) -> String;
extern fn tl_string_concat(a: String, b: String) -> String;
extern fn tl_string_from_int(n: i64) -> String;
extern fn tl_gguf_load(path: String) -> i64;
extern fn tl_tokenizer_new(path: String) -> i64;
extern fn tl_tokenizer_encode(tok: i64, prompt: String) -> Tensor<i64, 1>;
extern fn tl_tokenizer_decode(tok: i64, tokens: Tensor<i64, 1>) -> String;
extern fn tl_tensor_map_get(map: i64, key: String) -> Tensor<f32, 2>;
extern fn tl_tensor_map_get_1d(map: i64, key: String) -> Tensor<f32, 1>;
extern fn tl_tensor_embedding(id: Tensor<i64, 1>, w: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_rms_norm(x: Tensor<f32, 2>, w: Tensor<f32, 1>, e: f32) -> Tensor<f32, 2>;
extern fn tl_tensor_matmul(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_add(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_silu(x: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_mul(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_scale(t: Tensor<f32, 4>, s: f32) -> Tensor<f32, 4>;
extern fn tl_tensor_transpose_2d(t: Tensor<f32, 2>, d1: i64, d2: i64) -> Tensor<f32, 2>;
extern fn tl_tensor_transpose(t: Tensor<f32, 4>, d1: i64, d2: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_reshape_dims(t: Tensor<f32, 2>, d: Tensor<i64, 1>, n: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_reshape_2d(t: Tensor<f32, 4>, d: Tensor<i64, 1>, n: i64) -> Tensor<f32, 2>;
extern fn tl_tensor_apply_rope(x: Tensor<f32, 4>, c: Tensor<f32, 2>, s: Tensor<f32, 2>) -> Tensor<f32, 4>;
extern fn tl_tensor_narrow(t: Tensor<f32, 2>, d: i64, s: i64, l: i64) -> Tensor<f32, 2>;
extern fn tl_kv_cache_new(l: i64) -> i64;
extern fn tl_kv_cache_free(id: i64) -> void;
extern fn tl_kv_cache_update(id: i64, l: i64, k: Tensor<f32, 4>, v: Tensor<f32, 4>) -> void;
extern fn tl_kv_cache_get_k(id: i64, l: i64) -> Tensor<f32, 4>;
extern fn tl_kv_cache_get_v(id: i64, l: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_cat_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>, d: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_repeat_interleave(t: Tensor<f32, 4>, r: i64, d: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_matmul_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>) -> Tensor<f32, 4>;
extern fn tl_tensor_add_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>) -> Tensor<f32, 4>;
extern fn tl_tensor_softmax(t: Tensor<f32, 4>, d: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_new_causal_mask(l: i64) -> Tensor<f32, 2>;
extern fn tl_tensor_rope_new_cos(d: i64, l: i64, t: f32) -> Tensor<f32, 2>;
extern fn tl_tensor_rope_new_sin(d: i64, l: i64, t: f32) -> Tensor<f32, 2>;
extern fn tl_tensor_argmax(t: Tensor<f32, 2>, d: i64, k: bool) -> Tensor<i64, 1>;
extern fn tl_tensor_len(t: Tensor<i64, 1>) -> i64;
extern fn tl_tensor_item_i64(t: Tensor<i64, 1>) -> i64;
extern fn tl_tensor_cat_i64(a: Tensor<i64, 1>, b: Tensor<i64, 1>, d: i64) -> Tensor<i64, 1>;

struct Linear { w: Tensor<f32, 2> }
impl Linear {
    fn from_map(m: i64, k: String) -> Linear { return Linear { w: tl_tensor_map_get(m, k) }; }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { return tl_tensor_matmul(x, tl_tensor_transpose_2d(self.w, 0, 1)); }
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
        let q_4d = tl_tensor_reshape_dims(self.q.forward(x.clone()), [1, sl, 32, 64], 4);
        let k_4d = tl_tensor_reshape_dims(self.k.forward(x.clone()), [1, sl, 4, 64], 4);
        let v_4d = tl_tensor_reshape_dims(self.v.forward(x.clone()), [1, sl, 4, 64], 4);
        let cs = tl_tensor_narrow(cos, 0, sp, sl);
        let ss = tl_tensor_narrow(sin, 0, sp, sl);
        let qr = tl_tensor_apply_rope(q_4d, cs, ss);
        let kr = tl_tensor_apply_rope(k_4d, cs, ss);
        let kt = tl_tensor_transpose(kr, 1, 2);
        let vt = tl_tensor_transpose(v_4d, 1, 2);
        let k_tot = if sp == 0 { kt } else { tl_tensor_cat_4d(tl_kv_cache_get_k(ch, li), kt, 2) };
        let v_tot = if sp == 0 { vt } else { tl_tensor_cat_4d(tl_kv_cache_get_v(ch, li), vt, 2) };
        tl_kv_cache_update(ch, li, k_tot.clone(), v_tot.clone());
        let qt = tl_tensor_transpose(qr, 1, 2);
        let kw = tl_tensor_repeat_interleave(k_tot, 8, 1);
        let vw = tl_tensor_repeat_interleave(v_tot, 8, 1);
        let sm = if sl > 1 { tl_tensor_add_4d(tl_tensor_scale(tl_tensor_matmul_4d(qt, tl_tensor_transpose(kw, 2, 3)), 0.125), tl_tensor_reshape_dims(tl_tensor_new_causal_mask(sl), [1, 1, sl, sl], 4)) } else { tl_tensor_scale(tl_tensor_matmul_4d(qt, tl_tensor_transpose(kw, 2, 3)), 0.125) };
        let ao = tl_tensor_transpose(tl_tensor_matmul_4d(tl_tensor_softmax(sm, 3), vw), 1, 2);
        return self.o.forward(tl_tensor_reshape_2d(ao, [sl, 2048], 2));
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
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: i64, li: i64, sp: i64) -> Tensor<f32, 2> {
        let x2 = tl_tensor_add(x, self.at.forward(self.an.forward(x.clone()), cos, sin, sl, ch, li, sp));
        return tl_tensor_add(x2, self.mlp.forward(self.ffn.forward(x2.clone())));
    }
}
fn main() {
    println("TinyLlama Chatbot (Stable)");
    let weights = tl_gguf_load("/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf");
    let tok = tl_tokenizer_new("/Users/junsuzuki/.llm/models/tokenizer.json");
    let cos = tl_tensor_rope_new_cos(64, 2048, 10000.0);
    let sin = tl_tensor_rope_new_sin(64, 2048, 10000.0);
    let on = RMSNorm::new(weights, "output_norm.weight");
    let oh = Linear::from_map(weights, "output.weight");
    let emb = tl_tensor_map_get(weights, "token_embd.weight");
    
    while true {
        let input_str = tl_read_line("User> ");
        if input_str == "exit" { return 0; }
        
        let prompt = tl_string_concat("<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n", tl_string_concat(input_str, "</s>\n<|assistant|>\n"));
        let tokens = tl_tokenizer_encode(tok, prompt);
        let cache = tl_kv_cache_new(22);
        let cur_tokens = tokens.clone();
        let gen_idx = 0;
        print("Assistant> ");
        while gen_idx < 100 {
            let total_len = tl_tensor_len(tokens);
            let batch_len = tl_tensor_len(cur_tokens);
            let start_pos = total_len - batch_len;
            let xi = tl_tensor_embedding(cur_tokens, emb);
            let x = xi;
            let l = 0;
            while l < 22 {
                let blk = Block::from_map(weights, l);
                x = blk.forward(x, cos, sin, batch_len, cache, l, start_pos);
                l = l + 1;
            }
            let lo = oh.forward(on.forward(x));
            let nxt = tl_tensor_argmax(tl_tensor_narrow(lo, 0, batch_len - 1, 1), 1, false);
            let tid = tl_tensor_item_i64(nxt);
            print(tl_tokenizer_decode(tok, nxt));
            if tid == 2 { gen_idx = 200; }
            tokens = tl_tensor_cat_i64(tokens, nxt, 0);
            cur_tokens = nxt;
            gen_idx = gen_idx + 1;
        }
        tl_kv_cache_free(cache);
        println("");
    }
}
