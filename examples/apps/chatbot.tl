
// --- Externs ---
extern fn tl_http_download(url: String, path: String) -> bool;
extern fn tl_path_exists(path: String) -> bool;
extern fn tl_file_read_string(path: String) -> String;
extern fn tl_string_from_int(val: i64) -> String;
extern fn tl_string_concat(s1: String, s2: String) -> String;
extern fn tl_read_line(prompt: String) -> String;
extern fn tl_prompt(prompt: String) -> String;

extern fn tl_tokenizer_new(path: String) -> i64;
extern fn tl_tokenizer_encode(tok: i64, prompt: String) -> Tensor<i64, 1>;
extern fn tl_tokenizer_decode(tok: i64, ids: Tensor<i64, 1>) -> String;

extern fn tl_gguf_load(path: String) -> i64;
extern fn tl_tensor_map_get(map: i64, name: String) -> Tensor<f32, 2>;
extern fn tl_tensor_map_get_1d(map: i64, name: String) -> Tensor<f32, 1>;

extern fn tl_tensor_matmul(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_matmul_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>) -> Tensor<f32, 4>;
extern fn tl_tensor_add_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>) -> Tensor<f32, 4>;
extern fn tl_tensor_scale(t: Tensor<f32, 4>, s: Tensor<f32, 1>) -> Tensor<f32, 4>;
extern fn tl_tensor_silu(t: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_silu_4d(t: Tensor<f32, 4>) -> Tensor<f32, 4>;
extern fn tl_tensor_rms_norm(x: Tensor<f32, 2>, w: Tensor<f32, 1>, eps: f32) -> Tensor<f32, 2>;
extern fn tl_tensor_apply_rope(x: Tensor<f32, 4>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>) -> Tensor<f32, 4>;
extern fn tl_tensor_softmax(t: Tensor<f32, 4>, dim: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_transpose_2d(t: Tensor<f32, 2>, d0: i64, d1: i64) -> Tensor<f32, 2>;
extern fn tl_tensor_transpose(t: Tensor<f32, 4>, d0: i64, d1: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_cat2(a: Tensor<f32, 2>, b: Tensor<f32, 2>, dim: i64) -> Tensor<f32, 2>;
extern fn tl_tensor_reshape_dims(t: Tensor<f32, 2>, dims: Tensor<i64, 1>, n: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_get_shape(t: Tensor<f32, 2>) -> Tensor<i64, 1>;
extern fn tl_tensor_device_id(t: Tensor<f32, 2>) -> i64;

extern fn tl_tensor_rope_new_cos(dim: i64, len: i64, theta: f32) -> Tensor<f32, 2>;
extern fn tl_tensor_rope_new_sin(dim: i64, len: i64, theta: f32) -> Tensor<f32, 2>;
extern fn tl_tensor_embedding(tokens: Tensor<i64, 1>, weights: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_add(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_reshape_3d_to_2d(t: Tensor<f32, 3>, dims: Tensor<i64, 1>, n: i64) -> Tensor<f32, 2>;

// KV Cache Externs
extern fn tl_kv_cache_new(layers: i64) -> i64;
extern fn tl_kv_cache_free(cache: i64);
extern fn tl_kv_cache_get_k(cache: i64, layer: i64) -> Tensor<f32, 4>;
extern fn tl_kv_cache_get_v(cache: i64, layer: i64) -> Tensor<f32, 4>;
extern fn tl_kv_cache_update(cache: i64, layer: i64, k: Tensor<f32, 4>, v: Tensor<f32, 4>);

// Missing Ops
extern fn tl_tensor_cat_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>, dim: i64) -> Tensor<f32, 4>;

// RoPE/GQA support
extern fn tl_tensor_narrow(t: Tensor<f32, 2>, dim: i64, start: i64, length: i64) -> Tensor<f32, 2>;
extern fn tl_tensor_repeat_interleave(t: Tensor<f32, 4>, repeats: i64, dim: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_argmax(t: Tensor<f32, 2>, dim: i64, keep_dim: bool) -> Tensor<i64, 1>;
extern fn tl_tensor_new_causal_mask(dim: i64) -> Tensor<f32, 2>;
extern fn tl_tensor_cat_i64(a: Tensor<i64, 1>, b: Tensor<i64, 1>, dim: i64) -> Tensor<i64, 1>;
extern fn tl_tensor_len(t: Tensor<i64, 1>) -> i64;
extern fn tl_tensor_item_i64(t: Tensor<i64, 1>) -> i64;
extern fn tl_print_ptr(p: Tensor<i64, 1>); 

// --- Structs ---

struct RMSNorm {
    weight: Tensor<f32, 1>,
    eps: f32,
}

impl RMSNorm {
    fn new(map: i64, key: String) -> RMSNorm {
        let w = tl_tensor_map_get_1d(map, key);
        return RMSNorm { weight: w, eps: 0.00001 };
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        return tl_tensor_rms_norm(x, self.weight, self.eps);
    }
}

struct Linear {
    weight: Tensor<f32, 2>, 
}

impl Linear {
    fn from_map(map: i64, key: String) -> Linear {
        let w = tl_tensor_map_get(map, key);
        return Linear { weight: w }; 
    }

    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let wt = tl_tensor_transpose_2d(self.weight, 0, 1);
        let out = tl_tensor_matmul(x, wt);
        return out;
    }
}

struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    fn from_map(map: i64, prefix: String) -> MLP {
        // GGUF naming: blk.X.ffn_gate.weight, blk.X.ffn_up.weight, blk.X.ffn_down.weight
        let gate = Linear::from_map(map, tl_string_concat(prefix, "_gate.weight"));
        let up = Linear::from_map(map, tl_string_concat(prefix, "_up.weight"));
        let down = Linear::from_map(map, tl_string_concat(prefix, "_down.weight"));
        return MLP { gate_proj: gate, up_proj: up, down_proj: down };
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let g = self.gate_proj.forward(x.clone());
        let u = self.up_proj.forward(x);
        let silu_g = tl_tensor_silu(g);
        let gu = silu_g * u; 
        let d = self.down_proj.forward(gu);
        return d;
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    
    n_heads: i64,
    head_dim: i64,
}

impl Attention {
    fn from_map(map: i64, prefix: String) -> Attention {
        let q = Linear::from_map(map, tl_string_concat(prefix, "_q.weight"));
        let k = Linear::from_map(map, tl_string_concat(prefix, "_k.weight"));
        let v = Linear::from_map(map, tl_string_concat(prefix, "_v.weight"));
        let o = Linear::from_map(map, tl_string_concat(prefix, "_output.weight"));
        
        return Attention { 
            q_proj: q, k_proj: k, v_proj: v, o_proj: o, 
            n_heads: 32, head_dim: 64
        };
    }
    
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, seq_len: i64, cache: i64, layer_idx: i64, start_pos: i64) -> Tensor<f32, 2> {
        // x: [seq_len, hidden_size]
        // 1. QKV Projections
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x.clone());
        
        let batch_size = 1;
        
        // 2. Reshape & Transpose for RoPE and Attention
        let q_shape = [batch_size, seq_len, 32, 64];
        let k_shape = [batch_size, seq_len, 4, 64];
        let v_shape = [batch_size, seq_len, 4, 64];
        
        let q_4d = tl_tensor_reshape_dims(q, q_shape, 4);
        let k_4d = tl_tensor_reshape_dims(k, k_shape, 4);
        let v_4d = tl_tensor_reshape_dims(v, v_shape, 4);
        
        // 3. RoPE
        // Need to slice cos/sin based on start_pos and seq_len
        let cos_slice = tl_tensor_narrow(cos, 0, start_pos, seq_len);
        let sin_slice = tl_tensor_narrow(sin, 0, start_pos, seq_len);
        
        // apply rope
        let q_rope = tl_tensor_apply_rope(q_4d, cos_slice, sin_slice);
        let k_rope = tl_tensor_apply_rope(k_4d, cos_slice, sin_slice);
        
        // 4. KV Cache Update
        // Transpose K, V to [batch, heads, seq, dim] first
        let k_trans = tl_tensor_transpose(k_rope, 1, 2);
        let v_trans = tl_tensor_transpose(v_4d, 1, 2);
        
        // KV Cache logic temporarily disabled for debugging
        let k_total = k_trans;
        let v_total = v_trans;
        
        // Update Cache
        tl_kv_cache_update(cache, layer_idx, k_total.clone(), v_total.clone());
        
        return x;
    }
}

struct TransformerBlock {
    attn_norm: RMSNorm,
    ffn_norm: RMSNorm,
    attn: Attention,
    mlp: MLP,
}

impl TransformerBlock {
    fn from_map(map: i64, layer_idx: i64) -> TransformerBlock {
        let prefix = tl_string_concat("blk.", tl_string_from_int(layer_idx));
        
        let attn_norm = RMSNorm::new(map, tl_string_concat(prefix, ".attn_norm.weight"));
        let ffn_norm = RMSNorm::new(map, tl_string_concat(prefix, ".ffn_norm.weight"));
        let attn = Attention::from_map(map, tl_string_concat(prefix, ".attn"));
        let mlp = MLP::from_map(map, tl_string_concat(prefix, ".ffn"));
        
        return TransformerBlock { 
            attn_norm: attn_norm, 
            ffn_norm: ffn_norm, 
            attn: attn, 
            mlp: mlp 
        };
    }
    
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, seq_len: i64, cache: i64, layer_idx: i64, start_pos: i64) -> Tensor<f32, 2> {
        // Pre-norm attention
        let h = self.attn_norm.forward(x.clone());
        let attn_out = self.attn.forward(h, cos, sin, seq_len, cache, layer_idx, start_pos);
        let x2 = tl_tensor_add(x, attn_out);
        
        // Pre-norm FFN
        let h2 = self.ffn_norm.forward(x2.clone());
        let ffn_out = self.mlp.forward(h2);
        let out = tl_tensor_add(x2, ffn_out);
        
        return out;
    }
}

extern fn tl_tensor_reshape_2d(t: Tensor<f32, 4>, dims: Tensor<i64, 1>, n: i64) -> Tensor<f32, 2>;

fn main() {
    print("DEBUG: main() start");
    print("TinyLlama Chatbot Verification (Metal + KV Cache)");
    print("Device initialized.");
    
    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    let tokenizer_path = "/Users/junsuzuki/.llm/models/tokenizer.json";
    
    print("Loading GGUF...");
    let weights = tl_gguf_load(model_path);
    print("Loading Tokenizer...");
    let tok = tl_tokenizer_new(tokenizer_path);
    print("Loaded!");

    print("Precomputing RoPE...");
    let cos = tl_tensor_rope_new_cos(64, 2048, 10000.0);
    let sin = tl_tensor_rope_new_sin(64, 2048, 10000.0);
    
    print("Loading embedding weights...");
    // Chat Loop
    while true {
       let user_input = tl_read_line("User> ");
       
       let prompt = "<|user|>\n";
       let prompt = tl_string_concat(prompt, user_input);
       let prompt = tl_string_concat(prompt, "</s>\n<|assistant|>\n");
       
       let tokens = tl_tokenizer_encode(tok, prompt);
       
       let cache = tl_kv_cache_new(22);
       let gen_idx = 0;
       
       // Declare next_token with initial value (cloned from tokens to avoid move)
       let next_token = tokens.clone();

       while gen_idx < 100 {
           let len = tl_tensor_len(tokens);
           let start_pos = if gen_idx == 0 { 0 } else { len - 1 };
           let cur_len = if gen_idx == 0 { len } else { 1 };

           let input_tokens = if gen_idx == 0 { 
               tokens 
           } else { 
               next_token
           };

           let w_embedding = tl_tensor_map_get(weights, "token_embd.weight");
           let x = tl_tensor_embedding(input_tokens, w_embedding);
           
           let layer_idx = 0;
           while layer_idx < 22 {
               let blk = TransformerBlock::from_map(weights, layer_idx);
               x = blk.forward(x, cos, sin, cur_len, cache, layer_idx, start_pos);
               layer_idx = layer_idx + 1;
           }
           
           let norm = RMSNorm::new(weights, "output_norm.weight");
           let output = Linear::from_map(weights, "output.weight");
           
           let x_norm = norm.forward(x);
           let logits = output.forward(x_norm);
           
           let last_logits = tl_tensor_narrow(logits, 0, cur_len - 1, 1);
           
           // Update next_token
           next_token = tl_tensor_argmax(last_logits, 1, false); 
           
           let token_str = tl_tokenizer_decode(tok, next_token);
           print(token_str);

           let tid = tl_tensor_item_i64(next_token);
           if tid == 2 {
               gen_idx = 1000; 
           }
           
           tokens = tl_tensor_cat_i64(tokens, next_token, 0);
           
           gen_idx = gen_idx + 1;
       }
       
       tl_kv_cache_free(cache);
    }
}
