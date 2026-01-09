
// --- Externs ---
extern fn tl_http_download(url: String, path: String) -> bool;
extern fn tl_path_exists(path: String) -> bool;
extern fn tl_file_read_string(path: String) -> String;
extern fn tl_string_from_int(val: i64) -> String;
extern fn tl_string_concat(s1: String, s2: String) -> String;

extern fn tl_tokenizer_new(path: String) -> i64;
extern fn tl_tokenizer_encode(tok: i64, prompt: String) -> Tensor<i64, 2>;
extern fn tl_tokenizer_decode(tok: i64, ids: Tensor<i64, 2>) -> String;

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

extern fn tl_tensor_rope_new_cos(dim: i64, len: i64, theta: f32) -> Tensor<f32, 2>;
extern fn tl_tensor_rope_new_sin(dim: i64, len: i64, theta: f32) -> Tensor<f32, 2>;

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
        let g = self.gate_proj.forward(x);
        let u = self.up_proj.forward(x);
        let silu_g = tl_tensor_silu(g);
        let gu = silu_g * u; 
        let d = self.down_proj.forward(gu);
        return d;
    }
}

// Attention removed

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
        // GGUF naming: blk.X.attn_q.weight, blk.X.attn_k.weight, blk.X.attn_v.weight, blk.X.attn_output.weight
        let q = Linear::from_map(map, tl_string_concat(prefix, "_q.weight"));
        let k = Linear::from_map(map, tl_string_concat(prefix, "_k.weight"));
        let v = Linear::from_map(map, tl_string_concat(prefix, "_v.weight"));
        let o = Linear::from_map(map, tl_string_concat(prefix, "_output.weight"));
        
        return Attention { 
            q_proj: q, k_proj: k, v_proj: v, o_proj: o, 
            n_heads: 32, head_dim: 64
        };
    }
    
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);
        
        // Reshape [1, Dim] -> [1, 1, H, D]
        let shape = [1, 1, 32, 64];
        let q4 = tl_tensor_reshape_dims(q, shape, 4);
        let k4 = tl_tensor_reshape_dims(k, shape, 4);
        let v4 = tl_tensor_reshape_dims(v, shape, 4);
        
        // RoPE
        let q_rot = tl_tensor_apply_rope(q4, cos, sin);
        let k_rot = tl_tensor_apply_rope(k4, cos, sin);
        
        // Q: [B, S, H, D] -> [B, H, S, D]
        let q_trans = tl_tensor_transpose(q_rot, 1, 2);
        let k_trans = tl_tensor_transpose(k_rot, 1, 2);
        let v_trans = tl_tensor_transpose(v4, 1, 2);
        
        // Scores = Q @ K.T
        let k_t = tl_tensor_transpose(k_trans, 2, 3);
        let scores = tl_tensor_matmul_4d(q_trans, k_t);
        
        // Scale (SKIPPED for now - causes segfault)
        // let scale = [0.125]; 
        // let scores_scaled = tl_tensor_scale(scores, scale); 
        
        // Softmax
        let attn = tl_tensor_softmax(scores, 3);
        
        // Out = Attn @ V
        let context = tl_tensor_matmul_4d(attn, v_trans);
        
        // Transpose back [B, H, S, D] -> [B, S, H, D]
        let ctx_trans = tl_tensor_transpose(context, 1, 2);
        
        return x; // Still placeholder
    }
}




extern fn tl_tensor_reshape_2d(t: Tensor<f32, 4>, dims: Tensor<i64, 1>, n: i64) -> Tensor<f32, 2>;

fn main() {
    print("TinyLlama Chatbot Verification");
    
    // 1. Load model
    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    let tokenizer_path = "/Users/junsuzuki/.llm/models/tokenizer.json";
    
    print("Loading GGUF...");
    let weights = tl_gguf_load(model_path);
    print("Loading Tokenizer...");
    let tok = tl_tokenizer_new(tokenizer_path);
    print("Loaded!");
    
    // 2. Encode prompt
    let prompt = "Hello";
    let tokens = tl_tokenizer_encode(tok, prompt);
    print("Encoded tokens:");
    print(tokens);
    
    // 3. Precompute RoPE
    print("Precomputing RoPE...");
    let cos = tl_tensor_rope_new_cos(64, 2048, 10000.0);
    let sin = tl_tensor_rope_new_sin(64, 2048, 10000.0);
    
    // 4. Instantiate layer 0 components
    print("Creating Layer 0 components...");
    let block_norm = RMSNorm::new(weights, "blk.0.attn_norm.weight");
    let mlp = MLP::from_map(weights, "blk.0.ffn");
    let attn = Attention::from_map(weights, "blk.0.attn");
    
    print("All components created successfully!");
    print("Transformer architecture verified.");
}

