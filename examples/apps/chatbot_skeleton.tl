// TinyLlama Chatbot in TensorLogic (tl)

// --- Externs ---
// These function signatures must match builtins.rs

// Checkpointing / File IO (from builtins)
extern fn tl_http_download(url: string, path: string) -> bool;
extern fn tl_path_exists(path: string) -> bool;
extern fn tl_file_read_string(path: string) -> string;

// LLM Builtins (from llm.rs)
// Tokenizer
extern fn tl_tokenizer_new(path: string) -> *mut void;
extern fn tl_tokenizer_encode(tok: *mut void, prompt: string) -> Tensor<f32, 2>; // Returns [1, Seq]
extern fn tl_tokenizer_decode(tok: *mut void, ids: Tensor<f32, 2>) -> string;

// GGUF
extern fn tl_gguf_load(path: string) -> *mut void; // Returns *mut Map
extern fn tl_tensor_map_get(map: *mut void, name: string) -> Tensor<f32, 2>; // Generic get, returns tensor

// Tensor Ops
extern fn tl_tensor_cat(tensors: *mut void, dim: i64) -> Tensor<f32, 3>; // Generic Tensor return
extern fn tl_tensor_silu(t: Tensor<f32, 3>) -> Tensor<f32, 3>; // Generic
extern fn tl_tensor_apply_rope(x: Tensor<f32, 3>, cos: Tensor<f32, 3>, sin: Tensor<f32, 3>) -> Tensor<f32, 3>;

// Stdlib Tensor Ops (already available as builtins usually, but explicit externs help if compiler checks)
// matmul, softmax, embedding, etc are intrinsics.

// --- Constants ---
// TinyLlama-1.1B Config
// vocab_size = 32000
// hidden_size = 2048
// intermediate_size = 5632
// num_hidden_layers = 22
// num_attention_heads = 32
// num_key_value_heads = 4
// head_dim = 64
// max_position_embeddings = 2048

struct Config {
    vocab_size: i64,
    dim: i64,
    hidden_dim: i64,
    n_layers: i64,
    n_heads: i64,
    n_kv_heads: i64,
    head_dim: i64,
    norm_eps: f32,
}

struct RMSNorm {
    weight: Tensor<f32, 1>,
    eps: f32,
}

impl RMSNorm {
    fn new(w: Tensor<f32, 1>, eps: f32) -> RMSNorm {
        return RMSNorm(w, eps);
    }
    
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        // x: [B, S, D]
        // norm = x * rsqrt(x.pow(2).mean(-1, keepdim=true) + eps)
        // We lack mean/rsqrt/pow intrinsics easy access?
        // Let's rely on `candle_nn::ops::rms_norm` if exposed? No.
        
        // Manual implementation:
        // x_sq = x * x
        // var = x_sq.sum(-1, keep=true) / dim
        // inv_std = 1.0 / sqrt(var + eps)
        // return x * inv_std * weight
        
        let x2 = x * x;
        let dim_size = 2048.0; // Hardcoded for optimization or pass in?
        let sum_sq = sum(x2, 2, true); // keep dims? check builtin signature
        // `tl_tensor_sum_dim(t, dim, keep)`
        
        let var = sum_sq / dim_size;
        let inv_std = rsqrt(var + self.eps); // Need rsqrt builtin or 1/sqrt
        let norm = x * inv_std;
        
        // Broadcast weight: [D] -> [1, 1, D]
        // Currently `tl` handles broadcasting in mul if supported.
        return norm * self.weight;
    }
}

// NOTE: We need rsqrt. `tl` has `sqrt`. `1 / sqrt(x)` works.

struct Linear {
    W: Tensor<f32, 2>, 
    // Bias is optional in Llama, usually None. TinyLlama has no bias in Linear except maybe head?
    // TinyLlama: No bias in QKV, MLP, Output.
}

impl Linear {
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        return matmul(x, self.W);
    }
}

struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        let gate = self.gate_proj.forward(x);
        let gate = silu(gate); // Use extern `tl_tensor_silu` wrapped
        let up = self.up_proj.forward(x);
        let mid = gate * up;
        let out = self.down_proj.forward(mid);
        return out;
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: i64,
    n_kv_heads: i64,
    head_dim: i64,
}

impl Attention {
    fn forward(self, x: Tensor<f32, 3>, cos: Tensor<f32, 3>, sin: Tensor<f32, 3>) -> Tensor<f32, 3> {
        let q = self.q_proj.forward(x); // [B, S, D]
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);
        
        // Reshape for heads?
        // [B, S, H * D_head] -> [B, S, H, D_head]
        // `tl` lacks generic reshape easily.
        // BUT `apply_rope` (our builtin) handles the rotation assuming last dim is head_dim?
        // Let's assume `tl_tensor_apply_rope` handles the [B, S, D] -> rotate heads logic internally if we pass it right.
        // Actually, our `apply_rope` impl assumes [..., D] and rotates last dim.
        // So we should reshape Q, K to [B, S, H, D_head].
        
        // Since we don't have easy reshape in pure `tl` without explicit shapes,
        // we might keep it flat if RoPE handles it.
        // Standard RoPE rotates pairs.
        
        // Let's call `rope(q, cos, sin)`.
        let q_rot = rope(q, cos, sin); 
        let k_rot = rope(k, cos, sin);
        
        // GQA (Grouped Query Attention) needed for TinyLlama?
        // TinyLlama 1.1B: n_heads=32, n_kv_heads=4. Yes GQA.
        // We need to repeat K, V to match Q heads.
        // `repeat_kv` logic.
        
        // If we can't do GQA easily in `tl`, this chatbot will answer garbage.
        // We need `tl_tensor_repeat_interleave` or similar.
        
        // Backup: Use a simpler model? No, heavily requested "ChatGPT-like".
        // Let's assume we can implement `repeat` via broadcasting or loop?
        // Loop is slow.
        // Let's Add `tl_tensor_repeat_kv` to `llm.rs` later if needed.
        // For now, let's write code assuming we have it.
        
        // Attention Score: Q @ K.T / sqrt(D)
        // Softmax
        // Att @ V
        
        let out = matmul(q_rot, k_rot); // Placeholder
        let out = self.o_proj.forward(out);
        return out;
    }
}

// ... Main loop ...
fn main() {
    print("Initialize...");
    let model_dir = "/Users/junsuzuki/.llm/models";
    let model_path = model_dir + "/tinyllama-1.1b-chat-q4_0.gguf";
    let tokenizer_path = model_dir + "/tokenizer.json";
    
    // Check Download
    if !tl_path_exists(model_path) {
        print("Downloading Model...");
        tl_http_download("https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", model_path);
    }
    if !tl_path_exists(tokenizer_path) {
         print("Downloading Tokenizer...");
         tl_http_download("https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/tokenizer.json", tokenizer_path);
    }
    
    // Load
    print("Loading Model Weights...");
    let weights = tl_gguf_load(model_path);
    print("Loading Tokenizer...");
    let tok = tl_tokenizer_new(tokenizer_path);
    
    // Chat Loop
    print("Ready! (Type 'quit' to exit)");
    
    loop {
        // Read input (Need generic input? `tl` defaults to hardcoded input, let's pretend `tl_input`)
        // let text = input("> ");
        let text = "Hello"; // Hardcoded for test
        
        let tokens = tl_tokenizer_encode(tok, text);
        
        // Forward pass (Placeholder)
        // ...
        
        // Decode
        let output = tl_tokenizer_decode(tok, tokens);
        print(output);
        
        // Break for test
        break;
    }
}
