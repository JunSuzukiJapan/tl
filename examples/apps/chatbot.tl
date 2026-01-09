
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
extern fn tl_tensor_embedding(indices: Tensor<i64, 2>, weights: Tensor<f32, 2>) -> Tensor<f32, 3>;
extern fn tl_tensor_add(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
extern fn tl_tensor_reshape_3d_to_2d(t: Tensor<f32, 3>, dims: Tensor<i64, 1>, n: i64) -> Tensor<f32, 2>;

// RoPE/GQA support
extern fn tl_tensor_narrow(t: Tensor<f32, 2>, dim: i64, start: i64, length: i64) -> Tensor<f32, 2>;
extern fn tl_tensor_repeat_interleave(t: Tensor<f32, 4>, repeats: i64, dim: i64) -> Tensor<f32, 4>;
extern fn tl_tensor_argmax(t: Tensor<f32, 2>, dim: i64, keep_dim: bool) -> Tensor<i64, 1>;

extern fn tl_tensor_new_causal_mask(dim: i64) -> Tensor<f32, 2>;

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
        // x: [seq, dim] = [2, 2048]
        // TinyLlama: 32 Q heads, 4 K/V heads (GQA group ratio = 8)
        
        // 1. Project Q, K, V
        let q = self.q_proj.forward(x);  // [seq, 2048]
        let k = self.k_proj.forward(x);  // [seq, 256]
        let v = self.v_proj.forward(x);  // [seq, 256]
        
        // 2. Reshape to multi-head format
        // Q: [seq, 2048] -> [1, seq, 32, 64]
        let q_shape = [1, 2, 32, 64];
        let q4 = tl_tensor_reshape_dims(q, q_shape, 4);
        
        // K/V: [seq, 256] -> [1, seq, 4, 64]
        let kv_shape = [1, 2, 4, 64];
        let k4 = tl_tensor_reshape_dims(k, kv_shape, 4);
        let v4 = tl_tensor_reshape_dims(v, kv_shape, 4);
        
        // 3. Slice RoPE cos/sin for current sequence length
        // cos/sin: [max_len, head_dim/2] = [2048, 32]
        // We need [seq, head_dim/2] = [2, 32]
        let cos_slice = tl_tensor_narrow(cos, 0, 0, 2);
        let sin_slice = tl_tensor_narrow(sin, 0, 0, 2);
        
        // 4. Apply RoPE to Q and K (apply_rope handles broadcasting internally)
        let q_rot = tl_tensor_apply_rope(q4, cos_slice, sin_slice);
        let k_rot = tl_tensor_apply_rope(k4, cos_slice, sin_slice);
        
        // 6. GQA: Repeat K/V heads to match Q heads (4 -> 32, repeat 8x)
        // k_rot: [1, seq, 4, 64] -> [1, seq, 32, 64]
        let k_exp = tl_tensor_repeat_interleave(k_rot, 8, 2);
        let v_exp = tl_tensor_repeat_interleave(v4, 8, 2);
        
        // 7. Transpose: [B, S, H, D] -> [B, H, S, D]
        let q_trans = tl_tensor_transpose(q_rot, 1, 2);
        let k_trans = tl_tensor_transpose(k_exp, 1, 2);
        let v_trans = tl_tensor_transpose(v_exp, 1, 2);
        

        
        // 8. Compute attention scores: Q @ K.T / sqrt(d)
        let k_t = tl_tensor_transpose(k_trans, 2, 3);
        let scores = tl_tensor_matmul_4d(q_trans, k_t);
        
        // Add causal mask
        // mask: [seq, seq] = [2, 2]
        let mask = tl_tensor_new_causal_mask(2); 
        
        // Reshape mask to [1, 1, 2, 2]
        let mask_shape = [1, 1, 2, 2];
        let mask_4d = tl_tensor_reshape_dims(mask, mask_shape, 4);
        
        // Add mask to scores
        let scores = tl_tensor_add_4d(scores, mask_4d);
        
        // Scale by 1/sqrt(head_dim) = 1/sqrt(64) = 0.125
        // Skip for now due to tl_tensor_scale segfault
        // let scale = [0.125];
        // let scores_scaled = tl_tensor_scale(scores, scale);
        
        // 9. Softmax along last dimension
        let attn = tl_tensor_softmax(scores, 3);
        
        // 10. Apply attention: Attn @ V
        let context = tl_tensor_matmul_4d(attn, v_trans);
        
        // 11. Transpose back: [B, H, S, D] -> [B, S, H, D]
        let ctx_trans = tl_tensor_transpose(context, 1, 2);
        
        // 12. Flatten: [1, 2, 32, 64] -> [2, 2048]
        let out_shape = [2, 2048];
        let flat = tl_tensor_reshape_2d(ctx_trans, out_shape, 2);
        
        // 13. Output projection
        let out = self.o_proj.forward(flat);
        return out;
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
        // Build weight name prefix: "blk.X"
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
    
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>) -> Tensor<f32, 2> {
        // Pre-norm attention
        let h = self.attn_norm.forward(x);
        let attn_out = self.attn.forward(h, cos, sin);
        let x2 = tl_tensor_add(x, attn_out);
        
        // Pre-norm FFN
        let h2 = self.ffn_norm.forward(x2);
        let ffn_out = self.mlp.forward(h2);
        let out = tl_tensor_add(x2, ffn_out);
        
        return out;
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
    
    // 3. Load embedding weights and lookup
    print("Loading embedding weights...");
    let embed_w = tl_tensor_map_get(weights, "token_embd.weight");
    print("Embedding lookup...");
    let x_3d = tl_tensor_embedding(tokens, embed_w);
    print("Embedded 3D:");
    print(x_3d);
    
    // Reshape [1, seq, dim] -> [seq, dim]
    let new_shape = [2, 2048];  // Use literal Tensor directly
    let x = tl_tensor_reshape_3d_to_2d(x_3d, new_shape, 2);
    print("Reshaped to 2D:");
    print(x);
    
    // 4. Precompute RoPE
    print("Precomputing RoPE...");
    let cos = tl_tensor_rope_new_cos(64, 2048, 10000.0);
    let sin = tl_tensor_rope_new_sin(64, 2048, 10000.0);
    
    // 5. Forward through all 22 layers
    print("Starting forward pass through 22 layers...");
    
    // We use a loop 0..22
    // Note: 'x' is updated in each iteration.
    // In tl, variable reassignment with let shadowing inside loop might be tricky if not careful with scope.
    // But we can mutate a variable if it was mutable? No, tl doesn't have 'mut' keyword yet in typical way,
    // but we can reassign or shadow.
    // Actually, simply shadowing 'x' inside loop won't update the outer 'x' for next iteration easily
    // unless we use a mutable variable pattern or manual recursion.
    // BUT, 'tl' loop implementation:
    // for i in 0..22 { ... }
    
    // Workaround for lack of mutable outer variable update in loop (if that's a limitation):
    // Construct a list of blocks? No, too much memory.
    // We need to update 'current_x' through the layers.
    
    // Let's rely on shadowing or see if we can use a pointer swap or tensor copy?
    // Wait, 'x' is a Tensor handle.
    // If I do: `x = block.forward(x, ...)` inside loop, does it update outer x?
    // "tl" language spec: does it support reassignment? 
    // `let x = ...` shadows. `x = ...` (assignment) if implemented.
    
    // Assuming assignment works (stmt::Assign).
    // If not, we might need a manual unrolled loop or recursion.
    // Let's try standard loop with reassignment.
    
    let current_x = x;
    
    for i in 0..22 {
        print("Layer:");
        print(i);
        
        let block = TransformerBlock::from_map(weights, i);
        let next_x = block.forward(current_x, cos, sin);
        
        // Output from forward is new tensor. 
        // We need to update current_x.
        // Assuming assignment is supported for updating local vars.
        current_x = next_x;  
    }
    
    
    print("Forward pass complete.");
    print("Final output shape:");
    print(tl_tensor_get_shape(current_x));
    
    // 7. Final RMSNorm and Output Projection
    print("Final Norm and Output...");
    let norm = RMSNorm::new(weights, "output_norm.weight");
    let output = Linear::from_map(weights, "output.weight"); // This is just a linear layer, not full MLP
    
    // Norm
    let x_norm = norm.forward(current_x);
    
    // Output Projection [seq, dim] -> [seq, vocab]
    let logits = output.forward(x_norm);
    print("Logits shape:");
    print(tl_tensor_get_shape(logits));
    
    // 8. Argmax to get token ID
    // We want the last token's prediction.
    // logits: [seq, vocab]
    // argmax(dim=1) -> [seq] indices
    let pred = tl_tensor_argmax(logits, 1, false);
    print("Predicted tokens:");
    print(pred);
    
    // Decode the last token
    // We need to implement tensor indexing or slice to get the last element?
    // Or just decode the whole sequence of predictions?
    // tokenizer_decode expects 'tok' (handle) and 'ids' (Tensor<i64, 2>)
    // pred is Tensor<i64, 1>. We need to reshape to [1, seq] or similar?
    // tl_tokenizer_decode signature: (tok: i64, ids: Tensor<i64, 2>) -> String
    
    // Reshape [seq] -> [1, seq]
    // Skip reshaping for now as we don't have generic reshape.
    // Just print the prediction tensor.
}






