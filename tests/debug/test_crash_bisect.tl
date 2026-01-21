
// --- Externs ---

struct Map { _d: i64 }

// RoPE/GQA support

// --- Structs ---

struct RMSNorm {
    weight: Tensor<f32, 1>,
    eps: f32,
}

impl RMSNorm {
    fn new(map: Map, key: String) -> RMSNorm {
        let w = map.get_1d(key);
        return RMSNorm { weight: w, eps: 0.00001 };
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        return x.rms_norm(self.weight, self.eps);
    }
}

struct Linear {
    weight: Tensor<f32, 2>, 
}

impl Linear {
    fn from_map(map: Map, key: String) -> Linear {
        let w = map.get(key);
        return Linear { weight: w }; 
    }

    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let wt = self.weight.transpose_2d(0, 1);
        let out = x.matmul(wt);
        return out;
    }
}

struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    fn from_map(map: Map, prefix: String) -> MLP {
        // GGUF naming: blk.X.ffn_gate.weight, blk.X.ffn_up.weight, blk.X.ffn_down.weight
        let gate = Linear::from_map(map, prefix.concat("_gate.weight"));
        let up = Linear::from_map(map, prefix.concat("_up.weight"));
        let down = Linear::from_map(map, prefix.concat("_down.weight"));
        return MLP { gate_proj: gate, up_proj: up, down_proj: down };
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let g = self.gate_proj.forward(x.clone());
        let u = self.up_proj.forward(x);
        let silu_g = g.silu();
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
    fn from_map(map: Map, prefix: String) -> Attention {
        // GGUF naming: blk.X.attn_q.weight, blk.X.attn_k.weight, blk.X.attn_v.weight, blk.X.attn_output.weight
        let q = Linear::from_map(map, prefix.concat("_q.weight"));
        let k = Linear::from_map(map, prefix.concat("_k.weight"));
        let v = Linear::from_map(map, prefix.concat("_v.weight"));
        let o = Linear::from_map(map, prefix.concat("_output.weight"));
        
        return Attention { 
            q_proj: q, k_proj: k, v_proj: v, o_proj: o, 
            n_heads: 32, head_dim: 64
        };
    }
    
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, seq_len: i64) -> Tensor<f32, 2> {
        // x: [seq, dim]
        // TinyLlama: 32 Q heads, 4 K/V heads (GQA group ratio = 8)
        
        // 1. Project Q, K, V
        let q = self.q_proj.forward(x.clone());  // [seq, 2048]
        let k = self.k_proj.forward(x.clone());  // [seq, 256]
        let v = self.v_proj.forward(x);          // [seq, 256]
        
        // 2. Reshape to multi-head format
        // Q: [seq, 2048] -> [1, seq, 32, 64]
        let q4 = q.reshape([1, seq_len, 32, 64]);
        
        // K/V: [seq, 256] -> [1, seq, 4, 64]
        let k4 = k.reshape([1, seq_len, 4, 64]);
        let v4 = v.reshape([1, seq_len, 4, 64]);
        
        // 3. Slice RoPE cos/sin for current sequence length
        // cos/sin: [max_len, head_dim/2]
        // We need [seq, head_dim/2]
        let cos_slice = cos.narrow(0, 0, seq_len);
        let sin_slice = sin.narrow(0, 0, seq_len);
        
        // 4. Apply RoPE to Q and K (apply_rope handles broadcasting internally)
        let q_rot = q4.apply_rope(cos_slice.clone(), sin_slice.clone());
        let k_rot = k4.apply_rope(cos_slice, sin_slice);
        
        // 6. GQA: Repeat K/V heads to match Q heads (4 -> 32, repeat 8x)
        // k_rot: [1, seq, 4, 64] -> [1, seq, 32, 64]
        let k_exp = k_rot.repeat_interleave(8, 2);
        let v_exp = v4.repeat_interleave(8, 2);
        
        // 7. Transpose: [B, S, H, D] -> [B, H, S, D]
        let q_trans = q_rot.transpose(1, 2);
        let k_trans = k_exp.transpose(1, 2);
        let v_trans = v_exp.transpose(1, 2);
        
        // 8. Compute attention scores: Q @ K.T / sqrt(d)
        let k_t = k_trans.transpose(2, 3);
        let scores = q_trans.matmul_4d(k_t);
        
        // Add causal mask
        let mask = Tensor::new_causal_mask(seq_len); 
        
        // Reshape mask to [1, 1, seq, seq]
        let mask_4d = mask.reshape([1, 1, seq_len, seq_len]);
        
        // Add mask to scores
        let scores = scores.add_4d(mask_4d);
        
        // Scale by 1/sqrt(head_dim) = 1/sqrt(64) = 0.125
        // Skip for now due to tl_tensor_scale segfault
        // let scale = [0.125];
        // let scores_scaled = scores.scale(scale);
        
        // 9. Softmax along last dimension
        let attn = scores.softmax(3);
        
        // 10. Apply attention: Attn @ V
        let context = attn.matmul_4d(v_trans);
        
        // 11. Transpose back: [B, H, S, D] -> [B, S, H, D]
        let ctx_trans = context.transpose(1, 2);
        
        // 12. Flatten: [1, seq, 32, 64] -> [seq, 2048]
        let flat = ctx_trans.reshape([seq_len, 2048]);
        
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
    fn from_map(map: Map, layer_idx: i64) -> TransformerBlock {
        print("Block::from_map start");
        // Build weight name prefix: "blk.X"
        let prefix = "blk.".concat(String::from_int(layer_idx));
        print("Block::from_map prefix built");
        
        let attn_norm = RMSNorm::new(map, prefix.concat(".attn_norm.weight"));
        print("Block::from_map attn_norm done");
        
        let ffn_norm = RMSNorm::new(map, prefix.concat(".ffn_norm.weight"));
        print("Block::from_map ffn_norm done");
        
        let attn = Attention::from_map(map, prefix.concat(".attn"));
        print("Block::from_map attn done");
        
        let mlp = MLP::from_map(map, prefix.concat(".ffn"));
        print("Block::from_map mlp done");
        
        return TransformerBlock { 
            attn_norm: attn_norm, 
            ffn_norm: ffn_norm, 
            attn: attn, 
            mlp: mlp 
        };
    }
    
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, seq_len: i64) -> Tensor<f32, 2> {
        // Pre-norm attention
        let h = self.attn_norm.forward(x.clone());
        let attn_out = self.attn.forward(h, cos, sin, seq_len);
        let x2 = x.add(attn_out);
        
        // Pre-norm FFN
        let h2 = self.ffn_norm.forward(x2.clone());
        let ffn_out = self.mlp.forward(h2);
        let out = x2.add(ffn_out);
        
        return out;
    }
}

fn main() {
    print("Test 1: start");
    
    Param::set_device(Device::Auto);
    print("Test 2: set_device done");
    
    let model_path = "~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    print("Loading GGUF...");
    
    let weights = Map::load(model_path);
    print("Test 4: GGUF loaded");
    
    print("Precomputing RoPE...");
    let cos = Tensor::rope_new_cos(64, 2048, 10000.0);
    let sin = Tensor::rope_new_sin(64, 2048, 10000.0);
    print("Test 6: RoPE done");
    
    print("Loading embedding weights...");
    let w_embedding = weights.get("token_embd.weight");
    print("Test 7: Embedding loaded");
    
    // Tokens: [1, 15043]
    let tokens = [1, 15043];
    print("Test 8: Tokens created");
    
    let seq_len = 2; // i64
    
    print("Starting loop...");
    
    // Just 1 step
    print("Main start");
    let x = tokens.embedding(w_embedding);
    print("Embedding done");
    
    let current_x = x;
    print("x assigned");
    
    print("Block creation...");
    let blk = TransformerBlock::from_map(weights, 0);
    print("Block created");
    
    print("Forwarding...");
    current_x = blk.forward(current_x, cos, sin, seq_len);
    print("Forward done");
    
    print("Main done");
}
