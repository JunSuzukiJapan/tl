
struct RMSNorm {
    weight: Tensor<f32, 1>,
    eps: f32,
}

struct Linear {
    weight: Tensor<f32, 2>,
}

struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_head: i64,
    n_kv_head: i64,
}

struct TransformerBlock {
    attn_norm: RMSNorm,
    ffn_norm: RMSNorm,
    attn: Attention,
    mlp: MLP,
}


// --- Externs ---

struct Map { _d: i64 }

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
    
    let tokens = [1, 15043];
    print("Test 8: Tokens created");
    
    let seq_len = 2;
    
    print("Starting loop...");
    for i in 0..1 {
        print("Looping...");
    }
    
    print("Loop done");
}
