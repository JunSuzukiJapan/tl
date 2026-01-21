// Add a few externs just in case

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

fn main() {
    print("Structs defined. Running main.");
}
