
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

extern fn tl_tensor_rope_new_cos(dim: i64, len: i64, theta: f32) -> Tensor<f32, 2>;
extern fn tl_tensor_rope_new_sin(dim: i64, len: i64, theta: f32) -> Tensor<f32, 2>;

fn main() {
    print("Test 1: start");
    
    set_device("auto");
    print("Test 2: set_device done");
    
    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    print("Loading GGUF...");
    let weights = tl_gguf_load(model_path);
    print("Test 4: GGUF loaded");
    
    print("Precomputing RoPE...");
    let cos = tl_tensor_rope_new_cos(64, 2048, 10000.0);
    let sin = tl_tensor_rope_new_sin(64, 2048, 10000.0);
    print("Test 6: RoPE done");
    
    print("Loading embedding weights...");
    let w_embedding = tl_tensor_map_get(weights, "token_embd.weight");
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
