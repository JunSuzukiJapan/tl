
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
    
    Param::set_device(Device::Auto);
    print("Test 2: set_device done");
    
    let model_path = "~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    let tokenizer_path = "~/.llm/models/tokenizer.json";
    print("Test 3: paths set");
    
    print("Loading GGUF...");
    let weights = tl_gguf_load(model_path);
    print("Test 4: GGUF loaded");
    
    print("Loading Tokenizer...");
    let tok = tl_tokenizer_new(tokenizer_path);
    print("Test 5: Tokenizer loaded");
    
    print("Precomputing RoPE...");
    let cos = tl_tensor_rope_new_cos(64, 2048, 10000.0);
    let sin = tl_tensor_rope_new_sin(64, 2048, 10000.0);
    print("Test 6: RoPE done");
    
    print("Loading embedding weights...");
    let w_embedding = tl_tensor_map_get(weights, "token_embd.weight");
    print("Test 7: Embedding loaded");
    
    // Initial tokens
    let tokens = [1, 15043];
    print("Test 8: Tokens created");
    
    print("Test Loop Start");
    for i in 0..1 {
        print("Looping...");
    }
    print("Test Loop End");
    
    print("All tests passed!");
}
