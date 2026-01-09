
struct Map {}

extern fn tl_http_download(url: String, path: String) -> bool;

struct Tokenizer {}
extern fn tl_path_exists(path: String) -> bool;
extern fn tl_file_read_string(path: String) -> String;

extern fn tl_tokenizer_new(path: String) -> Tokenizer;
extern fn tl_tokenizer_encode(tok: Tokenizer, prompt: String) -> Tensor<f32, 2>;
extern fn tl_tokenizer_decode(tok: Tokenizer, ids: Tensor<f32, 2>) -> String;

extern fn tl_gguf_load(path: String) -> Map;
extern fn tl_tensor_map_get(map: Map, name: String) -> Tensor<f32, 2>;

// Skip cat for now
extern fn tl_tensor_silu(t: Tensor<f32, 3>) -> Tensor<f32, 3>;
extern fn tl_tensor_apply_rope(x: Tensor<f32, 3>, cos: Tensor<f32, 3>, sin: Tensor<f32, 3>) -> Tensor<f32, 3>;

struct RMSNorm {
    weight: Tensor<f32, 1>,
    eps: f32,
}

impl RMSNorm {
    fn new(w: Tensor<f32, 1>) -> RMSNorm {
        return RMSNorm(w, 0.00001);
    }
}


fn main() {
    print("TinyLlama Chatbot");
    
    // 1. Download
    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    let tokenizer_path = "/Users/junsuzuki/.llm/models/tokenizer.json";
    
    if !tl_path_exists(model_path) {
        print("Downloading Model...");
        let ok = tl_http_download("https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true", model_path);
        if !ok { print("Download failed!"); return (); }
    }
    
    if !tl_path_exists(tokenizer_path) {
        print("Downloading Tokenizer...");
        let ok = tl_http_download("https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/tokenizer.json", tokenizer_path);
        if !ok { print("Download failed!"); return (); }
    }
    
    // 2. Load
    print("Loading...");
    let weights = tl_gguf_load(model_path); 
    let tok = tl_tokenizer_new(tokenizer_path);
    print("Loaded.");
    
    // 3. Chat Loop
    let prompt = "Hello!";
    print("User: ");
    print(prompt);
    
    // DEBUG: Bypass tokenizer panic
    let tokens = [[1, 2, 3, 4, 5]]; // Dummy tokens (I64)
    // tokens: [1, Seq]
    // tokens: [1, Seq]
    
    print("Tokens:");
    print(tokens);
    
    // 4. Inference (Placeholder)
    // Run simple embedding lookup to prove weights loaded
    let emb_w = tl_tensor_map_get(weights, "token_embd.weight");
    // emb_w [Vocab, Dim]
    
    let x = embedding(tokens, emb_w);
    print("Embedding output shape:");
    print(x);
    
    // Decode back
    let out_str = tl_tokenizer_decode(tok, tokens);
    print("Decoded:");
    print(out_str);
}
