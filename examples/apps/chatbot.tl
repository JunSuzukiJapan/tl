
extern fn tl_http_download(url: String, path: String) -> bool;
extern fn tl_path_exists(path: String) -> bool;
extern fn tl_file_read_string(path: String) -> String;

extern fn tl_tokenizer_new(path: String) -> i64;
extern fn tl_tokenizer_encode(tok: i64, prompt: String) -> Tensor<i64, 2>;
extern fn tl_tokenizer_decode(tok: i64, ids: Tensor<i64, 2>) -> String;

extern fn tl_gguf_load(path: String) -> i64;
extern fn tl_tensor_map_get(map: i64, name: String) -> Tensor<f32, 2>;

fn main() {
    print("TinyLlama Chatbot Verification");

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
    
    // 3. Encode
    let prompt = "Hello!";
    print("User: ");
    print(prompt);
    
    let tokens = tl_tokenizer_encode(tok, prompt);
    
    print("Tokens:");
    print(tokens);

    // 4. Prove Map Access
    // 4. Prove Map Access
    // Commented out to verify Tokenizer first
}
