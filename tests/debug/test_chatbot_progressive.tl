
// --- Externs ---

struct Map { _d: i64 }

fn main() {
    print("Test 1: start");
    
    Param::set_device(Device::Auto);
    print("Test 2: set_device done");
    
    let model_path = "~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    let tokenizer_path = "~/.llm/models/tokenizer.json";
    print("Test 3: paths set");
    
    print("Loading GGUF...");
    let weights = Map::load(model_path);
    print("Test 4: GGUF loaded");
    
    print("Loading Tokenizer...");
    let tok = Tokenizer::new(tokenizer_path);
    print("Test 5: Tokenizer loaded");
    
    print("Precomputing RoPE...");
    let cos = Tensor::rope_new_cos(64, 2048, 10000.0);
    let sin = Tensor::rope_new_sin(64, 2048, 10000.0);
    print("Test 6: RoPE done");
    
    print("Loading embedding weights...");
    let w_embedding = weights.get("token_embd.weight");
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
