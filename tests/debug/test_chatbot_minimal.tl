
// --- Externs ---

struct Map { _d: i64 }

fn main() {
    print("Test 1: print works");
    
    Param::set_device(Device::Auto);
    print("Test 2: set_device works");
    
    let model_path = "~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    print("Test 3: let string works");
}
