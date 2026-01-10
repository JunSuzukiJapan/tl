extern fn tl_gguf_load(path: String) -> i64;
extern fn tl_tensor_map_keys(map: i64) -> i64;
extern fn tl_vec_string_len(v: i64) -> i64;
extern fn tl_vec_string_get(v: i64, i: i64) -> String;

fn main() {
    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    let weights = tl_gguf_load(model_path);
    // Note: tl_tensor_map_keys is not an extern yet. 
    // I should check builtins.rs for what I have.
}
