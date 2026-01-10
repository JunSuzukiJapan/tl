
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

fn main() {
    print("Test 1: print works");
    
    set_device("auto");
    print("Test 2: set_device works");
    
    let model_path = "~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    print("Test 3: let string works");
}
