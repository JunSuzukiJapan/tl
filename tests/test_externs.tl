// Test with extern declarations like chatbot.tl

extern fn tl_http_download(url: String, path: String) -> bool;
extern fn tl_path_exists(path: String) -> bool;
extern fn tl_file_read_string(path: String) -> String;
extern fn tl_string_from_int(val: i64) -> String;
extern fn tl_string_concat(s1: String, s2: String) -> String;

extern fn tl_tokenizer_new(path: String) -> i64;
extern fn tl_tokenizer_encode(tok: i64, prompt: String) -> Tensor<i64, 2>;
extern fn tl_tokenizer_decode(tok: i64, ids: Tensor<i64, 2>) -> String;

struct Map { _d: i64 }
extern fn tl_gguf_load(path: String) -> Map;
extern fn tl_tensor_map_get(map: Map, name: String) -> Tensor<f32, 2>;
extern fn tl_tensor_map_get_1d(map: Map, name: String) -> Tensor<f32, 1>;

fn main() {
    print("Testing externs...");
    print("Done!");
}
