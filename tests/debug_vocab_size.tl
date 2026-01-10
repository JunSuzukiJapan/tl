extern fn tl_gguf_load(path: String) -> i64;
extern fn tl_tensor_map_get(map: i64, key: String) -> Tensor<f32, 2>;
extern fn println(t: Tensor<f32, 2>) -> void;

fn main() {
    let weights = tl_gguf_load("~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf");
    let emb = tl_tensor_map_get(weights, "output.weight");
    println(emb);
}
