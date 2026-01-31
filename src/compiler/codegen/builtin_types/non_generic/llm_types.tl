
// LLM Builtin Types

struct Tokenizer {
    handle: i64,
}

impl Tokenizer {
    extern fn new(path: String) -> Tokenizer;
    extern fn encode(self, prompt: String) -> Tensor<i64, 1>;
    extern fn decode(self, ids: Tensor<i64, 1>) -> String;
}

struct KVCache {
    handle: i64,
}

impl KVCache {
    extern fn new(layers: i64) -> KVCache;
    extern fn free(self);
    extern fn get_k(self, layer: i64) -> Tensor<f32, 2>;
    extern fn get_v(self, layer: i64) -> Tensor<f32, 2>;
    extern fn update(self, layer: i64, k: Tensor<f32, 2>, v: Tensor<f32, 2>);
}
