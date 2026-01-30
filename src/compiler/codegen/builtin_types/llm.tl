
struct Tokenizer { handle: i64 }
struct KVCache { handle: i64 }
struct Map { handle: i64 }
struct VarBuilder { handle: i64 }

impl Tokenizer {
    extern fn encode(self, text: String) -> Tensor;
    extern fn decode(self, tokens: Tensor) -> String;
    extern fn token_id(self, token: String) -> i64;
    extern fn vocab_size(self) -> i64;
    extern fn free(self) -> ();
}

impl KVCache {
    extern fn get_k(self, layer: i64) -> Tensor;
    extern fn get_v(self, layer: i64) -> Tensor;
    extern fn update(self, layer: i64, k: Tensor, v: Tensor) -> ();
    extern fn free(self) -> ();
}

impl Map {
    extern fn get(self, key: String) -> Tensor;
    extern fn get_1d(self, key: String) -> Tensor;
    extern fn get_quantized(self, key: String) -> Tensor; 
    extern fn set(self, key: String, val: String) -> ();
    extern fn free(self) -> ();
}

impl VarBuilder {
    extern fn get(self, name: String, shape: Tensor) -> Tensor; // ScalarArray -> Tensor
    extern fn grad(self, name: String) -> Tensor;
}
