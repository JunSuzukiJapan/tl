struct Embedding { w: Tensor<f32, 2> }
impl Embedding { fn new() -> Embedding { return Embedding(randn([10, 10], true)); } }

struct GPT {
    wte: Embedding,
    wpe: Embedding,
}
impl GPT {
    fn new() -> GPT {
        return GPT(
            Embedding::new(),
            Embedding::new()
        );
    }
}
fn main() {}
