struct GPT {
    wte: Tensor<f32, 2>,
}
impl GPT {
    fn new(v: i64) -> GPT {
        return GPT(randn([v, v], true));
    }
}
fn main() {}
