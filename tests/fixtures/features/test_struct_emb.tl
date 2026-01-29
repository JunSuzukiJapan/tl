// Test embedding gradient with struct

struct Emb {
    weight: Tensor<f32, 2>,
}

impl Emb {
    fn new(vocab: i64, d: i64) -> Emb {
        Emb(Tensor::randn([vocab, d], true) * 0.1)
    }
    
    fn forward(self, idx: Tensor<f32, 2>) -> Tensor<f32, 3> {
        idx.embedding(self.weight)
    }
}

fn main() {
    print("Test: Embedding gradient through struct");
    
    let emb = Emb::new(14, 16);
    let idx = Tensor::randn([2, 4], false) * 0.0 + 1.0;
    
    let out = emb.forward(idx);
    let loss = out.sum();
    
    print("Loss:");
    print(loss);
    
    loss.backward();
    
    let gw = emb.weight.grad();
    print("Emb weight gradient:");
    print(gw);
}
