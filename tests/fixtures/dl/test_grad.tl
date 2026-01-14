// Minimal test to verify gradients work

fn main() {
    print("Test 1: Simple gradient");
    let x = Tensor::randn([2, 3], true);
    let y = x * 2.0;
    let loss = y.sum();
    
    print("Loss:");
    print(loss);
    
    loss.backward();
    
    let gx = x.grad();
    print("Gradient:");
    print(gx);
    
    print("Test 2: Embedding gradient");
    let vocab = 14;
    let d_model = 16;
    let weight = Tensor::randn([vocab, d_model], true) * 0.1;
    let idx = Tensor::randn([2, 4], false) * 0.0 + 1.0;
    
    let emb = idx.embedding(weight);
    let loss2 = emb.sum();
    
    print("Embedding loss:");
    print(loss2);
    
    loss2.backward();
    
    let gw = weight.grad();
    print("Embedding weight gradient:");
    print(gw);
}
