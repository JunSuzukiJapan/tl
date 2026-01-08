fn main() {
    print("Creating dummy data...");
    let bx = Tensor::randn([10, 784], true);
    
    print("Creating weights...");
    let w = Tensor::randn([784, 10], true);
    
    print("Forward...");
    let logits = matmul(bx, w);
    
    print("Sum Loss...");
    let loss = logits.sum();
    print("Loss:"); print(loss.item());
    
    print("Backward...");
    loss.backward();
    
    let g = w.grad();
    print("Grad:");
    print(g.abs().sum().item());
    
    if g.abs().sum().item() == 0.0 {
        print("GRADIENT MISSING!");
    } else {
        print("Gradient OK");
    }
}
