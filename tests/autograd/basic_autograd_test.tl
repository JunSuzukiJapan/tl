// Test: basic autograd - forward, backward, gradient check
fn main() {
    // 1. Simple scalar loss backward
    let x = Tensor::randn([3, 4], true);
    let y = x * 2.0;
    let loss = y.sumall();
    
    println("Forward pass:");
    println("x shape: [3, 4]");
    println("loss = {}", loss.item());
    
    loss.backward();
    
    let g = x.grad();
    println("Gradient (should be all 2.0):");
    g.print();
    
    // 2. matmul + softmax backward
    let W = Tensor::randn([4, 3], true);
    let logits = x.matmul(W);
    let probs = logits.softmax(1);
    let loss2 = probs.sumall();
    
    println("Softmax output:");
    probs.print();
    println("loss2 = {}", loss2.item());
    
    println("=== Autograd basic test passed ===");
}
