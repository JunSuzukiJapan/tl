// 最小再現: forループ2回の autograd
fn main() {
    let N = 4;
    let lr = 0.5;
    
    let mut board = Tensor::randn([N, N], true);
    
    for i in 0..3 {
        println("--- Epoch {} start ---", i);
        
        let probs = board.softmax(1);
        println("  softmax done");
        
        let loss = probs.sum();
        println("  sum done");
        
        loss.backward();
        println("  backward done");
        
        let g = board.grad();
        println("  grad done");
        
        board = board - g * lr;
        println("  update done");
        
        board = board.detach();
        println("  detach done");
        
        board.enable_grad();
        println("  enable_grad done");
        
        println("--- Epoch {} complete ---", i);
    }
    println("=== All epochs done ===");
}
