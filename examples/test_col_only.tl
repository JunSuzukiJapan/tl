// col_loss のみで backward が正しく動くか確認
fn main() {
    let N = 4;
    let lr = 0.5;
    let mut board = Tensor::randn([N, N], true);
    
    for i in 0..50 {
        let probs = board.softmax(1);
        let col_sums = probs.sum(0);
        let col_loss = (col_sums - 1.0).pow(2).sumall();
        
        if i % 10 == 0 {
            println("Epoch {} - col_loss: {}", i, col_loss.item());
        }
        
        col_loss.backward();
        let g = board.grad();
        board = board - g * lr;
        board = board.detach();
        board.enable_grad();
    }
    
    let probs = board.softmax(1);
    println("col_sums:");
    probs.sum(0).print();
    println("=== Done ===");
}
