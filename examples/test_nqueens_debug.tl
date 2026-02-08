// N-Queens gradient 検証テスト
fn main() {
    let N = 4;
    let lr = 0.5;
    let mut board = Tensor::randn([N, N], true);

    for i in 0..5 {
        let probs = board.softmax(1);
        let col_sums = probs.sum(0);
        let col_loss = (col_sums - 1.0).pow(2).sumall();
        
        let loss_val = col_loss.item();
        println("Epoch {} - loss: {}", i, loss_val);
        
        col_loss.backward();
        let g = board.grad();
        
        board = board - g * lr;
        board = board.detach();
        board.enable_grad();
    }
    
    // 最終 board の softmax を確認
    let final_probs = board.softmax(1);
    println("Final probs:");
    final_probs.print();
    
    // queens カウント
    let mut total = 0;
    let mut r = 0;
    while r < N {
        let mut c = 0;
        while c < N {
            if final_probs[r, c] > 0.5 {
                total = total + 1;
            }
            c = c + 1;
        }
        r = r + 1;
    }
    println("Queens found: {}", total);
    println("=== Done ===");
}
