// 対角線制約のみで backward テスト
fn main() {
    let N = 4;
    let K = 2 * N - 1;
    
    let anti_diag_mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
        if r + c == k { 1.0 } else { 0.0 }
    } ];
    
    println("anti_diag_mask shape:");
    anti_diag_mask.print();
    
    let lr = 0.5;
    let mut board = Tensor::randn([N, N], true);
    
    for i in 0..20 {
        let probs = board.softmax(1);
        let probs_b = probs.reshape([1, N, N]);
        let product = probs_b * anti_diag_mask;
        
        // product の shape を確認（初回のみ）
        if i == 0 {
            println("product shape check: should be [7,4,4]");
            product.print();
        }
        
        let anti_diag_sums = product.sum(2).sum(1);
        
        if i == 0 {
            println("anti_diag_sums (should be [7]):");
            anti_diag_sums.print();
        }
        
        // reluなしでテスト
        let anti_diag_loss = (anti_diag_sums - 1.0).pow(2).sumall();
        
        if i % 5 == 0 {
            println("Epoch {} - anti_diag_loss: {}", i, anti_diag_loss.item());
        }
        
        println("Calling backward...");
        anti_diag_loss.backward();
        println("backward done.");
        
        let g = board.grad();
        println("grad obtained.");
        
        let scaled_g = g * lr;
        // println("scaled_g done.");
        // println("g shape:"); g.print_shape(); // Assuming print_shape exists or use shape()
        println("g shape:");
        // tl_tensor_shape is not directly exposed as method usually, but print() works.
        // g.print(); // Might be too big? 4x4 is small.
        g.print();
        println("scaled_g shape:");
        scaled_g.print();
        
        board = board - scaled_g; 
        println("update done.");
        
        board = board.detach();
        println("detach done.");
        
        board.enable_grad();
        println("enable_grad done.");
    }
    
    let probs = board.softmax(1);
    println("Final probs:");
    probs.print();
    println("=== Done ===");
}
