// 制約損失と勾配の検証テスト
fn main() {
    let N = 4;
    let K = 2 * N - 1;
    
    // マスク
    let anti_diag_mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
        if r + c == k { 1.0 } else { 0.0 }
    } ];
    let main_diag_mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
        if r - c + N - 1 == k { 1.0 } else { 0.0 }
    } ];

    let lr = 0.5;
    let mut board = Tensor::randn([N, N], true);
    
    for i in 0..200 {
        let probs = board.softmax(1);
        
        // 列制約
        let col_sums = probs.sum(0);
        let col_loss = (col_sums - 1.0).pow(2).sumall();
        
        // 対角線制約
        let probs_b = probs.reshape([1, N, N]);
        let anti_diag_sums = (probs_b * anti_diag_mask).sum(2).sum(1);
        let main_diag_sums = (probs_b * main_diag_mask).sum(2).sum(1);
        let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sumall();
        let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sumall();
        
        let reg_loss = (probs * (1.0 - probs)).sumall() * 0.05;
        
        let constraint_loss = col_loss + anti_diag_loss + main_diag_loss;
        let total_loss = constraint_loss + reg_loss;
        
        if i % 20 == 0 {
            println("Epoch {} - total: {} col: {} anti_diag: {} main_diag: {} reg: {}", 
                i, total_loss.item(), col_loss.item(), anti_diag_loss.item(), main_diag_loss.item(), reg_loss.item());
        }
        
        total_loss.backward();
        let g = board.grad();
        board = board - g * lr;
        board = board.detach();
        board.enable_grad();
    }
    
    let probs = board.softmax(1);
    println("Final probs:");
    probs.print();
    
    // ボードを表示
    let mut rows = 0;
    while rows < N {
        let mut row_str = "";
        let mut cols = 0;
        while cols < N {
            if probs[rows, cols] > 0.5 {
                print(" Q ");
            } else {
                print(" . ");
            }
            cols = cols + 1;
        }
        println("");
        rows = rows + 1;
    }
    
    println("=== Done ===");
}
