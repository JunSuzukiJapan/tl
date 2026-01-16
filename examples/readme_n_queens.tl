fn main() {
    let N = 8; // ボードサイズ (8x8)
    let solutions_to_find = 5; // 見つける解の数
    let found_count = 0;

    print("Finding "); print(solutions_to_find); println(" solutions for N-Queens...");

    while found_count < solutions_to_find {
        print("=== Attempting solution #"); println(found_count + 1);
        
        let lr = 0.5;
        let epochs = 2000;

        // 1. ボードの確率分布を初期化 (ランダムノイズ)
        let board = Tensor::randn([N, N], true);
        println("Board initialized");

        // 最適化ループ
        for i in 0..epochs {
             let probs = board.softmax(1);

             // 制約1: 列制約
             let col_sums = probs.sum(0);
             let col_loss = (col_sums - 1.0).pow(2).sum();

             // 制約2: 対角線制約 (Tensor Comprehension)
             let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
             let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];

             let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
             let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
             
             let total_loss = col_loss + anti_diag_loss + main_diag_loss;

             // ログ出力 (100エポックごと)
             if i % 100 == 0 {
                 print("Epoch "); print(i); print(" - Loss: "); println(total_loss.item());
             }

             // 早期終了判定
             if total_loss.item() < 1e-4 {
                 print("Converged at epoch "); println(i);
             }

             total_loss.backward();
             let g = board.grad();
             let board = board - g * lr;
             let board = board.detach();
             board.enable_grad();
        }

        // --- 結果判定と表示 ---
        let probs = board.softmax(1);
        
        // 損失を再計算してチェック
        let col_sums = probs.sum(0);
        let col_loss = (col_sums - 1.0).pow(2).sum();
        let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
        let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];
        let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
        let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
        let total_loss = col_loss + anti_diag_loss + main_diag_loss;

        print("Final loss: "); println(total_loss.item());

        if total_loss.item() < 1e-3 {
            found_count = found_count + 1;
            print("Solution #"); println(found_count);
            
            let rows = 0;
            while rows < N {
                let cols = 0;
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
            println("----------------");
        } else {
            println("Failed to converge, retrying...");
        }
    }
    println("Done!");
}
