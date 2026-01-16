fn main() {
    let N = 8;
    let solutions_to_find = 5;
    let mut found_count = 0;

    print("Finding "); print(solutions_to_find); println(" solutions for N-Queens (Masking Approach)...");

    let K = 2 * N - 1;
    // Pre-compute masks for diagonals (Shape: [K, N, N])
    // These are constant (no gradient needed), used to filter the board.
    let anti_diag_mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
         if r + c == k { 1.0 } else { 0.0 }
    } ];
    let main_diag_mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
         if r - c + N - 1 == k { 1.0 } else { 0.0 }
    } ];

    while found_count < solutions_to_find {
        print("=== Attempting solution #"); println(found_count + 1);
        
        let lr = 0.5;
        let epochs = 2000;
        let mut board = Tensor::randn([N, N], true);

        for i in 0..epochs {
             let probs = board.softmax(1);
             
             // 1. Column constraint: Each column sum should be 1.0
             let col_sums = probs.sum(0);
             let col_loss = (col_sums - 1.0).pow(2).sum();

             // 2. Diagonal constraints: Each diagonal sum should be <= 1.0
             // Use broadcasting: (1, N, N) * (K, N, N) -> (K, N, N)
             let probs_b = probs.reshape([1, N, N]);
             
             let anti_diag_sums = (probs_b * anti_diag_mask).sum(2).sum(1); // Sum over r, c -> (K)
             let main_diag_sums = (probs_b * main_diag_mask).sum(2).sum(1);

             let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
             let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
             
             // 3. Regularization: Force values to be close to 0 or 1 (Integer constraints)
             // Minimize x * (1 - x), which is 0 at x=0 and x=1, max at x=0.5
             let reg_loss = (probs * (1.0 - probs)).sum() * 0.05;

             let constraint_loss = col_loss + anti_diag_loss + main_diag_loss;
             let total_loss = constraint_loss + reg_loss;

             if i % 100 == 0 {
                 print("Epoch "); print(i); print(" - Loss: "); println(total_loss.item());
             }

             if constraint_loss.item() < 1e-4 {
                 // Early exit handled by check outside loop for now
             }

             total_loss.backward();
             let g = board.grad();
             board = board - g * lr;
             board = board.detach();
             board.enable_grad();
        }

        // Check result
        let probs = board.softmax(1);
        let col_sums = probs.sum(0);
        let col_loss = (col_sums - 1.0).pow(2).sum();
        let probs_b = probs.reshape([1, N, N]);
        let anti_diag_sums = (probs_b * anti_diag_mask).sum(2).sum(1);
        let main_diag_sums = (probs_b * main_diag_mask).sum(2).sum(1);
        let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
        let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
        let reg_loss = (probs * (1.0 - probs)).sum() * 0.05;
        let constraint_loss = col_loss + anti_diag_loss + main_diag_loss;
        let total_loss = constraint_loss + reg_loss;

        print("Final loss: "); println(total_loss.item());

        if constraint_loss.item() < 1e-2 {
            found_count = found_count + 1;
            print("Solution #"); println(found_count);
            
            let mut rows = 0;
            while rows < N {
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
            println("----------------");
        } else {
            println("Failed to converge, retrying...");
        }
    }
    println("Done!");
}
