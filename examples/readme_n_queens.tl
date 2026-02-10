// CPU is faster than GPU for this task.
// To run on CPU, use the following command:
// ```bash
// cargo run --release -- --device=cpu examples/readme_n_queens.tl
// ```

fn main() {
    let N = 8;
    let solutions_to_find = 1;
    let mut found_count = 0;

    println("Finding {} solutions for N-Queens...", solutions_to_find);

    let K = 2 * N - 1;
    // Pre-compute masks for diagonals
    let anti_diag_mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
         if r + c == k { 1.0 } else { 0.0 }
    } ];
    let main_diag_mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
         if r - c + N - 1 == k { 1.0 } else { 0.0 }
    } ];

    while found_count < solutions_to_find {
        // Flush Metal command buffers to check for driver-side memory accumulation.
        System::metal_sync();
        println(
            "Memory: {} MB (metal_pool_mb={}, metal_pool_count={}, metal_pool_bytes={})",
            System::memory_bytes() / 1024 / 1024,
            System::metal_pool_mb(),
            System::metal_pool_count(),
            System::metal_pool_bytes()
        );
        println("=== Attempting solution #{}", found_count + 1);
        
        let lr = 0.5;
        let epochs = 2000;
        let mut board = Tensor::randn([N, N], true);

        for i in 0..epochs {
             let probs = board.softmax(1);
             
             // Constraint losses
             let col_sums = probs.sum(0);             
             let col_diff = col_sums - 1.0;
             let col_pow = col_diff.pow(2);
             let col_loss = col_pow.sumall();

             let probs_b = probs.reshape([1, N, N]);
             
             let anti_diag_sums = (probs_b * anti_diag_mask).sum(2).sum(1);
             let main_diag_sums = (probs_b * main_diag_mask).sum(2).sum(1);

             let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sumall();
             let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sumall();
             
             // Regularization to push toward 0/1
             let reg_loss = (probs * (1.0 - probs)).sumall() * 0.05;

             let constraint_loss = col_loss + anti_diag_loss + main_diag_loss;
             let total_loss = constraint_loss + reg_loss;

            if i % 100 == 0 {
                 println("Epoch {} (Mem: {} MB)", i, System::memory_bytes() / 1024 / 1024);

                 // Early exit: Stop when we have N queens (prob > 0.5)
                 let current_queens = [r, c | r <- 0..N, c <- 0..N {
                     if probs[r, c] > 0.5 { 1.0 } else { 0.0 }
                 }].sumall().item();

                 if current_queens == 8.0 {
                     println("Converged at epoch {} (found {} queens)", i, current_queens);
                     break;
                 }
            }

             total_loss.backward();
             let g = board.grad();
             board = board - g * lr;
             board = board.detach();
             board.enable_grad();
        }


        // Validate result
        let probs = board.softmax(1);
        
        // Count queens and validate
        let mut valid = true;
        let mut total_queens = 0;
        let mut rows = 0;
        while rows < N {
            let mut queen_count = 0;
            let mut cols = 0;
            while cols < N {
               if probs[rows, cols] > 0.5 {
                   queen_count = queen_count + 1;
                   total_queens = total_queens + 1;
               }
               cols = cols + 1;
            }
            if queen_count != 1 {
                valid = false;
            }
            rows = rows + 1;
        }

        if valid && total_queens == N {
            found_count = found_count + 1;
            println("Solution #{}", found_count);
            
            let mut rows2 = 0;
            while rows2 < N {
                let mut cols2 = 0;
                while cols2 < N {
                   if probs[rows2, cols2] > 0.5 {
                       print(" Q ");
                   } else {
                       print(" . ");
                   }
                   cols2 = cols2 + 1;
                }
                println("");
                rows2 = rows2 + 1;
            }
            println("----------------");
        } else {
            println("Invalid (queens={}), retrying...", total_queens);
        }

        if found_count == solutions_to_find {
            break;
        }

    }

    println("Done!");
}
