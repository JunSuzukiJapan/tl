fn main() {
    let argc = tl_args_count();
    
    let N = if argc > 0 {
        tl_string_to_i64(tl_args_get(0))
    } else {
        8
    };
    
    let num_solutions = if argc > 1 {
        tl_string_to_i64(tl_args_get(1))
    } else {
        10
    };
    
    print("Initializing N-Queens Solver. N="); print(N);
    print(", Solutions to find="); println(num_solutions);

    let lr = 0.5;
    let epochs = 2000;

    print("Initializing N-Queens Solver.");
    print(" N=");
    print(N);
    print(", Solutions to find=");
    println(num_solutions);
    
    let solutions_found = 0;
    
    // Loop to find multiple solutions
    // Since we don't have a 'while' with complex condition or 'break' easily,
    // we iterate enough times or use a large loop.
    // Optimization: Just loop `num_solutions` times for now, assuming 1 run = 1 attempt.
    // Ideally we retry until valid.
    
    let mut run = 0;
    while run < num_solutions {
        print("Run ");
        println(run + 1);

        // Initialize board logits with random noise
        // Using `requires_grad=true` for optimization
        let mut board = Tensor::randn([N, N], true);
        // let board = Tensor::zeros([N, N], true);
    
    // Optimizer loop
    for i in 0..epochs {
         let probs = board.softmax(1);
         let col_sums = probs.sum(0);
         let col_loss = (col_sums - 1.0).pow(2).sum();

         let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
         let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];

         let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
         let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
         
         let total_loss = col_loss + anti_diag_loss + main_diag_loss;

         if (i / 100) * 100 == i {
             print("Epoch ");
             print(i);
             print(" Loss: ");
             println(total_loss.item());
         }

         // Optimization Step
         total_loss.backward();
         
         let g = board.grad();
         board = board - g * lr;
         let board_detached = board.detach();
         board = board_detached;
         board.enable_grad();
    }

    print("Final Loss: ");
    let probs = board.softmax(1);
    let col_sums = probs.sum(0); 
    let col_loss = (col_sums - 1.0).pow(2).sum();
    let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
    let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];
    let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
    let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
    let total_loss = col_loss + anti_diag_loss + main_diag_loss;
    println(total_loss.item());

    print("Final Board (Probabilities > 0.5):");
    println("");
    
    let mut rows = 0;
    while rows < N {
        let mut cols = 0;
        while cols < N {
           let val = probs[rows, cols];
           if val > 0.5 {
               print(" Q ");
           } else {
               print(" . ");
           }
           cols = cols + 1;
        }
        println("");
        rows = rows + 1;
    }
    run = run + 1;
  }
}
