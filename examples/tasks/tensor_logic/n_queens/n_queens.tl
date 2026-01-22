fn main() {
    let argc = Args::count();
    
    let mut N = 8;
    if argc > 0 {
        N = Args::get(0).to_i64();
    }
    
    let mut num_solutions = 10;
    if argc > 1 {
        num_solutions = Args::get(1).to_i64();
    }
    
    println("Initializing N-Queens Solver. N={}, Solutions to find={}", N, num_solutions);

    let lr = 0.5;
    let epochs = 2000;
    
    let solutions_found = 0;
    
    let mut run = 0;
    while run < num_solutions {
        println("Run {}", run + 1);

        let mut board = Tensor::randn([N, N], true);
    
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
             println("Epoch {} Loss: {}", i, total_loss.item());
         }

         total_loss.backward();
         
         let g = board.grad();
         board = board - g * lr;
         let board_detached = board.detach();
         board = board_detached;
         board.enable_grad();
    }

    let probs = board.softmax(1);
    let col_sums = probs.sum(0); 
    let col_loss = (col_sums - 1.0).pow(2).sum();
    let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
    let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];
    let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
    let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
    let total_loss = col_loss + anti_diag_loss + main_diag_loss;
    println("Final Loss: {}", total_loss.item());

    println("Final Board (Probabilities > 0.5):");
    
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
