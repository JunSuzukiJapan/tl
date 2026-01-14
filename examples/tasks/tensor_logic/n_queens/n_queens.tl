fn main() {
    let N = 8;
    let lr = 0.5;
    let epochs = 2000;

    print("Initializing N-Queens Solver for N=");
    println(N);

    // Initialize board logits with random noise
    // Using `requires_grad=true` for optimization
    let board = Tensor::randn([N, N], true);
    // let board = Tensor::zeros([N, N], true);
    
    // Optimizer loop
    for i in 0..epochs {
         let probs = softmax(board, 1);
         let col_sums = sum(probs, 0);
         let col_loss = sum(pow(col_sums - 1.0, 2));

         let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
         let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];

         let anti_diag_loss = sum(pow(relu(anti_diag_sums - 1.0), 2));
         let main_diag_loss = sum(pow(relu(main_diag_sums - 1.0), 2));
         
         let total_loss = col_loss + anti_diag_loss + main_diag_loss;

         if (i / 100) * 100 == i {
             print("Epoch ");
             print(i);
             print(" Loss: ");
             println(item(total_loss));
         }

         // Optimization Step
         backward(total_loss);
         
         let g = board.grad();
         let board = board - g * lr;
         let board = board.detach();
         enable_grad(board);
    }

    print("Final Loss: ");
    let probs = softmax(board, 1);
    let col_sums = sum(probs, 0); 
    let col_loss = sum(pow(col_sums - 1.0, 2));
    let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
    let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];
    let anti_diag_loss = sum(pow(relu(anti_diag_sums - 1.0), 2));
    let main_diag_loss = sum(pow(relu(main_diag_sums - 1.0), 2));
    let total_loss = col_loss + anti_diag_loss + main_diag_loss;
    println(item(total_loss));

    print("Final Board (Probabilities > 0.5):");
    println("");
    
    let rows = 0;
    while rows < N {
        let cols = 0;
        while cols < N {
           let val = probs[rows, cols];
           if val > 0.5 {
               print(" Q ");
           } else {
               print(" . ");
           }
           let cols = cols + 1;
        }
        println("");
        let rows = rows + 1;
    }
}
