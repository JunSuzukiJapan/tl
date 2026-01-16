// test_n_queens_pattern.tl
// Reproduce N-Queens memory pattern: while loop with multiple attempts

fn main() {
    println("Testing N-Queens memory pattern...");

    let N = 8;
    let attempt = 0;
    let max_attempts = 5;  // Reduced for testing

    while attempt < max_attempts {
        print("=== Attempt "); println(attempt);
        
        // New board each attempt
        let board = Tensor::randn([N, N], true);
        let lr = 0.5;

        // Optimization loop (reduced from 2000)
        for i in 0..500 {
            if i % 100 == 0 {
                print("  Epoch "); println(i);
            }

            // Create many intermediate tensors (like N-Queens)
            let probs = board.softmax(1);
            let col_sums = probs.sum(0);
            let col_loss = (col_sums - 1.0).pow(2).sum();
            let total_loss = col_loss;

            // Backward and update
            total_loss.backward();
            let g = board.grad();
            board = board - g * lr;
            board = board.detach();
            board.enable_grad();
        }

        println("  Attempt done");
        attempt = attempt + 1;
    }
    
    println("All attempts done!");
}
