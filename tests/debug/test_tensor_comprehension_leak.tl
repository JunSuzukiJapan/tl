// test_tensor_comprehension_leak.tl
// Test if tensor comprehension causes memory leak in while loop

fn main() {
    println("Testing tensor comprehension memory...");

    let N = 8;
    let attempt = 0;
    let max_attempts = 5;

    while attempt < max_attempts {
        print("=== Attempt "); println(attempt);
        
        let board = Tensor::randn([N, N], true);

        for i in 0..200 {
            if i % 50 == 0 {
                print("  Epoch "); println(i);
            }

            let probs = board.softmax(1);
            
            // Tensor comprehension (like N-Queens diagonal sums)
            let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
            let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];
            
            let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
            let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
            let total_loss = anti_diag_loss + main_diag_loss;

            total_loss.backward();
            let g = board.grad();
            board = board - g * 0.5;
            board = board.detach();
            board.enable_grad();
        }

        println("  Attempt done");
        attempt = attempt + 1;
    }
    
    println("All attempts done!");
}
