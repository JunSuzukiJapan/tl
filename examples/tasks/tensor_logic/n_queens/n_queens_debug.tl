extern fn tl_get_refcount_count() -> i64;
extern fn tl_get_pool_count() -> i64;

fn main() {
    let N = 8;
    let K = 2 * N - 1;
    
    // Pre-compute masks (Tensor Comprehension Check)
    println("Creating Masks...");
    let anti_diag_mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
         if r + c == k { 1.0 } else { 0.0 }
    } ];
    let main_diag_mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
         if r - c + N - 1 == k { 1.0 } else { 0.0 }
    } ];
    println("Masks Created.");

    let initial_refs = tl_get_refcount_count();
    let initial_pool = tl_get_pool_count();
    println("Initial Refs: {}", initial_refs);
    println("Initial Pool: {}", initial_pool);

    let lr = 0.5;
    // Run enough epochs to see small leaks accumulate
    let epochs = 10;
    
    let epochs = 500;
    let mut board = Tensor::randn([N, N], true);

    for i in 0..epochs {
         let probs = board.softmax(1);
         
         let col_sums = probs.sum(0);
         let col_loss = (col_sums - 1.0).pow(2).sumall();

         let probs_b = probs.reshape([1, N, N]);
         // Broadcasting Ops check
         let anti_diag_sums = (probs_b * anti_diag_mask).sum(2).sum(1);
         let main_diag_sums = (probs_b * main_diag_mask).sum(2).sum(1);

         let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sumall();
         let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sumall();
         
         let reg_loss = (probs * (1.0 - probs)).sumall() * 0.05;

         let constraint_loss = col_loss + anti_diag_loss + main_diag_loss;
         let total_loss = constraint_loss + reg_loss;

         if i % 100 == 0 {
             println("Epoch {}: {}", i, total_loss.item());
             println("  Refs: {} Pool: {}", tl_get_refcount_count(), tl_get_pool_count());
         }

         total_loss.backward();
         let g = board.grad();
         board = board - g * lr;
         board = board.detach();
         board.enable_grad();
    }

    let final_refs = tl_get_refcount_count();
    let final_pool = tl_get_pool_count();
    println("Final Refs: {}", final_refs);
    println("Final Pool: {}", final_pool);
    println("Diff Refs: {}", final_refs - initial_refs);
}
