// Minimal N-Queens memory leak test - 3 epochs only
fn main() {
    let N = 8;
    let K = 2 * N - 1;

    let anti_diag_mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
         if r + c == k { 1.0 } else { 0.0 }
    } ];
    let main_diag_mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
         if r - c + N - 1 == k { 1.0 } else { 0.0 }
    } ];

    let lr = 0.5;
    let mut board = Tensor::randn([N, N], true);

    println("Before loop: {} MB", System::memory_bytes() / 1024 / 1024);

    for i in 0..3 {
         let probs = board.softmax(1);
         let col_sums = probs.sum(0);
         let col_diff = col_sums - 1.0;
         let col_pow = col_diff.pow(2);
         let col_loss = col_pow.sumall();

         let probs_b = probs.reshape([1, N, N]);
         let anti_diag_sums = (probs_b * anti_diag_mask).sum(2).sum(1);
         let main_diag_sums = (probs_b * main_diag_mask).sum(2).sum(1);

         let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sumall();
         let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sumall();

         let reg_loss = (probs * (1.0 - probs)).sumall() * 0.05;
         let constraint_loss = col_loss + anti_diag_loss + main_diag_loss;
         let total_loss = constraint_loss + reg_loss;

         println("Epoch {} (Mem: {} MB)", i, System::memory_bytes() / 1024 / 1024);

         total_loss.backward();
         let g = board.grad();
         board = board - g * lr;
         board = board.detach();
         board.enable_grad();
    }

    println("After loop: {} MB", System::memory_bytes() / 1024 / 1024);
    println("Done!");
}
