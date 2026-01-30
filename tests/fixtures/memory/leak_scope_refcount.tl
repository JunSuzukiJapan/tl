// STDOUT: LEAK_TEST_OK

fn main() {
    let start_scopes = System::scope_depth();
    let start_refs = System::refcount_count();

    let mut ok = 1;

    let mut iter = 0;
    while iter < 50 {
        // Tensor comprehension (compile_tensor_equation path)
        let N = 32;
        let K = 2 * N - 1;
        let _mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
            if r + c == k { 1.0 } else { 0.0 }
        } ];

        // Loop body with break/continue to exercise cleanup
        let mut i = 0;
        while i < 6 {
            if i == 1 {
                i = i + 1;
                continue;
            }
            if i == 2 {
                break;
            }

            let board = Tensor::randn([128, 128], true);
            let probs = board.softmax(1);
            let col_sums = probs.sum(0);
            let col_loss = (col_sums - 1.0).pow(2).sumall();
            let reg_loss = (probs * (1.0 - probs)).sumall() * 0.05;
            let total = col_loss + reg_loss;
            total.backward();
            let g = board.grad();
            let _gsum = g.sumall();
            let _v = (board * 0.5 + g * 0.1).sumall();

            i = i + 1;
        }

        iter = iter + 1;
    }

    let end_scopes = System::scope_depth();
    let end_refs = System::refcount_count();

    if end_scopes != start_scopes {
        print("FAILED: scope_depth ");
        print(start_scopes);
        print(" -> ");
        println(end_scopes);
        ok = 0;
    }

    if end_refs != start_refs {
        print("FAILED: refcount ");
        print(start_refs);
        print(" -> ");
        println(end_refs);
        ok = 0;
    }

    if ok == 1 {
        println("LEAK_TEST_OK");
    }
}
