fn main() {
    print("Vectorized TSP (Corrected)...\n");
    let N = 5;
    let cities = Tensor::randn([N, 2], false); // [N, 2]

    // 1. Squared Distance Matrix (Vectorized)
    // D_sq = ||X_i - X_j||^2 = ||X_i||^2 + ||X_j||^2 - 2 <X_i, X_j>
    
    // X^2 -> [N, 2]
    // sum(1) -> [N]. reshape([N, 1]) -> [N, 1] for broadcasting.
    let city_norms_sq = cities.pow(2).sum(1).reshape([N, 1]); 
    
    // term1 [N, 1] + term2 [1, N] -> [N, N] Broadcasting
    // transpose(0, 1) swaps dims 0 and 1.
    let term1 = city_norms_sq;
    let term2 = city_norms_sq.transpose(0, 1);
    let sum_sq = term1 + term2;
    
    // X @ X.T -> [N, N]
    let term3 = cities.matmul(cities.transpose(0, 1)); 
    
    let dist_sq = sum_sq - term3 * 2.0; // [N, N]
    
    // let dist_matrix = dist_sq.relu().pow(0.5);
    let dist_matrix = dist_sq.relu().pow(0.5);
    
    print("Distance Matrix:\n");
    // dist_matrix.print();

    // 2. Optimization Loop
    let mut logits = Tensor::randn([N, N], true);
    
    // Construct Shift Matrix S (constant)
    let mut S = Tensor::zeros([N, N]);
    let mut k = 0;
    while k < N {
        let next_k = if k + 1 == N { 0 } else { k + 1 };
        S[k, next_k] = 1.0;
        k = k + 1;
    }
    
    let mut i = 0;
    while i < 2000 {
        let probs = logits.softmax(1);
        
        let path = probs.matmul(cities); 
        let path_next = S.matmul(path); 
        
        let diff = path - path_next;
        let diff_sq = diff.pow(2); 
        // sum() -> scalar (rank 0)
        let tour_loss = diff_sq.sum(); 

        // Constraint: Each city visited exactly once (Column sum = 1)
        // Row sum = 1 is guaranteed by Softmax(1).
        // Check Col sums.
        // sum(0) -> [N].
        let col_sums = probs.sum(0); 
        let diff_col = col_sums - 1.0;
        let constraint_loss = diff_col.pow(2).sum();
        
        let total_loss = tour_loss + constraint_loss * 5.0;
        
        if (i / 100) * 100 == i {
            print("Epoch "); print(i);
            print(" Tour Length: "); print(tour_loss.item());
            print(" Constraint: "); println(constraint_loss.item());
        }
        
        total_loss.backward();
        
        let g = logits.grad();
        logits = logits - g * 0.1; 
        let logits_detached = logits.detach();
        logits = logits_detached;
        logits.enable_grad();
        
        i = i + 1;
    }
    
    print("Final:\n");
    let probs = logits.softmax(1);
    // probs.round().print();
    print("Tour Length: ");
    
    // Recalc final tour length
    let path = probs.matmul(cities);
    let path_next = S.matmul(path);
    let diff = path - path_next;
    let final_loss = diff.pow(2).sum();
    println(final_loss.item());

}
