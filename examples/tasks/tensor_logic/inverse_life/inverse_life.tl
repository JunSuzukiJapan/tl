fn main() {
    print("Inverse Game of Life...\n");

    let N = 10;
    let STEPS = 5;

    // 1. Generate Ground Truth (Initial State)
    // Random binary state 0 or 1
    // logits > 0 -> 1, < 0 -> 0
    // To make it more binary, we can multiply logits by large number
    let gt_logits = Tensor::randn([1, 1, N, N], false) * 5.0;
    let gt_state = gt_logits.sigmoid(); // approx 0 or 1

    print("Target Initial State (Ground Truth):\n");
    // gt_state.reshape([N, N]).print();

    // 2. Define Kernel for Neighbor Counting
    // 3x3 kernel with 1s, center 0
    // Manually construct: [1,1,1, 1,0,1, 1,1,1]
    let k_data = Tensor::ones([9], false); 
    // Manual assignments not needed if we subtract center later, or construct carefully
    // Let's just create ones and subtract the center identity if needed, 
    // OR create from array if possible. 
    // Actually, simple hack: Neighbors = Conv2d(Ones) - State (subtract self)
    let kernel = Tensor::ones([1, 1, 3, 3], false);

    // 3. Run Forward Dynamics (Non-differentiable / Exact for Target Generation)
    // We want the target S_T to be the result of ACTUAL GoL rules.
    // However, implementing exact GoL with 'if' per cell is slow/hard in vector logic.
    // We can use the same "Soft" logic but with very sharp sigmoid (high beta) to emulate exact rules.
    
    let mut current = gt_state.clone();
    let mut t = 0;
    while t < STEPS {
        // Neighbors: Padding=1 to keep shape [N, N]
        let neighbors = current.conv2d(kernel, 1, 1) - current;
        
        // Exact Rules:
        // Born: N == 3
        // Survive: N == 2 or N == 3
        // Else: 0
        
        // Emulate with masking for GT generation (using sharp math or logic if available)
        // Since we don't have boolean mask logic easily exposed for tensors yet (e.g. (n==3).float()),
        // we will use the Differentiable Approximation for BOTH generation and training,
        // but verify it acts "binary-like".
        
        // Soft Rule parameters
        let c = 10.0; // Sharpness
        let is_3 = ((neighbors - 3.0).abs() * -c).sigmoid(); // High if N near 3
        let is_2 = ((neighbors - 2.0).abs() * -c).sigmoid(); // High if N near 2
        
        // Update: is_3 + current * is_2
        let next_s = is_3 + current * is_2;
        current = next_s.clamp(0.0, 1.0); 
        t = t + 1;
    }
    let target_final = current.detach(); // S_T
    print("Target Final State (T=5):\n");
    // target_final.reshape([N, N]).print();

    // 4. Optimization Loop (Inverse Problem)
    // Init Learnable State
    let mut learnable_logits = Tensor::randn([1, 1, N, N], true);
    let opt_lr = 0.5;
    
    let mut iter = 0;
    while iter < 500 {
        // Forward Pass
        let state = learnable_logits.sigmoid(); // [0,1] continuous
        
        let mut t = 0;
        let mut current_est = state;
        
        while t < STEPS {
            let neighbors = current_est.conv2d(kernel, 1, 1) - current_est;
            
            // Same update rule (using exp based soft-equality)
            let diff_3 = neighbors - 3.0;
            let diff_2 = neighbors - 2.0;
            let is_3 = (diff_3.pow(2) * -1.0).exp();
            let is_2 = (diff_2.pow(2) * -1.0).exp();
            
            // Soft update
            let next_s = is_3 + current_est * is_2;
            current_est = next_s; // Allow gradients to flow
            
            t = t + 1;
        }
        
        // Loss: MSE(current_est, target_final)
        let diff = current_est - target_final;
        let loss = diff.pow(2).sumall();
        
        if (iter / 50) * 50 == iter {
             println("Iter {} Loss: {}", iter, loss.item());
        }
        
        loss.backward();
        
        let g = learnable_logits.grad();
        learnable_logits = learnable_logits - g * opt_lr;
        learnable_logits = learnable_logits.detach();
        learnable_logits.enable_grad();
        
        iter = iter + 1;
    }
    
    print("Optimization Done.\n");
    print("Final Loss: ");
    
    // Recalc final loss
    let mut state = learnable_logits.sigmoid();
    let mut t = 0;
    while t < STEPS {
        let neighbors = state.conv2d(kernel, 1, 1) - state;
        let is_3 = ((neighbors - 3.0).pow(2) * -1.0).exp();
        let is_2 = ((neighbors - 2.0).pow(2) * -1.0).exp();
        state = is_3 + state * is_2;
        t = t + 1;
    }
    let diff = state - target_final;
    let final_loss = diff.pow(2).sumall();
    println(final_loss.item());
    
    // Compare Recovered S_0 with Ground Truth S_0
    print("Recovered Initial State vs GT:\n");
    let recovered = learnable_logits.sigmoid();
    
    let diff_init = recovered - gt_state;
    let match_errors = diff_init.abs().sumall();
    println("Mismatched Cells in Initial State: {}", match_errors.item());
}
