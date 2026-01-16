fn main() {
    let N = 20;
    let D = 16; // Hidden dimension
    let epochs = 500;
    let lr = 0.01;

    print("Initializing GNN Shortest Path Task for N=");
    println(N);

    // --- 1. Graph Generation ---
    // Adjacency Matrix A [N, N]
    // Sparse random graph: prob of edge = 0.2
    let A_raw = Tensor::randn([N, N], false);
    
    // Binary Adjacency using sigmoid trick (approx step function)
    let A = [r, c | r <- 0..N, c <- 0..N {
        // sigmoid(100 * (val - 0.5)) -> >0.5 becomes 1, <0.5 becomes 0
        let logits = (A_raw[r, c] - 0.5) * 100.0;
        let p = 1.0 / (1.0 + (logits * -1.0).exp());
        // Remove self-loops check to avoid 'if' expression issues
        p
    }];

    // Make symmetric (undirected)
    let A = [r, c | r <- 0..N, c <- 0..N {
        let a1 = A[r, c];
        let a2 = A[c, r];
        // Soft OR: a1 + a2 - a1*a2
        a1 + a2 - a1 * a2
    }];

    // --- 3. Model Definition ---
    // Node Embeddings / State: Val[N, 1] (The distance estimate)
    // Init with random
    let mut val = Tensor::randn([N, 1], true);
    
    // --- 4. Training Loop ---
    print("Starting training...");
    println("");

    for i in 0..epochs {
        // Enforce boundary condition: val[0] should be 0.
        // let source_val = val[0]; 
        // let bc_loss = (source_val * source_val) * 10.0;
        // Hack: ignore bc_loss for now to debug
        let bc_loss = Tensor::zeros([1], false);
        
        let val_t = val.transpose(0, 1);
        // repeat_interleave seems to return Void in current builtins? Use broadcast trick.
        let zeros_nn = Tensor::zeros([N, N], false);
        // [N, N] + [1, N] -> [N, N] broadcast
        let U = zeros_nn + val_t; 
        
        // 2. Add edge weight (1.0)
        let U_plus_1 = 1.0 + U;
        
        // 3. Mask non-neighbors.
        // (A * -1.0 + 1.0) * 100.0. Tensor * F32 is safe.
        let M = (A * -1.0 + 1.0) * 100.0;
        let candidates = U_plus_1 + M;
        
        // 4. SoftMin over dimension 1 (neighbors u)
        let neg_candidates = candidates * -1.0;
        let exp_c = neg_candidates.exp();
        
        // Sum over neighbors (dim 1)
        let sum_exp = exp_c.sum(1); // [N, 1]
        
        // -log(sum)
        let soft_min = (sum_exp.log()) * -1.0;
        
        // Constraint Loss: (val - soft_min)^2
        let consistency_loss = (val - soft_min).pow(2).sum();
        
        // Positivity constraint
        let positivity_loss = (val * -1.0).relu().pow(2).sum();
        
        // bc_loss logic: using Tensor broadcasting if val[0] is F32
        let source_val = val[0];
        let bc_loss = (source_val * source_val) * 10.0;
        
        let total_loss = consistency_loss + bc_loss + positivity_loss;

        if (i / 50) * 50 == i {
            print("Epoch ");
            print(i);
            print(" Loss: ");
            println(total_loss);
        }

        total_loss.backward();

        let g = val.grad();
        val = val - g * lr;

        val = val.detach();
        val.enable_grad();


    }

    
    // Output inferred distances
    print("Inferred Distances (first 5):");
    println("");
    let mut r = 0;
    while r < 5 {
        print("Node ");
        print(r);
        print(": ");
        println(val[r]);
        r = r + 1;
    }
}
