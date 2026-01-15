fn main() {
    let N = 20;
    let epochs = 500;
    let lr = 0.5;

    print("Initializing MLN for N=");
    println(N);

    // --- Knowledge Base (Weights) ---
    // Rule 1: Smokes(x) => Cancer(x)
    let w_smokes_cancer = 2.0;
    // Rule 2: Friends(x, y) => (Smokes(x) <=> Smokes(y))
    let w_friends_same = 1.0;

    // --- Evidence ---
    // Friends network: sparse random graph
    print("Generating network...");
    let friends_raw = Tensor::randn([N, N], false);
    // Create symmetric friendship matrix (approx 1.0 if > 1.0, else 0.0)
    // Using sigmoid approximation for step function to avoid if-expr issues
    let friends = [r, c | r <- 0..N, c <- 0..N {
        let val = friends_raw[r, c];
        // sigmoid(50 * (val - 1.0))
        let logits = (val - 1.0) * 50.0;
        1.0 / (1.0 + (logits * -1.0).exp())
    }];
    
    // Make it symmetric: max(A, B) ~ A + B - A*B for 0/1? Or just (A+B)/2?
    // Or just symmetric logic: if (r < c) use [r,c] else [c,r].
    // But conditional...
    // Let's just use A*B + (1-A)*B ... no.
    // Soft OR: A + B - A*B
    let friends = [r, c | r <- 0..N, c <- 0..N {
        let f1 = friends[r, c];
        let f2 = friends[c, r];
        f1 + f2 - f1 * f2
    }];
    
    println(" Done.");

    // --- Predicates (Unknowns) ---
    // Smokes(x) and Cancer(x)
    // Initialize logits with random values
    let smokes_logits = Tensor::randn([N], true);
    let cancer_logits = Tensor::randn([N], true);

    // --- Inference (Training) Loop ---
    print("Starting inference...");
    println("");

    for i in 0..epochs {
        // Probability relaxation: sigmoid(logit)
        let smokes_prob = (smokes_logits * -1.0).exp();
        let smokes_prob = 1.0 / (1.0 + smokes_prob);

        let cancer_prob = (cancer_logits * -1.0).exp();
        let cancer_prob = 1.0 / (1.0 + cancer_prob);

        // Rule 1: Smokes => Cancer
        // Logic: A => B  is equivalent to  1 - A + A*B  (Reichenbach)
        let sat_1 = 1.0 - smokes_prob + smokes_prob * cancer_prob;
        let energy_1 = (sat_1 * w_smokes_cancer).sum();

        // Rule 2: Friends(x, y) => (Smokes(x) <=> Smokes(y))
        // Similarity: S(x) <=> S(y)  is  S(x)S(y) + (1-S(x))(1-S(y))
        // We broadcast manually using comprehensions
        let sim = [r, c | r <- 0..N, c <- 0..N {
            let s_r = smokes_prob[r];
            let s_c = smokes_prob[c];
            s_r * s_c + (1.0 - s_r) * (1.0 - s_c)
        }];
        
        // Apply Friends mask
        let sat_2 = friends * sim;
        let energy_2 = (sat_2 * w_friends_same).sum();

        // Total Energy (maximize this)
        let total_energy = energy_1 + energy_2;
        
        // Loss (minimize negative energy)
        let loss = total_energy * -1.0;

        if (i / 50) * 50 == i {
            print("Epoch ");
            print(i);
            print(" Loss: ");
            println(loss.item());
        }

        // Backprop
        loss.backward();

        // SGD Update
        let g_s = smokes_logits.grad();
        let g_c = cancer_logits.grad();



        smokes_logits = smokes_logits - g_s * lr;
        cancer_logits = cancer_logits - g_c * lr;

        // Detach and enable grad for next iteration
        smokes_logits = smokes_logits.detach();
        cancer_logits = cancer_logits.detach();
        smokes_logits.enable_grad();
        cancer_logits.enable_grad();
    }

    println("Inference complete.");
    
    // --- Analysis ---
    let smokes_prob = (smokes_logits * -1.0).exp();
    let smokes_prob = 1.0 / (1.0 + smokes_prob);
    let cancer_prob = (cancer_logits * -1.0).exp();
    let cancer_prob = 1.0 / (1.0 + cancer_prob);

    print("Avg Smokes Prob: ");
    println(smokes_prob.mean().item());
    print("Avg Cancer Prob: ");
    println(cancer_prob.mean().item());

    // Check Rule 1 Consistency
    let sat_1 = 1.0 - smokes_prob + smokes_prob * cancer_prob;
    print("Rule 1 (Smokes=>Cancer) Satisfaction Avg: ");
    // println(sat_1.mean().item());
    let mean_sat = sat_1.mean();
    println(mean_sat.item());
}
