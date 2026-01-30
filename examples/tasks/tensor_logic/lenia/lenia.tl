fn main() {
    let H = 64;
    let W = 64;
    
    // 1. Initial State: Random noise (0.0 to 1.0)
    // (B=1, C=1, H, W)
    let state = Tensor::randn([1, 1, H, W], false); 
    let state = state.abs().clamp(0.0, 1.0);
    
    // 2. Convolution Kernel (Ring / DoG - Difference of Gaussians)
    // For simplicity, we use a simple Sobel-like or circular kernel.
    // Let's create a 7x7 kernel with a ring pattern manually.
    let K_size = 7;
    // We can't do [x, y | ...] yet for tensor creation easily without full loops, 
    // or use tensor comprehension if implemented? 
    // Let's just use ones for neighbor sum and normalize.
    let kernel = Tensor::ones([1, 1, K_size, K_size], false);
    
    // Normalize kernel so sum is 1.0 roughly? Or just keep as sum.
    let k_sum = K_size * K_size; 
    let kernel = kernel / (k_sum as f32); 

    let steps = 100;
    
    println("Starting Lenia Simulation...");
    println("Grid Size: 64x64");
    println("Steps: 100");

    for i in 0..steps {
        // A. Convolution (Neighbor sensing)
        // input: (1, 1, H, W)
        // weight: (1, 1, K, K) (OutC=1, InC=1)
        // padding: K_size / 2 = 3
        let pad = K_size / 2;
        let neighbors = state.conv2d(kernel, pad, 1);
        
        // B. Growth Function
        // Lenia standard: G(u) = 2 * exp( - (u - mu)^2 / (2 * sigma^2) ) - 1
        // mu = 0.15, sigma = 0.015
        let mu = 0.15;
        let sigma = 0.015;
        
        let diff = neighbors - mu;
        let sq_diff = diff * diff;
        let exponent = sq_diff.neg() / (2.0 * sigma * sigma);
        let growth = exponent.exp() * 2.0 - 1.0;
        
        // C. Update State
        // state = clip(state + dt * growth, 0, 1)
        let dt = 0.1;
        let state = state + growth * dt;
        let state = state.clamp(0.0, 1.0);
        
        // D. Render (ASCII)
        // Every 10 steps
        if i % 10 == 0 {
             print("Step: ");
             println(i);
             // Sample center 10x10
             let center_y = H / 2;
             let center_x = W / 2;
             // Manual slice print loop (since slice syntax is limited for multi-dim print)
             // Just print sum for sanity check
             let total_mass = state.sumall();
             print("Total Mass: ");
             println(total_mass.item());
        }
    }
    
    println("Simulation Complete.");
}
