fn main() {
    print("Differentiable Ray Caster 2D (Vectorized)...\n");

    let N = 8;         // Scene grid size
    let NUM_RAYS = 16; // Number of rays
    let NUM_STEPS = 8; // Steps along each ray
    let MAX_DIST = 4.0;
    
    // 1. Create Ground Truth Scene
    let mut gt_scene = Tensor::zeros([N, N]);
    gt_scene[3, 3] = 1.0;
    gt_scene[3, 4] = 1.0;
    gt_scene[4, 3] = 1.0;
    gt_scene[4, 4] = 1.0;
    print("Ground Truth Scene Created.\n");

    // 2. Create Ray Directions (Vectorized)
    let PI = 3.14159265;
    let mut ray_dirs_x = Tensor::zeros([NUM_RAYS]);
    let mut ray_dirs_y = Tensor::zeros([NUM_RAYS]);
    
    // We construct angles manually or via loop, but since it's init logic, loop is fine.
    // However, to supply to vectorized render, we reshape them.
    let mut r = 0;
    while r < NUM_RAYS {
        let angle = 2.0 * PI * (r as f32) / (NUM_RAYS as f32);
        let angle_t = Tensor::zeros([1]) + angle;
        ray_dirs_x[r] = angle_t.cos().item();
        ray_dirs_y[r] = angle_t.sin().item();
        r = r + 1;
    }
    
    // Reshape for broadcasting: [NUM_RAYS, 1, 1]
    let rays_x_b = ray_dirs_x.reshape([NUM_RAYS, 1, 1]);
    let rays_y_b = ray_dirs_y.reshape([NUM_RAYS, 1, 1]);

    print("Ray Directions Created.\n");

    // 3. Grid for broadcasting [1, N, N]
    let mut grid_i = Tensor::zeros([N, N]);
    let mut grid_j = Tensor::zeros([N, N]);
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < N {
            grid_i[i, j] = i as f32;
            grid_j[i, j] = j as f32;
            j = j + 1;
        }
        i = i + 1;
    }
    let grid_i_b = grid_i.reshape([1, N, N]);
    let grid_j_b = grid_j.reshape([1, N, N]);

    let step_size = MAX_DIST / (NUM_STEPS as f32);
    let cx = N / 2;
    let cy = N / 2;
    let sigma = 1.5;
    let epsilon = 0.0001;

    // Helper function-like block for rendering
    // Since we don't have first-class functions easily, we duplicate logic or wrap in loop
    // But here we can't easily wrap "render from scene X" in a function due to tensor ownership quirks in example code style.
    // So we will just implement the loop inline for GT, then for Pred.

    // ---------------------------------------------------------
    // RENDER GROUND TRUTH
    // ---------------------------------------------------------
    let mut gt_image = Tensor::zeros([NUM_RAYS]); // Accumulator
    
    // Iterate steps (vectorized across rays)
    let mut s = 0;
    while s < NUM_STEPS {
        let dist = (s as f32) * step_size;
        
        // Positions [NUM_RAYS, 1, 1]
        let px = rays_x_b * dist + (cx as f32);
        let py = rays_y_b * dist + (cy as f32);
        
        // Differences [NUM_RAYS, N, N]
        let dx = grid_i_b - px;
        let dy = grid_j_b - py;
        let dist_sq = dx.pow(2) + dy.pow(2);
        
        // Gaussian weights
        let weights = (dist_sq * (-1.0 / (sigma * sigma))).exp();
        
        // Weighted sum
        // weights: [NUM_RAYS, N, N]
        // gt_scene: [N, N] -> Broadcast to [NUM_RAYS, N, N] implicitly?
        // Wait, standard broadcasting rules:
        // weights: [NUM_RAYS, N, N]
        // scene:             [N, N]
        // result:  [NUM_RAYS, N, N]
        // Correct.
        
        let weighted_scene = weights * gt_scene;
        
        // Sum over N, N dims.
        // sum(1) -> [NUM_RAYS, N]
        // sum(1) -> [NUM_RAYS]
        let num = weighted_scene.sum(1).sum(1);
        let den = weights.sum(1).sum(1) + epsilon;
        
        let sample = num / den;
        
        gt_image = gt_image + sample;
        
        s = s + 1;
    }
    // "Freeze" GT image (it's constant, but make sure we don't backprop through it if we used it later, though here it's numbers)
    // Actually gt_image is from non-variable tensors, so it has no grad.

    println("Ground Truth Image Rendered (sum={})", gt_image.sumall().item());


    // 4. Optimization
    let mut learnable_scene = Tensor::randn([N, N], true) * 0.1;

    let mut iter = 0;
    while iter < 200 {
        let mut pred_image = Tensor::zeros([NUM_RAYS]); // Accumulator
        
        // RENDER PREDICTION
        let mut s = 0;
        while s < NUM_STEPS {
            let dist = (s as f32) * step_size;
            let px = rays_x_b * dist + (cx as f32);
            let py = rays_y_b * dist + (cy as f32);
            
            let dx = grid_i_b - px;
            let dy = grid_j_b - py;
            let dist_sq = dx.pow(2) + dy.pow(2);
            let weights = (dist_sq * (-1.0 / (sigma * sigma))).exp();
            
            // learnable_scene is [N, N]. Broadcasts against weights [R, N, N].
            let weighted_scene = weights * learnable_scene;
            
            let num = weighted_scene.sum(1).sum(1);
            let den = weights.sum(1).sum(1) + epsilon;
            
            pred_image = pred_image + num / den;
            
            s = s + 1;
        }
        
        // Loss: MSE
        let diff = pred_image - gt_image;
        let loss = diff.pow(2).sumall();
        
        if (iter / 50) * 50 == iter {
            println("Iter {} Loss: {}", iter, loss.item());
        }
        
        loss.backward();
        
        let g = learnable_scene.grad();
        // Simple SGD
        learnable_scene = learnable_scene - g * 0.01;
        learnable_scene = learnable_scene.detach();
        learnable_scene.enable_grad(); // Re-enable for next iter
        
        iter = iter + 1;
    }

    print("Optimization Done.\n");
    print("Learned scene center value: ");
    println(learnable_scene[3, 3] + learnable_scene[4, 4]);
}
