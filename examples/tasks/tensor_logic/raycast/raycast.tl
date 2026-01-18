fn main() {
    print("Differentiable Ray Caster 2D (Soft Sampling)...\n");

    let N = 8;         // Scene grid size (smaller for faster iteration)
    let NUM_RAYS = 16; // Number of rays
    let NUM_STEPS = 8; // Steps along each ray
    let MAX_DIST = 4.0;
    
    // 1. Create Ground Truth Scene (bright center)
    let gt_scene = Tensor::zeros([N, N]);
    gt_scene[3, 3] = 1.0;
    gt_scene[3, 4] = 1.0;
    gt_scene[4, 3] = 1.0;
    gt_scene[4, 4] = 1.0;
    print("Ground Truth Scene Created.\n");

    // 2. Create Ray Directions
    let PI = 3.14159265;
    let ray_dirs_x = Tensor::zeros([NUM_RAYS]);
    let ray_dirs_y = Tensor::zeros([NUM_RAYS]);
    let mut r = 0;
    while r < NUM_RAYS {
        let angle = 2.0 * PI * (r as f32) / (NUM_RAYS as f32);
        let angle_t = Tensor::zeros([1]) + angle;
        ray_dirs_x[r] = angle_t.cos().item();
        ray_dirs_y[r] = angle_t.sin().item();
        r = r + 1;
    }
    print("Ray Directions Created.\n");

    // 3. Render using Soft Sampling
    // For each sample position (px, py), compute a soft weight for each scene cell.
    // weight[i, j] = exp(-((i - px)^2 + (j - py)^2) / sigma^2)
    // sampled_value = sum(weight * scene) / sum(weight)
    // This is fully differentiable.
    
    let step_size = MAX_DIST / (NUM_STEPS as f32);
    let cx = N / 2;
    let cy = N / 2;
    let sigma = 1.5; // Larger Gaussian width for smoother sampling
    let epsilon = 0.0001; // Avoid division by zero
    
    // Precompute grid indices for broadcasting
    let grid_i = Tensor::zeros([N, N]);
    let grid_j = Tensor::zeros([N, N]);
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
    
    // Render GT Image
    let gt_image = Tensor::zeros([NUM_RAYS]);
    r = 0;
    while r < NUM_RAYS {
        let mut ray_acc = 0.0;
        let s = 0;
        while s < NUM_STEPS {
            let dist = (s as f32) * step_size;
            let px = (cx as f32) + ray_dirs_x[r] * dist;
            let py = (cy as f32) + ray_dirs_y[r] * dist;
            
            // Soft sample
            let dx = grid_i - px;
            let dy = grid_j - py;
            let dist_sq = dx.pow(2) + dy.pow(2);
            let weights = (dist_sq * (-1.0 / (sigma * sigma))).exp();
            let sample = (weights * gt_scene).sum() / (weights.sum() + epsilon);
            ray_acc = ray_acc + sample.item();
            
            s = s + 1;
        }
        gt_image[r] = ray_acc;
        r = r + 1;
    }
    println("Ground Truth Image Rendered (sum={})", gt_image.sum().item());

    // 4. Optimization
    let mut learnable_scene = Tensor::randn([N, N], true) * 0.1;
    
    let mut iter = 0;
    while iter < 200 {
        // Render from learnable scene
        let pred_image = Tensor::zeros([NUM_RAYS]);
        r = 0;
        while r < NUM_RAYS {
            let mut ray_acc_t = Tensor::zeros([1]);
            let mut s = 0;
            while s < NUM_STEPS {
                let dist = (s as f32) * step_size;
                let px = (cx as f32) + ray_dirs_x[r] * dist;
                let py = (cy as f32) + ray_dirs_y[r] * dist;
                
                let dx = grid_i - px;
                let dy = grid_j - py;
                let dist_sq = dx.pow(2) + dy.pow(2);
                let weights = (dist_sq * (-1.0 / (sigma * sigma))).exp();
                let sample = (weights * learnable_scene).sum() / (weights.sum() + epsilon);
                ray_acc_t = ray_acc_t + sample;
                
                s = s + 1;
            }
            pred_image[r] = ray_acc_t.item();
            r = r + 1;
        }
        
        // Loss: MSE between images
        let diff = pred_image - gt_image;
        let loss = diff.pow(2).sum();
        
        if (iter / 50) * 50 == iter {
            println("Iter {} Loss: {}", iter, loss.item());
        }
        
        loss.backward();
        
        let g = learnable_scene.grad();
        learnable_scene = learnable_scene - g * 0.1;
        learnable_scene = learnable_scene.detach();
        learnable_scene.enable_grad();
        
        iter = iter + 1;
    }
    
    print("Optimization Done.\n");
    print("Learned scene center value: ");
    println(learnable_scene[3, 3] + learnable_scene[4, 4]);
}
