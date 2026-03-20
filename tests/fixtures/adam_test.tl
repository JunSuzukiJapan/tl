// EXIT: 0
// STDOUT: 0.9990
// STDOUT: 0.9980
// STDOUT: 1.0010
// STDOUT: 1.0020

extern fn tl_adam_step(
    w: Tensor<f32, 2>,
    grad: Tensor<f32, 2>,
    m: Tensor<f32, 2>,
    v: Tensor<f32, 2>,
    step: i64,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32
);

fn main() {
    println("TEST START");
    // 1. Initialize weights, gradients, m, v
    // Weight
    let mut w1 = Tensor::ones([2, 2], true);  // All 1.0
    // Gradients (fake accumulation)
    let g1 = Tensor::ones([2, 2], false); // Grads are 1.0
    
    // Optimizer States
    let m1 = Tensor::zeros([2, 2], false);
    let v1 = Tensor::zeros([2, 2], false);

    // Hyperparameters
    let lr = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let weight_decay = 0.0;
    
    let mut step: i64 = 0;

    // Adam step 1
    tl_adam_step(
        w1,
        g1,
        m1,
        v1,
        step,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay
    );

    // Initial value was 1.0. Gradient is 1.0.
    // m = 0.9 * 0 + 0.1 * 1.0 = 0.1
    // v = 0.999 * 0 + 0.001 * 1.0 = 0.001
    // m_hat = 0.1 / (1 - 0.9) = 0.1 / 0.1 = 1.0
    // v_hat = 0.001 / (1 - 0.999) = 0.001 / 0.001 = 1.0
    // update = lr * m_hat / (sqrt(v_hat) + eps) = 0.001 * 1.0 / 1.0 = 0.001
    // new_w1 = 1.0 - 0.001 = 0.9990
    println(w1[0, 0]);

    // Adam step 2
    step = 1;
    let g2 = Tensor::ones([2, 2], false); // Grad is 1.0 again
    
    tl_adam_step(
        w1,
        g2,
        m1, // m1 was mutated in place functionally
        v1,
        step,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay
    );

    // Expect weight to go down another ~0.001 -> 0.998
    println(w1[0, 0]);


    // Negative Gradient Test
    // Weight starts at 1.0
    let mut w2 = Tensor::ones([2, 2], true);
    let g_neg = Tensor::ones([2, 2], false) * -1.0; // Grads are -1.0
    let m2 = Tensor::zeros([2, 2], false);
    let v2 = Tensor::zeros([2, 2], false);

    // Adam step 1 (negative)
    step = 0;
    tl_adam_step(
        w2,
        g_neg,
        m2,
        v2,
        step,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay
    );

    // Expect weight to go UP by 0.001 -> 1.001
    println(w2[0, 0]);

    // Adam step 2 (negative)
    let g_neg2 = Tensor::ones([2, 2], false) * -1.0;
    step = 1;
    tl_adam_step(
        w2,
        g_neg2,
        m2,
        v2,
        step,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay
    );

    // Expect weight to go UP by ~0.001 -> 1.002
    println(w2[0, 0]);
}
