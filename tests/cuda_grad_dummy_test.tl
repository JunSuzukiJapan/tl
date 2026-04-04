fn main() {
    print("=== Testing LayerNorm ===");
    let x_ln = Tensor::randn([2, 5], true);
    let mut x_train_ln = x_ln.detach(true);
    let g = Tensor::randn([5], true);
    let mut g_train = g.detach(true);
    let b = Tensor::randn([5], true);
    let mut b_train = b.detach(true);

    let y_ln = x_train_ln.layer_norm(g_train, b_train, 0.00001 as f64);
    let target = Tensor::randn([2, 5], false);
    let loss_ln = (y_ln * target).sum();
    loss_ln.backward();
    
    print("LayerNorm grad_x magnitude (should be > 0):");
    let mut gx = x_train_ln.grad();
    (gx * gx).sum().print();
    print("LayerNorm grad_gamma magnitude (should be > 0):");
    let mut gg = g_train.grad();
    (gg * gg).sum().print();
    print("LayerNorm grad_beta magnitude (should be > 0):");
    let mut gb = b_train.grad();
    (gb * gb).sum().print();

    print("=== Testing Dropout ===");
    let x_dp = Tensor::randn([2, 5], true);
    let mut x_train_dp = x_dp.detach(true);
    let y_dp = x_train_dp.dropout(0.5 as f64, true);
    let target_dp = Tensor::randn([2, 5], false);
    let loss_dp = (y_dp * target_dp).sum();
    loss_dp.backward();
    
    print("Dropout grad_x magnitude (should be > 0):");
    let mut gdp = x_train_dp.grad();
    (gdp * gdp).sum().print();
}
