// ロードマップ フェーズ4a: 活性化関数・NN層 網羅テスト
fn main() {
    // ヘルパー
    let mut d5 = Vec::new();
    d5.push(-2.0);
    d5.push(-1.0);
    d5.push(0.0);
    d5.push(1.0);
    d5.push(2.0);
    let mut s5 = Vec::new();
    s5.push(5);
    let x = Tensor::from_vec(d5, s5);

    // --- 4.1 leaky_relu ---
    let lr = x.leaky_relu();
    print("leaky_relu:");
    print(lr);

    let lr2: f32 = 0.1;
    let lr_custom = x.leaky_relu(lr2);
    print("leaky_relu(0.1):");
    print(lr_custom);

    // --- 4.2 elu ---
    let e = x.elu();
    print("elu:");
    print(e);

    // --- 4.3 mish ---
    let m = x.mish();
    print("mish:");
    print(m);

    // --- 4.4 hardswish ---
    let mut dh = Vec::new();
    dh.push(-6.0);
    dh.push(-3.0);
    dh.push(0.0);
    dh.push(3.0);
    dh.push(6.0);
    let mut sh = Vec::new();
    sh.push(5);
    let xh = Tensor::from_vec(dh, sh);
    let hs = xh.hardswish();
    print("hardswish([-6,-3,0,3,6]):");
    print(hs);

    // --- 4.5 hardsigmoid ---
    let hsig = xh.hardsigmoid();
    print("hardsigmoid([-6,-3,0,3,6]):");
    print(hsig);

    // --- 4.6 linear ---
    let input = Tensor::full([1, 3], 1.0);
    let weight = Tensor::full([3, 2], 0.5);
    let bias = Tensor::full([2], 0.1);
    let lin = input.linear(weight, bias);
    print("linear:");
    print(lin);

    // --- 4.7 pad ---
    let mut dp = Vec::new();
    dp.push(1.0);
    dp.push(2.0);
    dp.push(3.0);
    let mut sp = Vec::new();
    sp.push(3);
    let t_pad = Tensor::from_vec(dp, sp);
    let padded = t_pad.pad(2, 1);
    print("pad(2,1):");
    print(padded);

    // --- 4.8 max_pool2d ---
    let pool_input = Tensor::full([1, 1, 4, 4], 1.0);
    let mp = pool_input.max_pool2d(2, 2, 0);
    print("max_pool2d(k=2,s=2,p=0):");
    print(mp);

    // --- 4.9 avg_pool2d ---
    let ap = pool_input.avg_pool2d(2, 2, 0);
    print("avg_pool2d(k=2,s=2,p=0):");
    print(ap);

    // --- 4.10 adaptive_avg_pool2d ---
    let aap = pool_input.adaptive_avg_pool2d(2, 2);
    print("adaptive_avg_pool2d(2,2):");
    print(aap);

    // --- 4.11 conv1d ---
    let c1d_input = Tensor::full([1, 1, 5], 1.0);
    let c1d_w = Tensor::full([1, 1, 3], 1.0);
    let c1d_b = Tensor::full([1], 0.0);
    let c1d = c1d_input.conv1d(c1d_w, c1d_b, 1, 0);
    print("conv1d:");
    print(c1d);

    // --- 4.12 interpolate ---
    let i_input = Tensor::full([1, 1, 2, 2], 1.0);
    let interp = i_input.interpolate(4, 4, 0);
    print("interpolate(2x2 -> 4x4, nearest):");
    print(interp);

    // --- 4.13 softmax ---
    let sf_input = Tensor::full([4], 1.0);
    let sf = sf_input.softmax(0);
    print("softmax([1,1,1,1]):");
    print(sf);

    // --- 4.14 relu / sigmoid / tanh ---
    let act_input = Tensor::full([3], -1.0);
    let r = act_input.relu();
    print("relu([-1,-1,-1]):");
    print(r);
    let sigm = act_input.sigmoid();
    print("sigmoid:");
    print(sigm);
    let th = act_input.tanh();
    print("tanh:");
    print(th);
}
