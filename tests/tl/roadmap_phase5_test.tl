// ロードマップ フェーズ5: 学習・最適化 網羅テスト
fn main() {
    // --- 5.1 損失関数 ---
    let pred = Tensor::full([4], 3.0);
    let target = Tensor::full([4], 1.0);

    // mse_loss: mean((3-1)^2) = 4.0
    let mse = Tensor::mse_loss(pred, target);
    print("mse_loss([3,3,3,3],[1,1,1,1]):");
    print(mse);

    // l1_loss: mean(|3-1|) = 2.0
    let l1 = Tensor::l1_loss(pred, target);
    print("l1_loss:");
    print(l1);

    // bce_loss: -mean(1*log(0.9)+(1-1)*log(1-0.9)) ≈ 0.105
    let p_bce = Tensor::full([4], 0.9);
    let t_bce = Tensor::full([4], 1.0);
    let bce = Tensor::bce_loss(p_bce, t_bce);
    print("bce_loss(0.9, 1.0):");
    print(bce);

    // nll_loss
    let mut p_nll_d = Vec::new();
    p_nll_d.push(-0.5);
    p_nll_d.push(-1.0);
    p_nll_d.push(-2.0);
    let mut t_nll_d = Vec::new();
    t_nll_d.push(1.0);
    t_nll_d.push(0.0);
    t_nll_d.push(0.0);
    let mut s3 = Vec::new();
    s3.push(3);
    let p_nll = Tensor::from_vec(p_nll_d, s3);
    let mut s3b = Vec::new();
    s3b.push(3);
    let t_nll = Tensor::from_vec(t_nll_d, s3b);
    let nll = Tensor::nll_loss(p_nll, t_nll);
    print("nll_loss:");
    print(nll);

    // kl_div_loss
    let p_kl = Tensor::full([4], 0.25);
    let t_kl = Tensor::full([4], 0.25);
    let kl = Tensor::kl_div_loss(p_kl, t_kl);
    print("kl_div_loss (same dist, ~0):");
    print(kl);

    // --- 5.2 dot (インスタンスメソッド) ---
    let mut da = Vec::new();
    da.push(1.0);
    da.push(2.0);
    da.push(3.0);
    let mut db = Vec::new();
    db.push(4.0);
    db.push(5.0);
    db.push(6.0);
    let mut sa = Vec::new();
    sa.push(3);
    let vec_a = Tensor::from_vec(da, sa);
    let mut sb = Vec::new();
    sb.push(3);
    let vec_b = Tensor::from_vec(db, sb);
    let dot = vec_a.dot(vec_b);
    print("dot([1,2,3],[4,5,6]) = 32:");
    print(dot);
}
