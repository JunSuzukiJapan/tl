// ロードマップ フェーズ1: テンソル生成 網羅テスト
fn main() {
    // --- 1.1 full ---
    let t_full = Tensor::full([2, 3], 3.14);
    print("full([2,3], 3.14):");
    print(t_full);

    // --- 1.2 eye ---
    let t_eye = Tensor::eye(3);
    print("eye(3):");
    print(t_eye);

    // --- 1.3 arange ---
    let t_arange = Tensor::arange(0.0, 5.0, 1.0);
    print("arange(0,5,1):");
    print(t_arange);

    let t_arange2 = Tensor::arange(1.0, 10.0, 2.5);
    print("arange(1,10,2.5):");
    print(t_arange2);

    // --- 1.4 from_vec ---
    let mut data = Vec::new();
    data.push(10.0);
    data.push(20.0);
    data.push(30.0);
    data.push(40.0);
    data.push(50.0);
    data.push(60.0);
    let mut shape = Vec::new();
    shape.push(2);
    shape.push(3);
    let t_fv = Tensor::from_vec(data, shape);
    print("from_vec([10..60], [2,3]):");
    print(t_fv);

    // --- 1.5 zeros_like / ones_like ---
    let src = Tensor::full([2, 4], 99.0);
    let zl = Tensor::zeros_like(src);
    print("zeros_like:");
    print(zl);
    let ol = Tensor::ones_like(src);
    print("ones_like:");
    print(ol);

    // --- 1.6 linspace ---
    let t_ls = Tensor::linspace(0.0, 1.0, 5);
    print("linspace(0,1,5):");
    print(t_ls);

    let t_ls2 = Tensor::linspace(-1.0, 1.0, 3);
    print("linspace(-1,1,3):");
    print(t_ls2);

    // --- 1.7 rand ---
    let t_rand = Tensor::rand([2, 3]);
    print("rand([2,3]):");
    print(t_rand);

    // --- 1.8 rand_like / randn_like ---
    let rl = Tensor::rand_like(src);
    print("rand_like:");
    print(rl);

    let rnl = Tensor::randn_like(src);
    print("randn_like:");
    print(rnl);
}
