// ロードマップ フェーズ4b: 正規化層テスト (layer_norm, batch_norm, group_norm, instance_norm)
fn main() {
    // --- layer_norm (eps バグ修正済み) ---
    let ln_input = Tensor::full([1, 4], 2.0);
    let ln_w = Tensor::full([4], 1.0);
    let ln_b = Tensor::full([4], 0.0);
    let ln_eps: f64 = 0.00001;
    let ln = ln_input.layer_norm(ln_w, ln_b, ln_eps);
    print("layer_norm:");
    print(ln);

    // --- batch_norm ---
    let bn_input = Tensor::full([1, 2, 2, 2], 1.0);
    let bn_rm = Tensor::full([2], 0.0);
    let bn_rv = Tensor::full([2], 1.0);
    let bn_w = Tensor::full([2], 1.0);
    let bn_b = Tensor::full([2], 0.0);
    let bn = bn_input.batch_norm(bn_rm, bn_rv, bn_w, bn_b, false);
    print("batch_norm:");
    print(bn);

    // --- group_norm (eps バグ修正済み) ---
    let gn_input = Tensor::full([1, 4, 2, 2], 1.0);
    let gn_w = Tensor::full([4], 1.0);
    let gn_b = Tensor::full([4], 0.0);
    let gn_eps: f64 = 0.00001;
    let gn = gn_input.group_norm(2, gn_w, gn_b, gn_eps);
    print("group_norm:");
    print(gn);

    // --- instance_norm (新規登録) ---
    let in_input = Tensor::full([1, 2, 4, 4], 2.0);
    let in_w = Tensor::full([2], 1.0);
    let in_b = Tensor::full([2], 0.0);
    let in_eps: f64 = 0.00001;
    let in_result = in_input.instance_norm(in_w, in_b, in_eps);
    print("instance_norm:");
    print(in_result);
}
