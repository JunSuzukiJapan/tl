// ロードマップ フェーズ8: 線形代数 網羅テスト (全メソッド TypeManager 登録済み)
fn main() {
    // --- 8.1 dot ---
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
    let a = Tensor::from_vec(da, sa);
    let mut sb = Vec::new();
    sb.push(3);
    let b = Tensor::from_vec(db, sb);
    let dot = a.dot(b);
    print("dot([1,2,3],[4,5,6]) = 32:");
    print(dot);

    // --- 8.2 matmul ---
    let m1 = Tensor::eye(3);
    let m2 = Tensor::full([3, 3], 2.0);
    let mm = m1.matmul(m2);
    print("eye(3).matmul(full(3,3,2.0)):");
    print(mm);

    // --- 8.3 inverse ---
    let mat = Tensor::eye(3);
    let inv = mat.inverse();
    print("inverse(eye(3)):");
    print(inv);

    // --- 8.4 det ---
    let det = mat.det();
    print("det(eye(3)) = 1:");
    print(det);

    // --- 8.5 svd ---
    let u = mat.svd_u();
    print("svd_u(eye(3)):");
    print(u);
    let s = mat.svd_s();
    print("svd_s(eye(3)):");
    print(s);
    let v = mat.svd_v();
    print("svd_v(eye(3)):");
    print(v);

    // --- 8.6 eig ---
    let evals = mat.eig_values();
    print("eig_values(eye(3)):");
    print(evals);
    let evecs = mat.eig_vectors();
    print("eig_vectors(eye(3)):");
    print(evecs);

    // --- 8.7 solve ---
    // A * x = b  where A = eye(3), b = [1,2,3]  →  x = [1,2,3]
    let mut bd = Vec::new();
    bd.push(1.0);
    bd.push(2.0);
    bd.push(3.0);
    let mut bs = Vec::new();
    bs.push(3);
    bs.push(1);
    let b_solve = Tensor::from_vec(bd, bs);
    let x = mat.solve(b_solve);
    print("solve(eye(3), [1,2,3]) = [1,2,3]:");
    print(x);
}
