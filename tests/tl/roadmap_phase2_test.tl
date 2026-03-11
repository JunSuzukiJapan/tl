// ロードマップ フェーズ2: 要素操作・リダクション 網羅テスト
fn main() {
    // ヘルパー: Vec -> Tensor 作成
    let mut d4 = Vec::new();
    d4.push(1.0);
    d4.push(2.0);
    d4.push(3.0);
    d4.push(4.0);
    let mut s4 = Vec::new();
    s4.push(4);
    let x = Tensor::from_vec(d4, s4);

    // --- 2.1 where_cond ---
    let mut cd = Vec::new();
    cd.push(1.0);
    cd.push(0.0);
    cd.push(1.0);
    cd.push(0.0);
    let mut cs = Vec::new();
    cs.push(4);
    let cond = Tensor::from_vec(cd, cs);
    let a_wc = Tensor::full([4], 10.0);
    let b_wc = Tensor::full([4], -1.0);
    let wc = Tensor::where_cond(cond, a_wc, b_wc);
    print("where_cond:");
    print(wc);

    // --- 2.2 masked_fill ---
    let mut md = Vec::new();
    md.push(1.0);
    md.push(0.0);
    md.push(1.0);
    md.push(0.0);
    let mut ms = Vec::new();
    ms.push(2);
    ms.push(2);
    let mask = Tensor::from_vec(md, ms);
    let t_mf = Tensor::full([2, 2], 5.0);
    let filled = t_mf.masked_fill(mask, -999.0);
    print("masked_fill:");
    print(filled);

    // --- 2.3 var / std ---
    print("var:");
    let v = x.var(0);
    print(v);
    print("std:");
    let sd = x.std(0);
    print(sd);

    // --- 2.4 logical_and / logical_or / logical_not ---
    let mut la = Vec::new();
    la.push(1.0);
    la.push(0.0);
    la.push(1.0);
    la.push(0.0);
    let mut lb = Vec::new();
    lb.push(1.0);
    lb.push(1.0);
    lb.push(0.0);
    lb.push(0.0);
    let mut ls1 = Vec::new();
    ls1.push(4);
    let t_a = Tensor::from_vec(la, ls1);
    let mut ls2 = Vec::new();
    ls2.push(4);
    let t_b = Tensor::from_vec(lb, ls2);
    let land = Tensor::logical_and(t_a, t_b);
    print("logical_and:");
    print(land);
    let lor = Tensor::logical_or(t_a, t_b);
    print("logical_or:");
    print(lor);
    let lnot = t_a.logical_not();
    print("logical_not:");
    print(lnot);

    // --- 2.5 fill_ ---
    let t_fill = Tensor::full([3], 1.0);
    t_fill.fill_(42.0);
    print("fill_(42.0):");
    print(t_fill);

    // --- 2.6 prod ---
    let mut dp = Vec::new();
    dp.push(2.0);
    dp.push(3.0);
    dp.push(4.0);
    let mut sp = Vec::new();
    sp.push(3);
    let t_prod = Tensor::from_vec(dp, sp);
    let p = t_prod.prod();
    print("prod([2,3,4]):");
    print(p);

    // --- 2.7 cumsum ---
    let cs_result = x.cumsum(0);
    print("cumsum([1,2,3,4]):");
    print(cs_result);

    // --- 2.8 norm ---
    let mut dn = Vec::new();
    dn.push(3.0);
    dn.push(4.0);
    let mut sn = Vec::new();
    sn.push(2);
    let t_norm = Tensor::from_vec(dn, sn);
    let n = t_norm.norm(2.0);
    print("norm([3,4], p=2):");
    print(n);

    // --- 2.9 topk ---
    let mut dt = Vec::new();
    dt.push(5.0);
    dt.push(1.0);
    dt.push(9.0);
    dt.push(3.0);
    dt.push(7.0);
    let mut st = Vec::new();
    st.push(5);
    let t_tk = Tensor::from_vec(dt, st);
    let tk = t_tk.topk(3);
    print("topk([5,1,9,3,7], k=3):");
    print(tk);

    // --- 2.10 dot ---
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
