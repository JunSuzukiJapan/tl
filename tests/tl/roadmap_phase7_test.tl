// ロードマップ フェーズ7: LLM/Transformer 推論 網羅テスト
fn main() {
    // --- 7.1 top_k_sample ---
    let mut ld = Vec::new();
    ld.push(1.0);
    ld.push(5.0);
    ld.push(2.0);
    ld.push(8.0);
    ld.push(3.0);
    let mut ls = Vec::new();
    ls.push(5);
    let logits = Tensor::from_vec(ld, ls);
    let topk = logits.top_k_sample(2);
    print("top_k_sample(2):");
    print(topk);

    // --- 7.2 top_p_sample ---
    let p: f64 = 0.9;
    let topp = logits.top_p_sample(p);
    print("top_p_sample(0.9):");
    print(topp);

    // --- 7.3 temperature_scale ---
    let temp: f64 = 2.0;
    let scaled = logits.temperature_scale(temp);
    print("temperature_scale(2.0):");
    print(scaled);

    // --- 7.4 repetition_penalty ---
    let mut td = Vec::new();
    td.push(0.0);
    td.push(3.0);
    let mut ts = Vec::new();
    ts.push(2);
    let tokens = Tensor::from_vec(td, ts);
    let pen: f64 = 1.5;
    let penalized = logits.repetition_penalty(tokens, pen);
    print("repetition_penalty(1.5):");
    print(penalized);

    // --- 7.5 scaled_dot_product_attention ---
    let q = Tensor::full([1, 1, 2, 4], 1.0);
    let k = Tensor::full([1, 1, 2, 4], 1.0);
    let v = Tensor::full([1, 1, 2, 4], 1.0);
    let attn = Tensor::sdpa(q, k, v);
    print("sdpa(Q,K,V):");
    print(attn);
}
