fn main() {
    let mut logits_data = Vec::new();
    logits_data.push(1.0);
    logits_data.push(5.0);
    logits_data.push(2.0);
    logits_data.push(8.0);
    logits_data.push(3.0);
    let mut s = Vec::new();
    s.push(5);
    let logits = Tensor::from_vec(logits_data, s);

    let topk = logits.top_k_sample(2);
    print("top_k(2):");
    print(topk);

    let p: f64 = 0.9;
    let topp = logits.top_p_sample(p);
    print("top_p(0.9):");
    print(topp);
}
