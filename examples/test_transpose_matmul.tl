// 厳密にreverse_train.tlのパターンを再現

fn main() {
    println("Reproducing exact MHA pattern from reverse_train.tl");
    
    // MHAの正確なパターン
    let Q_orig = Tensor::randn([1, 9, 128], false);
    let K_orig = Tensor::randn([1, 9, 128], false);
    
    // reshape: [1, 9, 128] -> [9, 4, 32]
    let Q_split = reshape(Q_orig, 9, 4, 32);
    let K_split = reshape(K_orig, 9, 4, 32);
    
    println("Q_split shape: [9, 4, 32]");
    println("K_split shape: [9, 4, 32]");
    
    // transpose: [9, 4, 32] -> [4, 9, 32]
    let Q_heads = transpose(Q_split, 0, 1);
    let K_heads = transpose(K_split, 0, 1);
    
    println("Q_heads shape after transpose(0,1): [4, 9, 32]");
    println("K_heads shape after transpose(0,1): [4, 9, 32]");
    
    // transpose: [4, 9, 32] -> [4, 32, 9]
    let K_heads_T = transpose(K_heads, 1, 2);
    
    println("K_heads_T shape after transpose(1,2): [4, 32, 9]");
    
    // Scale
    let K_scaled = K_heads_T * 0.176;
    
    println("K_scaled shape: [4, 32, 9]");
    
    // TEST: contiguous()を使って修正
    println("Applying contiguous() to Q_heads...");
    let Q_cont = Q_heads.contiguous();
    
    println("About to call matmul(Q_cont, K_scaled)...");
    let logits = matmul(Q_cont, K_scaled);
    
    println("logits shape: [4, 9, 9]");
    println("Test completed successfully!");
}
