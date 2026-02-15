fn main() -> i64 {
    let M = 16;
    let K = 16;
    let N = 16;

    let A = Tensor::randn([M, K]);
    let B = Tensor::randn([K, N]);

    // This should trigger the new optimization path
    // Result is [16, 16, 16] (broadcasting/elementwise), not [16, 16] (contraction)
    let C = A[i, j] * B[j, k];

    print("Shape Rank:");
    let s = C.shape();
    print(s.len());
    if s.len() > 0 {
        print("Dim 0:");
        let opt = s.get(0);
        match opt {
            Option::Some(v) => print(v),
            Option::None => print("None"),
        }
    }
    
    // Contraction A[i, j] * B[j, k] -> C[i, k] (Rank 2)
    // Or Elementwise with broadcast?
    // Wait, indices: i,j and j,k.
    // If it's contraction, it sums over j? 
    // Einstein summation: A_ij * B_jk -> C_ik.
    // So Rank Should be 2.
    // If it was broadcasting [16, 16] * [16, 16] -> [16, 16] (Elementwise)?
    // No, indices dictate.
    // A[i, j] * B[j, k] implies outer product if not summed?
    // But TL likely infers MatMul if common index exists.
    
    if s.len() == 2 {
        print("SUCCESS: Rank 2");
    } else {
        print("FAILURE: Rank mismatch");
    }
    
    return 0;
// Output:
// STDOUT: Shape Rank:2
// STDOUT: Dim 0:16
// STDOUT: SUCCESS: Rank 2
}
