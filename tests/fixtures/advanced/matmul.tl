// STDOUT: MatMul result shape:
// STDOUT: Tensor(shape=[16, 16, 16],

fn main() -> i64 {
    let M = 16;
    let K = 16;
    let N = 16;

    let A = Tensor::randn([M, K]);
    let B = Tensor::randn([K, N]);

    // This should trigger the new optimization path
    // Result is [16, 16, 16] (broadcasting/elementwise), not [16, 16] (contraction)
    let C = A[i, j] * B[j, k];

    print("MatMul result shape:");
    print(C);
    
    return 0;
}
