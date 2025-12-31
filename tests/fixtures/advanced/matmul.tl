// STDOUT: MatMul result shape:
// STDOUT: Tensor[[256, 256], f32]

fn main() {
    let M = 256;
    let K = 256;
    let N = 256;

    let A = randn([M, K]);
    let B = randn([K, N]);

    // This should trigger the new optimization path
    let C[i, k] = A[i, j] * B[j, k];

    print("MatMul result shape:");
    print(C);
}
