// STDOUT: MatMul result shape:
// STDOUT: Tensor[[16, 16, 16], f32,

fn main() {
    let M = 16;
    let K = 16;
    let N = 16;

    let A = Tensor::randn([M, K]);
    let B = Tensor::randn([K, N]);

    // This should trigger the new optimization path
    let C = A[i, j] * B[j, k];

    print("MatMul result shape:");
    print(C);
}
