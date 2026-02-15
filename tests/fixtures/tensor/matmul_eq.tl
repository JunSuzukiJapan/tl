fn main() {
    let N = 16;
    let A = Tensor::randn([N, N], false);
    let B = Tensor::randn([N, N], false);
    
    // Explicit MatMul via equation
    // This should trigger the new optimization path
    let C = A[i, j] * B[j, k];

    print("MatMul result shape:");
    print(C);
    
    let D = A.matmul(B);
    let diff = C - D;
    let s = diff.sumall();
    // Should be close to 0
    print("Difference sum (should be approx 0):");
    print(s);
}
