fn main() {
    let N = 256;
    let A = randn([N, N]);
    let B = randn([N, N]);
    
    // Explicit MatMul via equation
    // This should trigger the new optimization path
    let C[i, k] = A[i, j] * B[j, k];

    // print("MatMul result shape:");
    // print(C);
    
    let D = matmul(A, B);
    let diff = C - D;
    let s = sum(diff);
    // Should be close to 0
    // print("Difference sum (should be approx 0):");
    // print(s);
}
