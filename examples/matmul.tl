fn main() {
    let A = [[1.0, 2.0], [3.0, 4.0]];
    let B = [[5.0, 6.0], [7.0, 8.0]];
    
    // MatMul: 2x2 * 2x2 = 2x2
    // C[i, k] = sum(A[i, j] * B[j, k])
    let C[i, k] = A[i, j] * B[j, k];
    
    print(C);
}
