fn main() {
    // 1. Integer Array Literal -> Tensor<i64, 1>
    let i_arr: Tensor<i64, 1> = [1, 2, 3];
    println("Int Array OK");

    // 2. Float Array Literal -> Tensor<f32, 1>
    let f_arr: Tensor<f32, 1> = [1.0, 2.0, 3.0];
    println("Float Array OK");

    // 3. Mixed Array Literal -> Tensor<f32, 1> (Implicit cast to float)
    let m_arr: Tensor<f32, 1> = [1, 2.0, 3];
    println("Mixed Array OK");

    // 4. Nested Array Literal (2D)
    // Currently parser might parse [[1, 2], [3, 4]] as TensorLiteral containing TensorLiterals? 
    // Or maybe flattened?
    // Let's stick to 1D for now as that was the main fix target.
}

// STDOUT: Int Array OK
// STDOUT: Float Array OK
// STDOUT: Mixed Array OK
