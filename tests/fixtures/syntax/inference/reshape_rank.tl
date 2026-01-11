// STDOUT: Reshape Rank 2 OK
// STDOUT: Reshape Rank 3 OK
// STDOUT: Reshape Const Rank 2 OK

fn main() {
    let t: Tensor<f32, 1> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // 1. Reshape with TensorLiteral -> Infer Rank 2
    let t2: Tensor<f32, 2> = t.reshape([2, 3]);
    println("Reshape Rank 2 OK");

    // 2. Reshape with TensorLiteral -> Infer Rank 3
    let t3: Tensor<f32, 3> = t.reshape([1, 2, 3]);
    println("Reshape Rank 3 OK");

    // 3. Reshape with TensorConstLiteral (optimization check) -> Infer Rank 2
    // If the compiler optimizes [2, 3] to a const literal, this path is taken.
    // In current implementation, [2, 3] might be parsed as normal Literal unless optimized.
    // The previous fix handled both cases in semantics.rs.
    let t2_const: Tensor<f32, 2> = t.reshape([3, 2]);
    println("Reshape Const Rank 2 OK");
    
    // 4. Verify that assigning to wrong rank fails compilation (Negative test)
    // Uncommenting the line below should cause a type mismatch error.
    // let t_fail: Tensor<f32, 4> = t.reshape([2, 3]); 
}
