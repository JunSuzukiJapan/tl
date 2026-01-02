fn main() {
    // Define tensors using let inside function
    let A: Tensor<f32, 2> = [[1.0, 2.0], [3.0, 4.0]];
    
    // Transpose (0, 1) -> [[1.0, 3.0], [2.0, 4.0]]
    let T = transpose(A, 0, 1);
    print(T);

    // Reshape
    let V: Tensor<f32, 1> = [1.0, 2.0, 3.0, 4.0];
    let S: Tensor<f32, 1> = [2.0, 2.0];
    let M = reshape(V, S);
    print(M);
}
