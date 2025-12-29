fn main() {
    let A: Tensor<f32, 2> = [[1.0, 2.0], [3.0, 4.0]];
    let S: Tensor<f32, 2> = [[10.0, 20.0]];
    
    // Broadcasting add
    // [[1, 2], [3, 4]] + [10, 20] -> [[11, 22], [13, 24]]
    let D = A + S;
    print(D);
}
