// Test tensor operations
// STDOUT: Library test complete

tensor A: Tensor<f32, 1> = [1.0, 2.0, 3.0, 4.0];
tensor B: Tensor<f32, 1> = [2.0, 2.0, 2.0, 2.0];
tensor M1: Tensor<f32, 1> = [1.0, 2.0, 3.0, 4.0];
tensor M2: Tensor<f32, 1> = [1.0, 0.0, 0.0, 1.0];

fn main() {
    // Test Div
    let C = A / B; // Should be [0.5, 1.0, 1.5, 2.0]
    print(C);

    // Test Exp
    let E = B.exp(); // [7.389, ...]
    print(E);

    // Test Log
    let L = E.log(); // [2.0, ...]
    print(L);

    // Test Sqrt
    let S = B.sqrt(); // [1.414, ...]
    print(S);

    // Test MatMul
    let M1_res = M1.reshape([2, 2]);
    let M2_res = M2.reshape([2, 2]);

    let M3 = M1_res.matmul(M2_res); // Should be same as M1
    print(M3);
    print("Library test complete\n");
}
