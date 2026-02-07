// Minimal crash repro - step by step
tensor A: Tensor<f32, 1> = [1.0, 2.0, 3.0, 4.0];
tensor B: Tensor<f32, 1> = [2.0, 2.0, 2.0, 2.0];
tensor M2: Tensor<f32, 1> = [1.0, 0.0, 0.0, 1.0];

fn main() {
    // Test 1: div
    let C = A / B;
    print(C);
    print("\n--- test1 done ---\n");
    
    // Test 2: exp
    let E = B.exp();
    print(E);
    print("\n--- test2 done ---\n");
    
    // Test 3: sqrt
    let S = B.sqrt();
    print(S);
    print("\n--- test3 done ---\n");
    
    // Test 4: reshape
    let M1 = A.reshape([2, 2]);
    print("\n--- test4 (reshape) done ---\n");
    
    // Test 5: matmul
    let M2_res = M2.reshape([2, 2]);
    let M3 = M1.matmul(M2_res);
    print(M3);
    print("\n--- test5 (matmul) done ---\n");
    
    print("ALL DONE\n");
}
