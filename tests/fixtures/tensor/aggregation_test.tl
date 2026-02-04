// STDOUT: Aggregation test passed
tensor A: Tensor<f32, 1> = [1.0, 2.0, 3.0, 4.0, 5.0];
tensor B: Tensor<f32, 1> = [10.5, 20.0, 5.5];

fn main() {
    print(A);
    
    // Sum
    let s = A.sumall();
    print(s); // Expect 15 (1+2+3+4+5)
    
    // Mean
    let a = B.mean();
    print(a); // Expect 12.0 ((10.5+20.0+5.5)/3 = 36/3 = 12)
    
    // Max
    let m = B.max();
    print(m); // Expect 20.0
    
    print("Aggregation test passed\n");
}
