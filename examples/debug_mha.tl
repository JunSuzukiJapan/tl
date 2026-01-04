fn main() {
    print("Debugging MHA types...");
    
    // 1. Create a dummy tensor [1, 9, 128]
    // Use randn which we know works and returns Tensor<f32>
    let x = Tensor::randn([1, 9, 128], true);
    
    // 2. Test reshape
    // Target: [9, 4, 32] (flatten batch 1)
    let split = reshape(x, 9, 4, 32);
    
    // 3. Test transpose
    let heads = transpose(split, 0, 1);
    
    // 4. Test BinOp on heads
    // This panicked in the real script
    let scaled = heads * 0.176;
    
    print("Passed!");
}
