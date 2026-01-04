fn main() {
    print("Testing Save/Load...");
    // Create random tensor
    let t = Tensor::randn([2, 2], false);
    print("Original:");
    print(t);
    
    // Save
    print("Saving to 'test_tensor.safetensors'...");
    save_weights(t, "test_tensor.safetensors");
    
    // Load
    print("Loading...");
    let t2 = load_weights("test_tensor.safetensors");
    print("Loaded:");
    print(t2);
    
    // Verify math works on loaded tensor
    let t3 = t2 + 1.0;
    print("Loaded + 1.0:");
    print(t3);
}
