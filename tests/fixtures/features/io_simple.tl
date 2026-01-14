fn main() {
    print("Testing Save/Load...");
    // Create random tensor
    let t = Tensor::randn([2, 2], false);
    print("Original:");
    print(t);
    
    // Save
    print("Saving to 'test_tensor.safetensors'...");
    // Tensor save usage: Param::save works for Structs, does it work for single Tensor?
    // If not, we might need a wrapper struct or use lower level API.
    // Based on previous file logic, Param::save might be the generalized one.
    // Let's assume Param::save works for any Serializable item, including Tensor.
    Param::save(t, "test_tensor.safetensors");
    
    // Load
    print("Loading...");
    // Tensor load: Tensor::load(path)
    let t2 = Tensor::load("test_tensor.safetensors");
    print("Loaded:");
    print(t2);
    
    // Verify math works on loaded tensor
    let t3 = t2 + 1.0;
    print("Loaded + 1.0:");
    print(t3);
}
