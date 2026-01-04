// Simple VarBuilder test without String type

fn main() {
    print("Test: VarBuilder functions");
    
    // Test varbuilder_get - create parameter tensors
    let weight = VarBuilder::get("test_weight", [10, 5]);
    let bias = VarBuilder::get("test_bias", [5]);
    
    print("Created parameters");
    print(weight);
    print(bias);
    
    // Simple computation
    let x = Tensor::randn([4, 10], false);
    let output = matmul(x, weight) + bias;
    let loss = sum(output);
    
    print("Loss:");
    print(loss);
    
    // Backward
    backward(loss);
    
    // Get gradients
    let gw = VarBuilder::grad("test_weight");
    let gb = VarBuilder::grad("test_bias");
    
    print("Weight gradient:");
    print(gw);
    print("Bias gradient:");
    print(gb);
    
    // Update parameters
    update_all_params(0.01);
    
    print("Parameters updated!");
}
